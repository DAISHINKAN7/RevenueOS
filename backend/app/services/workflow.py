"""Recovery workflow orchestration.

Sequence enforced here, and nowhere else:

    ML predicts -> financial engine calculates -> policy authorizes
    -> executor executes -> webhook verifies -> audit records

Two invariants the code defends explicitly:

1. No financial side effect happens without a legal state transition and a
   durable execution row committed in the same transaction.
2. The idempotency key is unique in the database, so concurrent execute calls
   collapse to one external order rather than two.
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from backend.app.core.config import (
    FINANCIAL_ENGINE_VERSION, MerchantPolicy, WORKFLOW_VERSION, settings,
)
from backend.app.db.models import (
    ActionFinancialEvaluation, ActionPrediction, AuditEvent, Opportunity,
    PaymentFailureRecord, PolicyEvaluation, PolicyRuleEvaluation,
    RecoveryExecution, RecoveryOutcome, utcnow,
)
from backend.app.domain import (
    DuplicateExecution, InvalidStateTransition, State, can_transition,
)
from backend.app.services.policy_engine import (
    ActionEconomics, Decision, PolicyDecision, PolicyEngine,
)
from backend.app.services.predictor import RecoveryPredictor, get_predictor
from ml.actions import Action, spec_for
from ml.financial_engine import OpportunityEconomics, valuate_action

log = logging.getLogger("revenueos.workflow")
Q = Decimal("0.01")


def money(x) -> Decimal:
    return Decimal(str(float(x))).quantize(Q, rounding=ROUND_HALF_UP)


def new_trace_id() -> str:
    return f"trc_{uuid.uuid4().hex[:16]}"


def idempotency_key(opportunity_id: str, action: str, attempt: int) -> str:
    """Stable across retries of the same attempt, different across attempts."""
    raw = f"{opportunity_id}|{action}|{attempt}|{WORKFLOW_VERSION}"
    return hashlib.sha256(raw.encode()).hexdigest()


# --------------------------------------------------------------- eligibility
def eligible_actions(ctx: dict) -> list[Action]:
    """Eligibility is *not* authorization — it only asks whether the action concept
    makes sense for this opportunity."""
    is_failure = ctx.get("opportunity_type") == "PAYMENT_FAILURE"
    fee = float(ctx.get("shipping_fee_charged") or 0)
    cart = float(ctx.get("cart_value") or 0)
    margin = float(ctx.get("base_contribution_margin") or 0)
    reason = ctx.get("failure_reason") or "NO_PAYMENT_FAILURE"

    out = [Action.DO_NOTHING, Action.HUMAN_ESCALATION]
    if fee > 0:
        out.append(Action.FREE_SHIPPING)
    if cart > 0 and margin > 0:
        out += [Action.SMALL_DISCOUNT, Action.MEDIUM_DISCOUNT]
    if is_failure:
        out.append(Action.PAYMENT_METHOD_SWITCH)
        out.append(Action.DELAYED_RETRY)
        if reason not in ("USER_CANCELLED",):
            out.append(Action.IMMEDIATE_RETRY)
    out.append(Action.PAYMENT_LINK)
    return list(dict.fromkeys(out))


# -------------------------------------------------------------------- audit
class AuditRecorder:
    """Append-only audit writer. Corrections append AUDIT_CORRECTION rows."""

    REDACT = ("key_secret", "webhook_secret", "api_key", "authorization", "signature")

    def __init__(self, session, opportunity: Opportunity):
        self.s = session
        self.opp = opportunity

    def _next_seq(self) -> int:
        rows = self.s.execute(
            select(AuditEvent.sequence_number)
            .where(AuditEvent.opportunity_id == self.opp.id)
        ).scalars().all()
        return (max(rows) + 1) if rows else 1

    @classmethod
    def redact(cls, payload: dict) -> dict:
        out = {}
        for k, v in (payload or {}).items():
            if any(s in k.lower() for s in cls.REDACT):
                out[k] = "[REDACTED]"
            elif isinstance(v, dict):
                out[k] = cls.redact(v)
            elif isinstance(v, Decimal):
                out[k] = float(v)
            else:
                out[k] = v
        return out

    def record(self, event_type: str, summary: str, payload: dict | None = None,
               state_before: str | None = None, state_after: str | None = None,
               actor_type: str = "SYSTEM", actor_id: str = "revenueos",
               model_version: str | None = None, policy_version: str | None = None,
               execution_id: str | None = None) -> AuditEvent:
        ev = AuditEvent(
            trace_id=self.opp.trace_id, opportunity_id=self.opp.id,
            sequence_number=self._next_seq(), event_type=event_type,
            actor_type=actor_type, actor_id=actor_id, timestamp=utcnow(),
            workflow_state_before=state_before, workflow_state_after=state_after,
            summary=summary, structured_payload=self.redact(payload or {}),
            model_version=model_version, policy_version=policy_version,
            execution_id=execution_id,
        )
        self.s.add(ev)
        self.s.flush()
        return ev


def transition(session, opp: Opportunity, dst: State, audit: AuditRecorder,
               event_type: str, summary: str, payload: dict | None = None) -> None:
    src = State(opp.state)
    if not can_transition(src, dst):
        raise InvalidStateTransition(
            f"illegal transition {src.value} -> {dst.value}",
            opportunity_id=opp.id, source=src.value, target=dst.value)
    opp.state = dst.value
    opp.updated_at = utcnow()
    audit.record(event_type, summary, payload, state_before=src.value, state_after=dst.value)


# ---------------------------------------------------------------- analysis
@dataclass
class CandidateResult:
    action: str
    probability: float | None
    valid: bool
    expected_value: Decimal
    incremental_expected_value: Decimal
    incentive_cost: Decimal
    fixed_cost: Decimal
    base_margin: Decimal
    expected_return_loss: Decimal
    expected_cancellation_loss: Decimal
    error: str | None = None

    def as_dict(self) -> dict:
        return {
            "action": self.action, "probability": self.probability,
            "expected_value": float(self.expected_value),
            "incremental_expected_value": float(self.incremental_expected_value),
            "incentive_cost_if_recovered": float(self.incentive_cost),
            "fixed_action_cost": float(self.fixed_cost),
            "valid": self.valid, "error": self.error,
        }


class RecoveryWorkflow:
    def __init__(self, session, predictor: RecoveryPredictor | None = None,
                 policy: MerchantPolicy | None = None):
        self.s = session
        self.predictor = predictor or get_predictor()
        self.policy_engine = PolicyEngine(policy)

    # ---------------------------------------------------------------- helpers
    def _econ(self, ctx: dict) -> OpportunityEconomics:
        return OpportunityEconomics(
            cart_value=float(ctx["cart_value"]),
            cogs=float(ctx["cart_cogs"]),
            shipping_cost=float(ctx["shipping_cost"]),
            shipping_fee_charged=float(ctx["shipping_fee_charged"]),
            return_probability=float(ctx.get("return_probability", 0.0)),
            cancellation_probability=float(ctx.get("cancellation_probability", 0.0)),
        )

    def _existing_keys(self, opportunity_id: str) -> set[str]:
        return set(self.s.execute(
            select(RecoveryExecution.idempotency_key)
            .where(RecoveryExecution.opportunity_id == opportunity_id)
        ).scalars().all())

    # ---------------------------------------------------------------- analyze
    def analyze(self, opp: Opportunity) -> dict:
        audit = AuditRecorder(self.s, opp)
        ctx = dict(opp.context)
        attempt = opp.current_attempt

        RETRY_STATES = (State.PAYMENT_FAILED_RECOVERABLE, State.NOT_RECOVERED,
                        State.EXECUTION_FAILED)
        if State(opp.state) in RETRY_STATES:
            # A retry is a NEW attempt. Without incrementing here the idempotency
            # key would be identical to the failed attempt's, and
            # RULE_DUPLICATE_ACTION_PREVENTION would stop the workflow — the
            # recovery loop would be unable to make a second attempt at all.
            opp.current_attempt += 1
            attempt = opp.current_attempt
            self.s.flush()
            transition(self.s, opp, State.ANALYZING, audit, "ANALYSIS_STARTED",
                       f"re-analysis after failed attempt {attempt - 1}; "
                       f"starting attempt {attempt}",
                       {"previous_attempt": attempt - 1, "attempt": attempt})
        else:
            transition(self.s, opp, State.ANALYZING, audit, "ANALYSIS_STARTED",
                       f"analysis started for attempt {attempt}")

        actions = eligible_actions(ctx)
        econ_base = self._econ(ctx)

        # --- score ---------------------------------------------------------
        preds = self.predictor.score_candidate_actions(ctx, [a.value for a in actions])
        for p in preds:
            self.s.add(ActionPrediction(
                opportunity_id=opp.id, attempt_number=attempt, action=p.action,
                probability=p.probability, valid=p.valid, error=p.error,
                model_version=p.model_version,
                feature_pipeline_version=p.feature_pipeline_version))
            audit.record("ACTION_SCORED",
                         f"{p.action} P={p.probability:.4f}" if p.valid
                         else f"{p.action} prediction invalid",
                         {"action": p.action, "probability": p.probability,
                          "valid": p.valid, "error": p.error},
                         model_version=p.model_version)
        self.s.flush()
        transition(self.s, opp, State.CANDIDATES_SCORED, audit, "ACTIONS_RANKED",
                   f"{len(preds)} actions scored")

        # --- finance -------------------------------------------------------
        by_action = {p.action: p for p in preds}
        dn = by_action.get(Action.DO_NOTHING.value)
        if dn is None or not dn.valid:
            transition(self.s, opp, State.STOPPED, audit, "WORKFLOW_STOPPED",
                       "DO_NOTHING baseline unavailable; cannot compute incremental value")
            return {"state": opp.state, "selected_action": None,
                    "candidate_actions": [], "policy": None,
                    "reason": "baseline_unavailable"}

        ev_nothing = valuate_action(econ_base, Action.DO_NOTHING, dn.probability)
        candidates: list[CandidateResult] = []
        for p in preds:
            if not p.valid:
                candidates.append(CandidateResult(
                    p.action, None, False, Decimal("0"), Decimal("0"), Decimal("0"),
                    Decimal("0"), money(econ_base.base_contribution_margin),
                    Decimal("0"), Decimal("0"), p.error))
                continue
            v = valuate_action(econ_base, Action(p.action), p.probability)
            candidates.append(CandidateResult(
                action=p.action, probability=p.probability, valid=True,
                expected_value=money(v.expected_value),
                incremental_expected_value=money(v.expected_value - ev_nothing.expected_value),
                incentive_cost=money(v.incentive_cost), fixed_cost=money(v.fixed_cost),
                base_margin=money(econ_base.base_contribution_margin),
                expected_return_loss=money(v.expected_return_loss),
                expected_cancellation_loss=money(v.expected_cancellation_loss)))

        candidates.sort(key=lambda c: (-c.incremental_expected_value, c.action))
        for rank, c in enumerate(candidates, start=1):
            self.s.add(ActionFinancialEvaluation(
                opportunity_id=opp.id, attempt_number=attempt, action=c.action,
                recovery_probability=Decimal(str(c.probability or 0)),
                base_contribution_margin=c.base_margin,
                incentive_cost_if_recovered=c.incentive_cost,
                fixed_action_cost=c.fixed_cost,
                expected_return_loss=c.expected_return_loss,
                expected_cancellation_loss=c.expected_cancellation_loss,
                expected_value=c.expected_value,
                incremental_expected_value=c.incremental_expected_value,
                rank=rank, calculation_version=FINANCIAL_ENGINE_VERSION))
        self.s.flush()
        transition(self.s, opp, State.ECONOMICALLY_RANKED, audit, "ACTIONS_RANKED",
                   "candidates ranked by incremental expected value",
                   {"ranking": [c.as_dict() for c in candidates[:5]]})

        # --- provisional selection ----------------------------------------
        interventions = [c for c in candidates
                         if c.action != Action.DO_NOTHING.value and c.valid]
        best = interventions[0] if interventions else None
        runner_up = interventions[1] if len(interventions) > 1 else None
        if best is None or best.incremental_expected_value <= 0:
            best = next(c for c in candidates if c.action == Action.DO_NOTHING.value)
            runner_up = None
        margin = (best.incremental_expected_value - runner_up.incremental_expected_value
                  if runner_up else Decimal("999999"))

        # --- policy ---------------------------------------------------------
        spec = spec_for(best.action)
        key = idempotency_key(opp.id, best.action, attempt)
        minutes = (utcnow() - opp.detected_at.replace(tzinfo=timezone.utc)).total_seconds() / 60

        decision = self.policy_engine.evaluate(
            ActionEconomics(
                action=best.action, recovery_probability=best.probability,
                cart_value=money(ctx["cart_value"]),
                base_contribution_margin=best.base_margin,
                incentive_cost_if_recovered=best.incentive_cost,
                fixed_action_cost=best.fixed_cost,
                incremental_expected_value=best.incremental_expected_value,
                decision_margin=margin,
                discount_percent=Decimal(str(spec.discount_percent)),
                prediction_valid=best.valid),
            workflow_state=State(opp.state), attempt_number=attempt,
            minutes_since_detection=minutes,
            existing_execution_keys=self._existing_keys(opp.id),
            proposed_idempotency_key=key,
            customer_declined=bool(ctx.get("customer_declined")))

        # Rejection is not the end: fall back to the next valid candidate that
        # passes, which in the worst case is DO_NOTHING.
        if decision.status is Decision.REJECT and interventions:
            audit.record("ACTION_REJECTED",
                         f"{best.action} rejected by {decision.reason_code}",
                         decision.as_dict(), policy_version=decision.policy_version)
            for alt in interventions[1:] + [
                    next(c for c in candidates if c.action == Action.DO_NOTHING.value)]:
                alt_spec = spec_for(alt.action)
                alt_dec = self.policy_engine.evaluate(
                    ActionEconomics(
                        action=alt.action, recovery_probability=alt.probability,
                        cart_value=money(ctx["cart_value"]),
                        base_contribution_margin=alt.base_margin,
                        incentive_cost_if_recovered=alt.incentive_cost,
                        fixed_action_cost=alt.fixed_cost,
                        incremental_expected_value=alt.incremental_expected_value,
                        decision_margin=Decimal("999999"),
                        discount_percent=Decimal(str(alt_spec.discount_percent)),
                        prediction_valid=alt.valid),
                    workflow_state=State(opp.state), attempt_number=attempt,
                    minutes_since_detection=minutes,
                    existing_execution_keys=self._existing_keys(opp.id),
                    proposed_idempotency_key=idempotency_key(opp.id, alt.action, attempt))
                if alt_dec.status in (Decision.PASS, Decision.REQUIRE_APPROVAL):
                    best, decision = alt, alt_dec
                    break

        pe = PolicyEvaluation(
            opportunity_id=opp.id, attempt_number=attempt, action=best.action,
            policy_version=decision.policy_version, decision=decision.status.value,
            maximum_authorized_downside=decision.maximum_authorized_downside,
            reason_code=decision.reason_code)
        self.s.add(pe)
        self.s.flush()
        for r in decision.triggered_rules:
            self.s.add(PolicyRuleEvaluation(
                policy_evaluation_id=pe.id, rule_id=r.rule_id, passed=r.passed,
                decision=r.decision.value, input_value=r.input_value,
                threshold=r.threshold, reason=r.reason))

        opp.selected_action = best.action
        transition(self.s, opp, State.POLICY_CHECKED, audit, "POLICY_EVALUATED",
                   f"{best.action}: {decision.status.value} ({decision.reason_code})",
                   decision.as_dict(), )

        if decision.status is Decision.PASS:
            transition(self.s, opp, State.AUTHORIZED, audit, "ACTION_AUTHORIZED",
                       f"{best.action} authorized; max downside "
                       f"INR {decision.maximum_authorized_downside}",
                       {"action": best.action,
                        "maximum_authorized_downside": float(decision.maximum_authorized_downside)},
                       )
        elif decision.status is Decision.REQUIRE_APPROVAL:
            transition(self.s, opp, State.AWAITING_APPROVAL, audit, "APPROVAL_REQUIRED",
                       f"{best.action} requires human approval ({decision.reason_code})",
                       decision.as_dict())
        else:
            transition(self.s, opp, State.STOPPED, audit, "WORKFLOW_STOPPED",
                       f"stopped by {decision.reason_code}", decision.as_dict())

        return {
            "opportunity_id": opp.id,
            "state": opp.state,
            "selected_action": best.action,
            "explanation": self.explain(best, candidates, decision),
            "candidate_actions": [c.as_dict() for c in candidates],
            "policy": decision.as_dict(),
            "model_version": self.predictor.model_version,
            "policy_version": decision.policy_version,
        }

    @staticmethod
    def explain(best: CandidateResult, candidates: list[CandidateResult],
                decision: PolicyDecision) -> list[str]:
        """Deterministic structured explanation. No LLM involved."""
        lines = [f"Selected {best.action} because:"]
        if best.action == "DO_NOTHING":
            lines.append("- no intervention had positive incremental expected value")
        else:
            lines.append(f"- highest valid ΔEV = INR {best.incremental_expected_value}")
            if best.probability is not None:
                lines.append(f"- predicted recovery probability = {best.probability:.0%}")
            lines.append(f"- intervention cost if recovered = INR {best.incentive_cost}")
        runner = next((c for c in candidates
                       if c.action not in (best.action, "DO_NOTHING")), None)
        if runner:
            lines.append(f"- next best was {runner.action} at "
                         f"ΔEV INR {runner.incremental_expected_value}")
        failed = [r for r in decision.triggered_rules if not r.passed]
        lines.append(f"- policy {decision.status.value}"
                     + (f" ({', '.join(r.rule_id for r in failed)})" if failed
                        else "; all rules passed"))
        lines.append(f"- maximum authorized downside = INR {decision.maximum_authorized_downside}")
        return lines

    # ---------------------------------------------------------------- approve
    def approve(self, opp: Opportunity, actor_id: str) -> dict:
        audit = AuditRecorder(self.s, opp)
        transition(self.s, opp, State.AUTHORIZED, audit, "ACTION_AUTHORIZED",
                   f"approved by {actor_id}", {"actor": actor_id})
        return {"state": opp.state, "selected_action": opp.selected_action}

    def reject(self, opp: Opportunity, actor_id: str) -> dict:
        audit = AuditRecorder(self.s, opp)
        transition(self.s, opp, State.STOPPED, audit, "WORKFLOW_STOPPED",
                   f"rejected by {actor_id}", {"actor": actor_id})
        return {"state": opp.state}

    # ---------------------------------------------------------------- execute
    def execute(self, opp: Opportunity, executor) -> dict:
        """Idempotent. Concurrent calls collapse to one execution row."""
        audit = AuditRecorder(self.s, opp)
        action = opp.selected_action
        if not action:
            raise InvalidStateTransition("no selected action to execute")

        # Idempotency is checked BEFORE the state guard. A repeat call after a
        # successful execution has already advanced the state must return the
        # original execution, not raise — a client retry is not an error.
        key = idempotency_key(opp.id, action, opp.current_attempt)
        existing = self.s.execute(
            select(RecoveryExecution).where(RecoveryExecution.idempotency_key == key)
        ).scalar_one_or_none()
        if existing is None and State(opp.state) is not State.AUTHORIZED:
            raise InvalidStateTransition(
                f"execute requires AUTHORIZED, got {opp.state}",
                opportunity_id=opp.id)
        if existing is not None:
            audit.record("EXECUTION_CREATED",
                         "duplicate execute request; returning existing execution",
                         {"execution_id": existing.execution_id, "duplicate": True},
                         execution_id=existing.execution_id)
            return {"execution_id": existing.execution_id, "duplicate": True,
                    "status": existing.status, "state": opp.state,
                    "external_order_id": existing.external_order_id}

        transition(self.s, opp, State.EXECUTION_PENDING, audit, "EXECUTION_CREATED",
                   f"execution pending for {action}")

        execution = RecoveryExecution(
            execution_id=f"exe_{uuid.uuid4().hex[:16]}", opportunity_id=opp.id,
            attempt_number=opp.current_attempt, action=action,
            idempotency_key=key, execution_provider=opp.execution_mode,
            status="PENDING", currency=settings.currency)
        self.s.add(execution)
        try:
            # Commit the claim on the idempotency key BEFORE any external call.
            self.s.commit()
        except IntegrityError:
            self.s.rollback()
            existing = self.s.execute(
                select(RecoveryExecution).where(RecoveryExecution.idempotency_key == key)
            ).scalar_one_or_none()
            if existing is None:
                raise DuplicateExecution("idempotency conflict without a visible row")
            return {"execution_id": existing.execution_id, "duplicate": True,
                    "status": existing.status, "state": opp.state,
                    "external_order_id": existing.external_order_id}

        opp = self.s.get(Opportunity, opp.id)
        audit = AuditRecorder(self.s, opp)
        transition(self.s, opp, State.EXECUTING, audit, "EXECUTION_CREATED",
                   f"executing {action} via {opp.execution_mode}",
                   {"execution_id": execution.execution_id},
                   )

        try:
            result = executor.execute(opp, execution, self.s)
        except Exception as exc:  # noqa: BLE001 — never invent success
            execution.status = "FAILED"
            execution.error_code = getattr(exc, "code", "EXECUTION_ERROR")
            execution.error_message = str(exc)[:500]
            transition(self.s, opp, State.EXECUTION_FAILED, audit, "EXECUTION_ERROR",
                       f"execution failed: {execution.error_code}",
                       {"error_code": execution.error_code})
            self.s.commit()
            return {"execution_id": execution.execution_id, "status": "FAILED",
                    "state": opp.state, "error_code": execution.error_code}

        execution.status = result.get("status", "SUBMITTED")
        execution.external_order_id = result.get("order_id")
        execution.external_payment_id = result.get("payment_id")
        execution.external_payment_link_id = result.get("payment_link_id")
        execution.amount = money(result["amount"]) if result.get("amount") is not None else None

        if result.get("order_id"):
            audit.record("RAZORPAY_ORDER_CREATED",
                         f"order {result['order_id']} for INR {execution.amount}",
                         {"order_id": result["order_id"], "amount": float(execution.amount or 0)},
                         execution_id=execution.execution_id)

        if result.get("terminal_state"):
            transition(self.s, opp, State(result["terminal_state"]), audit,
                       "RECOVERY_CONFIRMED" if result["terminal_state"] == "RECOVERED"
                       else "WORKFLOW_STOPPED",
                       result.get("summary", "execution completed"),
                       execution_id=execution.execution_id)
            if result["terminal_state"] == "RECOVERED":
                self.confirm_recovery(opp, execution, gross=result.get("amount", 0))
        else:
            transition(self.s, opp, State.AWAITING_PAYMENT, audit, "CHECKOUT_STARTED",
                       "awaiting payment confirmation via webhook",
                       {"order_id": result.get("order_id")},
                       )
        self.s.commit()
        return {"execution_id": execution.execution_id, "duplicate": False,
                "status": execution.status, "state": opp.state,
                "external_order_id": execution.external_order_id,
                "checkout": result.get("checkout")}

    # ------------------------------------------------------------ finalization
    def confirm_recovery(self, opp: Opportunity, execution: RecoveryExecution,
                         gross: float | Decimal, payment_id: str | None = None) -> bool:
        """Idempotent: one RecoveryOutcome row per opportunity, enforced by PK."""
        existing = self.s.get(RecoveryOutcome, opp.id)
        if existing is not None:
            return False

        ctx = dict(opp.context)
        spec = spec_for(execution.action)
        cart = Decimal(str(ctx.get("cart_value", gross)))
        discount = money(cart * Decimal(str(spec.discount_percent)) / 100)
        subsidy = (money(ctx.get("shipping_fee_charged", 0))
                   if spec.waives_shipping_fee else Decimal("0"))
        net_gmv = money(cart - discount)
        contribution = money(
            Decimal(str(ctx.get("base_contribution_margin", 0)))
            - discount - subsidy - Decimal(str(spec.fixed_cost)))

        self.s.add(RecoveryOutcome(
            opportunity_id=opp.id, execution_id=execution.execution_id,
            gross_order_value=money(cart), discount_amount=discount,
            shipping_subsidy=subsidy, net_recovered_gmv=net_gmv,
            realized_contribution=contribution,
            payment_id=payment_id or execution.external_payment_id,
            order_id=execution.external_order_id))
        return True


# ------------------------------------------------------------------ executors
class SimulatorRecoveryExecutor:
    """Deterministic local executor. No external calls, no Razorpay traffic."""

    name = "SIMULATOR"

    def __init__(self, force_outcome: str | None = None):
        self.force_outcome = force_outcome

    def execute(self, opp: Opportunity, execution: RecoveryExecution, session) -> dict:
        ctx = dict(opp.context)
        amount = float(ctx.get("cart_value", 0))
        spec = spec_for(execution.action)
        amount -= amount * spec.discount_percent / 100.0

        if execution.action == Action.DO_NOTHING.value:
            return {"status": "COMPLETED", "amount": 0,
                    "terminal_state": "NOT_RECOVERED",
                    "summary": "no intervention executed"}
        if self.force_outcome == "RECOVERED":
            return {"status": "COMPLETED", "amount": amount,
                    "terminal_state": "RECOVERED", "summary": "simulated recovery"}
        if self.force_outcome == "NOT_RECOVERED":
            return {"status": "COMPLETED", "amount": amount,
                    "terminal_state": "NOT_RECOVERED", "summary": "simulated non-recovery"}
        return {"status": "SUBMITTED", "amount": amount,
                "order_id": f"sim_order_{execution.execution_id[-8:]}"}