"""Agent tools (spec §23-24, §32-34).

This module is the security boundary. The LLM can only ever name a tool and
supply an opportunity id; everything financial is recomputed server-side from
the database. Three properties matter:

1. **The action space is closed.** A tool cannot execute an action the agent
   invented; `request_execution` ignores any action the agent names and uses the
   one the policy engine authorized.
2. **Agent-supplied authorization is ignored.** An `approved: true` field from
   the model has no effect — policy is re-evaluated server-side.
3. **No business logic lives here.** Tools are thin wrappers over the existing
   services, so the agent path and the API path cannot drift apart.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field
from sqlalchemy import select

from backend.app.db.models import (
    ActionFinancialEvaluation, AuditEvent, Opportunity, PaymentFailureRecord,
    PolicyEvaluation, RecoveryExecution, RecoveryOutcome,
)
from backend.app.domain import State
from backend.app.agents.authorizer import PLANNER_TOOLS
from backend.app.services.adaptive import ADAPTIVE_RULES_VERSION
from backend.app.services.razorpay import RazorpayRecoveryExecutor
from backend.app.services.workflow import (
    RecoveryWorkflow, SimulatorRecoveryExecutor, build_retry_context,
)


class ToolError(Exception):
    def __init__(self, code: str, message: str = ""):
        super().__init__(message or code)
        self.code = code
        self.message = message or code


# ------------------------------------------------------------------ schemas
class OpportunityRef(BaseModel):
    opportunity_id: str = Field(min_length=1, max_length=64)


class ExecutionRequest(BaseModel):
    """Note what is absent: no action, no amount, no approval flag.

    The agent asks to execute *what policy already authorized*. It cannot
    smuggle a different action or a discount percentage through this call.
    """

    opportunity_id: str = Field(min_length=1, max_length=64)
    authorized_policy_evaluation_id: int
    attempt_number: int


class ActionScore(BaseModel):
    action: str
    probability: float | None
    base_probability: float | None = None
    adaptive_delta: float = 0.0
    incremental_expected_value: float
    incentive_cost: float
    eligible: bool = True


# ---------------------------------------------------------------- tool impl
class AgentTools:
    """Bounded tool surface over existing backend services.

    Planner-facing names are deliberately few and stable. Each is a thin wrapper
    over a service that already exists, so the agent path and the HTTP API
    cannot drift apart. None of these accept a monetary argument.
    """

    READ_ONLY = {
        "get_opportunity", "get_customer_context", "get_payment_history",
        "get_recovery_history", "diagnose_recovery_context",
        "get_execution_status", "get_latest_provider_events", "get_audit_timeline",
    }
    MUTATING = {"analyze_opportunity", "request_execution", "stop_recovery",
                "escalate_to_human"}
    ALL = READ_ONLY | MUTATING | {"wait_for_payment_state"} | set(PLANNER_TOOLS)

    def __init__(self, session, opportunity_id: str):
        self.s = session
        self.opportunity_id = opportunity_id
        self.wf = RecoveryWorkflow(session)

    # Which tools can legally act in each workflow state. Advertising a tool
    # that the state machine will reject sets the planner up to fail: it burns a
    # step, produces a refusal, and teaches the model nothing. Offering only
    # valid options is both safer and far easier for a small model to get right.
    STATE_TOOLS: dict[str, set[str]] = {
        State.DETECTED.value: {"diagnose_recovery_context", "analyze_opportunity"},
        State.PAYMENT_FAILED_RECOVERABLE.value: {
            "get_payment_history", "diagnose_recovery_context", "analyze_opportunity"},
        State.NOT_RECOVERED.value: {"analyze_opportunity", "stop_recovery"},
        State.EXECUTION_FAILED.value: {"analyze_opportunity", "stop_recovery",
                                       "escalate_to_human"},
        State.AUTHORIZED.value: {"request_execution", "stop_recovery"},
        State.AWAITING_APPROVAL.value: {"wait_for_payment_state", "escalate_to_human"},
        # Deliberately no wait tool here: at AWAITING_PAYMENT the correct
        # answer is the WAIT verb, which ends the run. Offering a "check the
        # state" tool invites a poll loop that burns the step budget and
        # changes nothing.
        State.AWAITING_PAYMENT.value: {"get_latest_provider_events"},
        State.EXECUTING.value: {"get_execution_status"},
        State.EXECUTION_PENDING.value: {"get_execution_status"},
    }

    # These return the same answer every time for an unchanged opportunity, so
    # re-offering one after it has been called invites a loop with no new
    # information. Excluded once already called.
    IDEMPOTENT_READS = {
        "get_opportunity", "get_customer_context", "get_payment_history",
        "get_recovery_history", "diagnose_recovery_context", "get_audit_timeline",
        "wait_for_payment_state",
    }

    @classmethod
    def available_for_state(cls, state: str,
                            already_called: set[str] | None = None) -> list[str]:
        """Read-only tools are always safe; mutating ones are state-gated.

        Tools whose answer cannot have changed are dropped once called, which
        keeps a small planner from cycling on the same read.
        """
        allowed = cls.STATE_TOOLS.get(state, set())
        safe_reads = cls.READ_ONLY - {"analyze_opportunity"}
        tools = safe_reads | allowed
        if already_called:
            tools -= (set(already_called) & cls.IDEMPOTENT_READS)
        return sorted(tools)

    @classmethod
    def guidance_for_state(cls, state: str) -> str:
        return {
            State.DETECTED.value:
                "Diagnose first, then analyze_opportunity to score actions and "
                "run the policy check.",
            State.AUTHORIZED.value:
                "Policy has already authorized an action. The only productive "
                "step is request_execution. Do NOT analyze again.",
            State.AWAITING_PAYMENT.value:
                "An external payment is pending. Choose WAIT.",
            State.AWAITING_APPROVAL.value:
                "A human must approve. Choose WAIT.",
            State.PAYMENT_FAILED_RECOVERABLE.value:
                "A payment failed. Gather payment evidence, re-diagnose, then "
                "analyze_opportunity for a fresh attempt.",
            State.RECOVERED.value: "Terminal. Choose STOP.",
            State.STOPPED.value: "Terminal. Choose STOP.",
            State.NOT_RECOVERED.value: "Terminal unless a new attempt is warranted.",
        }.get(state, "Gather evidence, or choose STOP if no safe step remains.")

    def _opp(self) -> Opportunity:
        o = self.s.get(Opportunity, self.opportunity_id)
        if o is None:
            raise ToolError("OPPORTUNITY_NOT_FOUND", self.opportunity_id)
        return o

    def call(self, tool_name: str, payload: dict[str, Any] | None = None) -> dict:
        if tool_name not in self.ALL and tool_name not in PLANNER_TOOLS:
            # An unknown tool is a hallucination, not an instruction.
            raise ToolError("UNKNOWN_TOOL", tool_name)
        fn = getattr(self, tool_name, None)
        if fn is None:
            raise ToolError("UNKNOWN_TOOL", tool_name)
        return fn(payload or {})

    # ---------------------------------------------- planner-facing allowlist
    def get_opportunity_state(self, _: dict) -> dict:
        """Compact, read-only workflow snapshot."""
        o = self._opp()
        pol = self.s.execute(
            select(PolicyEvaluation).where(PolicyEvaluation.opportunity_id == o.id)
            .order_by(PolicyEvaluation.id.desc())).scalars().first()
        return {"opportunity_id": o.id, "state": o.state,
                "attempt_number": o.current_attempt,
                "selected_action": o.selected_action,
                "policy_decision": pol.decision if pol else None,
                "opportunity_type": o.opportunity_type}

    def get_policy_summary(self, _: dict) -> dict:
        """Read-only view of the most recent policy evaluation."""
        o = self._opp()
        pol = self.s.execute(
            select(PolicyEvaluation).where(PolicyEvaluation.opportunity_id == o.id)
            .order_by(PolicyEvaluation.id.desc())).scalars().first()
        if pol is None:
            return {"policy_decision": None, "rules_evaluated": 0}
        return {"policy_decision": pol.decision, "reason_code": pol.reason_code,
                "policy_version": pol.policy_version,
                "action": pol.action,
                "maximum_authorized_downside": float(pol.maximum_authorized_downside),
                "rules_evaluated": len(pol.rules),
                "rules_failed": [r.rule_id for r in pol.rules if not r.passed]}

    def get_audit_summary(self, _: dict) -> dict:
        rows = self.s.execute(
            select(AuditEvent).where(AuditEvent.opportunity_id == self.opportunity_id)
            .order_by(AuditEvent.sequence_number.desc()).limit(8)).scalars().all()
        return {"recent_events": [{"type": e.event_type, "summary": e.summary}
                                  for e in reversed(rows)]}

    def request_human_approval(self, payload: dict) -> dict:
        """Record that a human decision is required. Does NOT grant approval.

        The agent can flag and wait; only an authenticated operator acting
        through the API can actually approve.
        """
        from backend.app.services.workflow import AuditRecorder

        o = self._opp()
        AuditRecorder(self.s, o).record(
            "APPROVAL_REQUIRED",
            "agent flagged this opportunity for human approval",
            {"reason": str(payload.get("reason", ""))[:200],
             "requested_by": "recovery-orchestrator"})
        self.s.commit()
        return {"state": o.state, "approval_requested": True,
                "approved": False,
                "note": "approval must be granted by an operator via the API"}

    def stop_workflow(self, payload: dict) -> dict:
        return self.stop_recovery(payload)

    # ------------------------------------------------------------ read-only
    def get_opportunity(self, _: dict) -> dict:
        o = self._opp()
        ctx = dict(o.context)
        return {"opportunity_id": o.id, "type": o.opportunity_type, "state": o.state,
                "attempt_number": o.current_attempt,
                "revenue_at_risk": float(o.revenue_at_risk),
                "contribution_margin_at_risk": float(o.contribution_margin_at_risk),
                "selected_action": o.selected_action,
                "execution_mode": o.execution_mode,
                "cart_value": ctx.get("cart_value"),
                "failure_reason": ctx.get("failure_reason")}

    def get_customer_context(self, _: dict) -> dict:
        ctx = dict(self._opp().context)
        return {k: ctx.get(k) for k in
                ("customer_segment", "orders_lifetime", "average_order_value",
                 "previous_checkout_abandonments", "previous_payment_failures",
                 "coupon_history_missing")}

    def get_payment_history(self, _: dict) -> dict:
        rows = self.s.execute(
            select(PaymentFailureRecord)
            .where(PaymentFailureRecord.opportunity_id == self.opportunity_id)
            .order_by(PaymentFailureRecord.id)).scalars().all()
        return {"failures": [{"failure_code": r.failure_code,
                              "failure_step": r.failure_step,
                              "payment_method": r.payment_method,
                              "payment_id": r.payment_id} for r in rows],
                "count": len(rows)}

    def get_recovery_history(self, _: dict) -> dict:
        execs = self.s.execute(
            select(RecoveryExecution)
            .where(RecoveryExecution.opportunity_id == self.opportunity_id)
            .order_by(RecoveryExecution.attempt_number)).scalars().all()
        return {"executions": [{"attempt": e.attempt_number, "action": e.action,
                                "status": e.status, "order_id": e.external_order_id}
                               for e in execs]}

    def diagnose_recovery_context(self, _: dict) -> dict:
        """Structured diagnosis (spec §44). Advisory only — never authoritative."""
        o = self._opp()
        retry = build_retry_context(self.s, o)
        ctx = dict(o.context)

        evidence: list[str] = []
        fee_ratio = float(ctx.get("shipping_fee_to_cart_ratio") or 0)
        if retry.previous_failure_category is not None:
            primary = "PAYMENT_FRICTION"
            secondary = "SHIPPING_FRICTION" if fee_ratio > 0.01 else None
            confidence = 0.81
            evidence.append(
                f"provider reported {retry.previous_failure_category.value}"
                + (f" at {retry.previous_failure_step}" if retry.previous_failure_step else ""))
            if retry.active_incentives:
                evidence.append(
                    f"incentive already active from a prior attempt: "
                    f"{', '.join(retry.active_incentives)}")
            evidence.append(f"attempt {retry.attempt_number}")
        elif fee_ratio > 0.015:
            primary, secondary, confidence = "SHIPPING_FRICTION", None, 0.72
            evidence.append(f"shipping fee is {fee_ratio:.1%} of cart value")
            evidence.append(f"checkout stage {ctx.get('abandonment_stage')}")
        else:
            primary, secondary, confidence = "PRICE_OR_INTENT", None, 0.45
            evidence.append("no dominant friction signal; confidence is low")

        return {"primary_reason": primary, "secondary_reason": secondary,
                "confidence": confidence, "evidence": evidence,
                "retry_context": retry.as_dict(),
                "adaptive_rules_version": ADAPTIVE_RULES_VERSION,
                "note": "Diagnosis is advisory context. Financial and policy "
                        "layers remain authoritative."}

    def get_execution_status(self, _: dict) -> dict:
        rows = self.s.execute(
            select(RecoveryExecution)
            .where(RecoveryExecution.opportunity_id == self.opportunity_id)).scalars().all()
        outcome = self.s.get(RecoveryOutcome, self.opportunity_id)
        return {"executions": [{"execution_id": e.execution_id, "status": e.status,
                                "attempt": e.attempt_number} for e in rows],
                "recovered": outcome is not None,
                "net_recovered_gmv": float(outcome.net_recovered_gmv) if outcome else None}

    def get_latest_provider_events(self, _: dict) -> dict:
        rows = self.s.execute(
            select(AuditEvent)
            .where(AuditEvent.opportunity_id == self.opportunity_id,
                   AuditEvent.event_type.in_(
                       ["WEBHOOK_RECEIVED", "PAYMENT_FAILED", "RECOVERY_CONFIRMED"]))
            .order_by(AuditEvent.sequence_number.desc()).limit(5)).scalars().all()
        return {"events": [{"type": e.event_type, "summary": e.summary} for e in rows]}

    def get_audit_timeline(self, _: dict) -> dict:
        rows = self.s.execute(
            select(AuditEvent).where(AuditEvent.opportunity_id == self.opportunity_id)
            .order_by(AuditEvent.sequence_number)).scalars().all()
        return {"events": [{"sequence": e.sequence_number, "type": e.event_type,
                            "summary": e.summary} for e in rows]}

    def wait_for_payment_state(self, _: dict) -> dict:
        o = self._opp()
        return {"state": o.state, "waiting": o.state in
                (State.AWAITING_PAYMENT.value, State.AWAITING_APPROVAL.value)}

    # ------------------------------------------------------------- mutating
    def analyze_opportunity(self, _: dict) -> dict:
        """Run the full deterministic pipeline: score, rank, policy-check."""
        o = self._opp()
        result = self.wf.analyze(o)
        self.s.commit()
        pol = self.s.execute(
            select(PolicyEvaluation)
            .where(PolicyEvaluation.opportunity_id == o.id)
            .order_by(PolicyEvaluation.id.desc())).scalars().first()
        return {
            "state": result["state"],
            "selected_action": result["selected_action"],
            "policy_decision": result["policy"]["decision"],
            "policy_evaluation_id": pol.id if pol else None,
            "maximum_authorized_downside": result["policy"]["maximum_authorized_downside"],
            "attempt_number": o.current_attempt,
            "scores": [ActionScore(
                action=c["action"], probability=c["probability"],
                base_probability=c.get("base_probability"),
                adaptive_delta=c.get("adaptive_delta", 0.0),
                incremental_expected_value=c["incremental_expected_value"],
                incentive_cost=c["incentive_cost_if_recovered"]).model_dump()
                for c in result["candidate_actions"]],
            "explanation": result["explanation"],
        }

    def request_execution(self, payload: dict) -> dict:
        """Execute the action policy authorized — not one the agent chose.

        The agent's own `selected_action` or `approved` fields, if present, are
        deliberately discarded. Policy state is re-read from the database.
        """
        req = ExecutionRequest(
            opportunity_id=self.opportunity_id,
            authorized_policy_evaluation_id=int(
                payload.get("authorized_policy_evaluation_id") or 0),
            attempt_number=int(payload.get("attempt_number") or 0))

        o = self._opp()
        pol = self.s.get(PolicyEvaluation, req.authorized_policy_evaluation_id)
        if pol is None or pol.opportunity_id != o.id:
            raise ToolError("POLICY_EVALUATION_NOT_FOUND",
                            "no matching policy evaluation for this opportunity")
        if pol.decision != "PASS":
            raise ToolError("POLICY_NOT_AUTHORIZED",
                            f"policy decision is {pol.decision}, not PASS")
        if o.state != State.AUTHORIZED.value:
            raise ToolError("NOT_AUTHORIZED_STATE",
                            f"opportunity state is {o.state}, not AUTHORIZED")
        if pol.action != o.selected_action:
            raise ToolError("ACTION_MISMATCH", "policy evaluation does not match "
                                               "the currently selected action")

        executor = (RazorpayRecoveryExecutor() if o.execution_mode == "RAZORPAY_TEST"
                    else SimulatorRecoveryExecutor())
        result = self.wf.execute(o, executor)
        return {"execution_id": result["execution_id"], "status": result["status"],
                "duplicate": result.get("duplicate", False),
                "state": result["state"],
                "order_id": result.get("external_order_id")}

    def stop_recovery(self, payload: dict) -> dict:
        o = self._opp()
        if State(o.state) in (State.RECOVERED, State.STOPPED):
            return {"state": o.state, "already_terminal": True}
        r = self.wf.reject(o, actor_id="recovery-orchestrator")
        self.s.commit()
        return r

    def escalate_to_human(self, payload: dict) -> dict:
        from backend.app.services.workflow import AuditRecorder
        o = self._opp()
        AuditRecorder(self.s, o).record(
            "AGENT_ESCALATED", "agent escalated to a human operator",
            {"reason": str(payload.get("reason", ""))[:200]})
        self.s.commit()
        return {"state": o.state, "escalated": True}