"""Deterministic policy engine.

The only component permitted to authorize a financial action. It takes no
model, no LLM and no free text as input to its decisions — only numbers already
computed by the financial engine, the merchant's versioned policy object, and
workflow state.

Every rule is named, evaluated in a fixed order, and individually recorded, so
"why was this allowed?" is answerable from stored rows rather than by rerunning
code.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum

from backend.app.core.config import MerchantPolicy
from backend.app.domain import State


class Decision(str, Enum):
    PASS = "PASS"
    REJECT = "REJECT"
    REQUIRE_APPROVAL = "REQUIRE_APPROVAL"
    STOP = "STOP"


# Severity ordering: the worst outcome across all rules wins.
_SEVERITY = {Decision.PASS: 0, Decision.REQUIRE_APPROVAL: 1,
             Decision.REJECT: 2, Decision.STOP: 3}


@dataclass
class RuleResult:
    rule_id: str
    passed: bool
    decision: Decision
    reason: str
    input_value: str | None = None
    threshold: str | None = None

    def as_dict(self) -> dict:
        return {"rule_id": self.rule_id, "passed": self.passed,
                "decision": self.decision.value, "reason": self.reason,
                "input": self.input_value, "threshold": self.threshold}


@dataclass
class PolicyDecision:
    status: Decision
    triggered_rules: list[RuleResult] = field(default_factory=list)
    policy_version: str = ""
    maximum_authorized_downside: Decimal = Decimal("0")
    reason_code: str = "OK"
    evaluated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def authorized(self) -> bool:
        return self.status is Decision.PASS

    def as_dict(self) -> dict:
        return {
            "decision": self.status.value,
            "policy_version": self.policy_version,
            "maximum_authorized_downside": float(self.maximum_authorized_downside),
            "reason_code": self.reason_code,
            "rules": [r.as_dict() for r in self.triggered_rules],
            "evaluated_at": self.evaluated_at.isoformat(),
        }


@dataclass
class ActionEconomics:
    """Everything the policy engine needs, all pre-computed. No model access."""

    action: str
    recovery_probability: float | None
    cart_value: Decimal
    base_contribution_margin: Decimal
    incentive_cost_if_recovered: Decimal
    fixed_action_cost: Decimal
    incremental_expected_value: Decimal
    decision_margin: Decimal = Decimal("0")
    discount_percent: Decimal = Decimal("0")
    prediction_valid: bool = True

    @property
    def is_discount(self) -> bool:
        return self.discount_percent > 0

    @property
    def maximum_downside(self) -> Decimal:
        """Worst case: incentive granted, order later returned, plus fixed cost."""
        return self.incentive_cost_if_recovered + self.fixed_action_cost

    @property
    def remaining_margin_percent(self) -> Decimal:
        if self.cart_value <= 0:
            return Decimal("0")
        remaining = self.base_contribution_margin - self.incentive_cost_if_recovered
        return (remaining / self.cart_value) * Decimal("100")


class PolicyEngine:
    """Evaluates all rules; the most severe outcome decides."""

    def __init__(self, policy: MerchantPolicy | None = None):
        self.policy = policy or MerchantPolicy.load()

    def evaluate(
        self,
        econ: ActionEconomics,
        workflow_state: State,
        attempt_number: int,
        minutes_since_detection: float = 0.0,
        existing_execution_keys: set[str] | None = None,
        proposed_idempotency_key: str | None = None,
        customer_declined: bool = False,
    ) -> PolicyDecision:
        p = self.policy
        rules: list[RuleResult] = []

        def add(rule_id, passed, decision, reason, inp=None, thr=None, ok_reason=None):
            # `reason` describes why the rule WOULD fail. When it passes, that text
            # is misleading, so a passing rule reports the satisfied condition
            # instead. An audit trail full of alarming text next to green checks
            # is worse than no text at all.
            text = (ok_reason or f"ok: {reason}") if passed else reason
            rules.append(RuleResult(rule_id, passed, decision if not passed else Decision.PASS,
                                    text, str(inp) if inp is not None else None,
                                    str(thr) if thr is not None else None))

        # 1. workflow validity
        ok = workflow_state in (State.ECONOMICALLY_RANKED, State.POLICY_CHECKED,
                                State.AWAITING_APPROVAL, State.ANALYZING,
                                State.CANDIDATES_SCORED)
        add("RULE_WORKFLOW_STATE_VALID", ok, Decision.STOP,
            f"state {workflow_state.value} is not a valid point to authorize",
            workflow_state.value,
            ok_reason=f"state {workflow_state.value} is valid for authorization")

        # 2. model-output validity — an invalid probability must never buy anything
        valid_prob = (econ.prediction_valid and econ.recovery_probability is not None
                      and math.isfinite(econ.recovery_probability)
                      and 0.0 <= econ.recovery_probability <= 1.0)
        add("RULE_INVALID_MODEL_OUTPUT", valid_prob, Decision.STOP,
            "model output invalid; autonomous action disabled",
            econ.recovery_probability,
            ok_reason="model output is a finite probability in [0, 1]")

        # 3. duplicate detection
        dup = bool(existing_execution_keys and proposed_idempotency_key
                   and proposed_idempotency_key in existing_execution_keys)
        add("RULE_DUPLICATE_ACTION_PREVENTION", not dup, Decision.STOP,
            "an execution already exists for this opportunity/action/attempt",
            proposed_idempotency_key,
            ok_reason="no prior execution for this opportunity/action/attempt")

        # 4. attempt limit
        add("RULE_MAX_RECOVERY_ATTEMPTS", attempt_number <= p.max_recovery_attempts,
            Decision.STOP, "recovery attempt limit reached",
            attempt_number, p.max_recovery_attempts,
            ok_reason=f"attempt {attempt_number} of {p.max_recovery_attempts} allowed")

        # 5. expiry
        add("RULE_OPPORTUNITY_EXPIRED",
            minutes_since_detection <= p.opportunity_ttl_minutes, Decision.STOP,
            "opportunity older than merchant TTL",
            round(minutes_since_detection, 1), p.opportunity_ttl_minutes,
            ok_reason="opportunity is within the merchant TTL window")

        # 5b. explicit customer decline
        add("RULE_CUSTOMER_DECLINED", not customer_declined, Decision.STOP,
            "customer explicitly declined further contact",
            ok_reason="customer has not declined further contact")

        is_do_nothing = econ.action == "DO_NOTHING"

        # 6-9 only constrain actions that actually spend money.
        if not is_do_nothing:
            add("RULE_MINIMUM_DELTA_EV",
                econ.incremental_expected_value > p.minimum_incremental_expected_value,
                Decision.REJECT, "incremental expected value not above threshold",
                econ.incremental_expected_value, p.minimum_incremental_expected_value,
                ok_reason="incremental expected value clears the threshold")

            add("RULE_MINIMUM_MODEL_CONFIDENCE",
                (econ.recovery_probability or 0) >= p.minimum_recovery_probability,
                Decision.REJECT, "recovery probability below minimum",
                econ.recovery_probability, p.minimum_recovery_probability,
                ok_reason="recovery probability meets the minimum confidence")

            add("RULE_MINIMUM_DECISION_MARGIN",
                econ.decision_margin >= p.minimum_decision_margin,
                Decision.REQUIRE_APPROVAL,
                "gap to the runner-up action is too small to decide autonomously",
                econ.decision_margin, p.minimum_decision_margin,
                ok_reason="clear enough gap to the runner-up to decide autonomously")

            if econ.is_discount:
                add("RULE_DISCOUNT_PERCENT_LIMIT",
                    econ.discount_percent <= p.max_autonomous_discount_percent,
                    Decision.REJECT, "discount percent exceeds autonomous limit",
                    econ.discount_percent, p.max_autonomous_discount_percent,
                    ok_reason="discount percent within autonomous limit")
                add("RULE_DISCOUNT_AMOUNT_LIMIT",
                    econ.incentive_cost_if_recovered <= p.max_autonomous_discount_amount,
                    Decision.REJECT, "discount amount exceeds autonomous limit",
                    econ.incentive_cost_if_recovered, p.max_autonomous_discount_amount,
                    ok_reason="discount amount within autonomous limit")
                add("RULE_HUMAN_APPROVAL_DISCOUNT_AMOUNT",
                    econ.incentive_cost_if_recovered
                    <= p.human_approval_required_above_discount_amount,
                    Decision.REQUIRE_APPROVAL, "discount amount requires human approval",
                    econ.incentive_cost_if_recovered,
                    p.human_approval_required_above_discount_amount,
                    ok_reason="discount below the human-approval threshold")

            if econ.action == "FREE_SHIPPING":
                add("RULE_FREE_SHIPPING_LIMIT",
                    econ.incentive_cost_if_recovered <= p.max_free_shipping_cost,
                    Decision.REJECT, "shipping subsidy exceeds limit",
                    econ.incentive_cost_if_recovered, p.max_free_shipping_cost,
                    ok_reason="shipping subsidy within limit")

            add("RULE_MINIMUM_MARGIN",
                econ.remaining_margin_percent
                >= p.minimum_remaining_contribution_margin_percent,
                Decision.REJECT, "remaining contribution margin below floor",
                round(econ.remaining_margin_percent, 2),
                p.minimum_remaining_contribution_margin_percent,
                ok_reason="remaining contribution margin protected")

            if p.human_approval_required_for_high_value_orders:
                add("RULE_HIGH_VALUE_REQUIRES_APPROVAL",
                    econ.cart_value < p.high_value_order_threshold,
                    Decision.REQUIRE_APPROVAL,
                    "high-value order requires human approval",
                    econ.cart_value, p.high_value_order_threshold,
                    ok_reason="order below the high-value approval threshold")

        failed = [r for r in rules if not r.passed]
        status = Decision.PASS
        reason_code = "OK"
        if failed:
            worst = max(failed, key=lambda r: _SEVERITY[r.decision])
            status = worst.decision
            reason_code = worst.rule_id

        downside = Decimal("0") if is_do_nothing else econ.maximum_downside
        return PolicyDecision(
            status=status, triggered_rules=rules, policy_version=p.policy_version,
            maximum_authorized_downside=downside, reason_code=reason_code,
        )