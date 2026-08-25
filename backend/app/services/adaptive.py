"""Adaptive Recovery Adjustment Layer (spec Part A §7-12).

Sits **after** the frozen model and **before** economic ranking. It exists
because the model was trained on a single-shot decision and has no vocabulary
for "this exact action already failed once, for this reason".

Explicitly not machine learning
-------------------------------
Every adjustment here is a hand-written, versioned, bounded rule with a stated
justification. It is reported separately from the model probability so nobody
can mistake it for a learned effect::

    base_model_probability  ->  adjustment (with reason)  ->  adapted_probability

The frozen model is never retrained or mutated to accommodate retries.

Three responsibilities
----------------------
1. **Eligibility** — remove actions that make no sense given what just failed.
2. **Adjustment** — shift relative priority toward the family of actions that
   addresses the *new* blocker, within a hard bound.
3. **Incentive state** — track which incentives are already active so attempt 2
   does not book the same shipping subsidy twice.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal

from backend.app.services.failure_taxonomy import FailureCategory
from ml.actions import Action, spec_for

ADAPTIVE_RULES_VERSION = "adaptive-recovery-rules-1.0.0"

# Hard bound on any single adjustment. Keeps the layer a nudge to relative
# ordering, never a replacement for the model's estimate.
MAX_ABS_ADJUSTMENT = 0.12

# Repeating an action that already failed is not worthless (transient failures
# exist) but it is materially less likely to work than it was the first time.
REPEAT_ACTION_PENALTY = 0.06

# Which action families address which blocker. Derived from the same domain
# logic encoded in the simulator's response surface, stated here explicitly.
CATEGORY_PREFERENCES: dict[FailureCategory, dict[Action, float]] = {
    FailureCategory.PAYMENT_INFRASTRUCTURE_FAILURE: {
        Action.DELAYED_RETRY: +0.10,       # transient; waiting is the fix
        Action.PAYMENT_METHOD_SWITCH: +0.05,
        Action.IMMEDIATE_RETRY: -0.08,     # same broken rail, same instant
        Action.SMALL_DISCOUNT: -0.05,      # a price cut cannot fix a bank
        Action.MEDIUM_DISCOUNT: -0.06,
        Action.FREE_SHIPPING: -0.04,
    },
    FailureCategory.PAYMENT_METHOD_FAILURE: {
        Action.PAYMENT_METHOD_SWITCH: +0.12,
        Action.PAYMENT_LINK: +0.06,        # lets the customer pick another rail
        Action.IMMEDIATE_RETRY: -0.09,     # the instrument itself was refused
        Action.DELAYED_RETRY: -0.05,
        Action.SMALL_DISCOUNT: -0.04,
        Action.MEDIUM_DISCOUNT: -0.05,
    },
    FailureCategory.AUTHENTICATION_FAILURE: {
        Action.PAYMENT_LINK: +0.09,        # fresh auth session
        Action.PAYMENT_METHOD_SWITCH: +0.07,
        Action.IMMEDIATE_RETRY: -0.04,
        Action.MEDIUM_DISCOUNT: -0.04,
    },
    FailureCategory.CUSTOMER_FUNDS_FAILURE: {
        Action.DELAYED_RETRY: +0.09,       # balance may change later
        Action.SMALL_DISCOUNT: +0.04,      # a genuine affordability lever
        Action.IMMEDIATE_RETRY: -0.10,     # funds will not have appeared
        Action.FREE_SHIPPING: +0.02,
    },
    FailureCategory.CUSTOMER_ABORT: {
        # The customer actively walked away. Pushing harder is the wrong move;
        # policy stopping rules generally take over here.
        Action.IMMEDIATE_RETRY: -0.10,
        Action.DELAYED_RETRY: -0.06,
        Action.MEDIUM_DISCOUNT: -0.05,
        Action.SMALL_DISCOUNT: -0.04,
        Action.PAYMENT_LINK: -0.04,
    },
    FailureCategory.UNKNOWN_PAYMENT_FAILURE: {
        # No reliable signal: prefer cheap, low-commitment options only.
        Action.PAYMENT_LINK: +0.03,
        Action.DELAYED_RETRY: +0.03,
        Action.MEDIUM_DISCOUNT: -0.03,
    },
}

# Actions that make no sense at all given the blocker, regardless of score.
CATEGORY_INELIGIBLE: dict[FailureCategory, set[Action]] = {
    FailureCategory.CUSTOMER_ABORT: {Action.IMMEDIATE_RETRY},
}


@dataclass
class Adjustment:
    action: str
    base_probability: float
    adapted_probability: float
    delta: float
    reasons: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"action": self.action,
                "base_probability": round(self.base_probability, 6),
                "adapted_probability": round(self.adapted_probability, 6),
                "delta": round(self.delta, 6), "reasons": self.reasons}


@dataclass
class RetryContext:
    """Structured workflow evidence available to the adaptive layer (§4)."""

    attempt_number: int = 1
    previous_action: str | None = None
    previous_action_status: str | None = None
    previous_failure_reason: str | None = None
    previous_failure_step: str | None = None
    previous_failure_category: FailureCategory | None = None
    previous_payment_method: str | None = None
    previous_payment_id: str | None = None
    minutes_since_previous_attempt: float | None = None
    prior_failed_attempts: int = 0
    prior_interventions: list[str] = field(default_factory=list)
    active_incentives: list[str] = field(default_factory=list)
    cumulative_incentive_cost: Decimal = Decimal("0")
    cumulative_fixed_cost: Decimal = Decimal("0")
    customer_declined: bool = False

    @property
    def is_retry(self) -> bool:
        return self.attempt_number > 1

    @property
    def cumulative_recovery_cost(self) -> Decimal:
        return self.cumulative_incentive_cost + self.cumulative_fixed_cost

    def as_dict(self) -> dict:
        return {
            "attempt_number": self.attempt_number,
            "previous_action": self.previous_action,
            "previous_action_status": self.previous_action_status,
            "previous_failure_reason": self.previous_failure_reason,
            "previous_failure_step": self.previous_failure_step,
            "previous_failure_category": (self.previous_failure_category.value
                                          if self.previous_failure_category else None),
            "previous_payment_method": self.previous_payment_method,
            "previous_payment_id": self.previous_payment_id,
            "minutes_since_previous_attempt": self.minutes_since_previous_attempt,
            "prior_failed_attempts": self.prior_failed_attempts,
            "prior_interventions": self.prior_interventions,
            "active_incentives": self.active_incentives,
            "cumulative_incentive_cost": float(self.cumulative_incentive_cost),
            "cumulative_fixed_cost": float(self.cumulative_fixed_cost),
            "cumulative_recovery_cost": float(self.cumulative_recovery_cost),
            "customer_declined": self.customer_declined,
            "rules_version": ADAPTIVE_RULES_VERSION,
        }


def filter_eligible(actions: list[Action], retry: RetryContext) -> list[Action]:
    """Drop actions that the new blocker makes meaningless."""
    if not retry.is_retry or retry.previous_failure_category is None:
        return actions
    blocked = CATEGORY_INELIGIBLE.get(retry.previous_failure_category, set())
    return [a for a in actions if a not in blocked]


def adapt_probabilities(base: dict[str, float], retry: RetryContext) -> dict[str, Adjustment]:
    """Apply bounded, documented adjustments to model probabilities.

    On the first attempt this is the identity function: there is no workflow
    evidence yet, so the model's estimate stands unmodified.
    """
    out: dict[str, Adjustment] = {}
    for action, p in base.items():
        reasons: list[str] = []
        delta = 0.0

        if retry.is_retry:
            cat = retry.previous_failure_category
            if cat is not None:
                pref = CATEGORY_PREFERENCES.get(cat, {})
                shift = pref.get(Action(action), 0.0)
                if shift:
                    delta += shift
                    direction = "favoured" if shift > 0 else "deprioritised"
                    reasons.append(
                        f"{direction} for {cat.value} (blocker observed on the "
                        f"previous attempt)")

            # An action that already failed gets a penalty — but DO_NOTHING is
            # a baseline, not an intervention, so it is exempt.
            if (action == retry.previous_action
                    and action != Action.DO_NOTHING.value):
                delta -= REPEAT_ACTION_PENALTY
                reasons.append("already attempted unsuccessfully on this opportunity")

        delta = max(-MAX_ABS_ADJUSTMENT, min(MAX_ABS_ADJUSTMENT, delta))
        adapted = max(0.0, min(1.0, p + delta))
        out[action] = Adjustment(action, p, adapted, adapted - p, reasons)
    return out


def incremental_incentive_cost(action: str, base_incentive: Decimal,
                               retry: RetryContext) -> tuple[Decimal, str | None]:
    """Cost of granting this incentive *given what is already active* (§10).

    If free shipping was already granted on attempt 1 and remains in force,
    granting it again costs nothing incremental — the subsidy is already
    committed. Charging it twice would understate attempt 2's value and could
    wrongly push the system toward doing nothing.
    """
    if action in retry.active_incentives:
        return Decimal("0"), (
            f"{action} incentive already active from a previous attempt; "
            "no additional incremental cost")
    return base_incentive, None


def active_incentives_from_history(prior_actions: list[str]) -> list[str]:
    """Which previously executed actions leave a standing incentive.

    A discount or waived shipping fee persists for the customer. A retry or a
    payment link does not create an ongoing concession.
    """
    out = []
    for a in prior_actions:
        try:
            s = spec_for(a)
        except (KeyError, ValueError):
            continue
        if s.discount_percent > 0 or s.waives_shipping_fee:
            out.append(a)
    return list(dict.fromkeys(out))