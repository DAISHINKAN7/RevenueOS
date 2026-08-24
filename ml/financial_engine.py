"""Canonical financial engine (spec Sections 6, 7, 24, 33, 60).

This module is the single source of truth for every rupee the system reasons
about. Nothing here depends on an LLM or on a trained model: given a context, a
recovery probability and an action, the arithmetic is fully deterministic.

Canonical objective
-------------------
::

    EV(a) = P(recovery | context, a)
            * ( base_contribution_margin
                - incentive_cost_if_recovered(a)
                - expected_return_loss(a)
                - expected_cancellation_loss(a) )
            - fixed_action_cost(a)

    dEV(a) = EV(a) - EV(DO_NOTHING)

Two invariants are enforced by tests:

1. Incentive costs are conditional (paid only on recovery); fixed costs are
   unconditional. They are never mixed.
2. An incentive is subtracted exactly ONCE. `base_contribution_margin` is always
   the pre-incentive margin, so callers must not pre-net a discount into it.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from ml.actions import Action, eligible_actions, spec_for

# Cost of processing a return / cancellation beyond the lost margin itself
# (reverse logistics, restocking, payment gateway fee already sunk).
DEFAULT_RETURN_HANDLING_COST = 120.0
DEFAULT_CANCELLATION_HANDLING_COST = 25.0


class FinancialInputError(ValueError):
    """Raised when a financial input is structurally invalid.

    Deliberately loud: a malformed probability or a negative cart must never be
    silently coerced into a plausible-looking number.
    """


@dataclass(frozen=True)
class OpportunityEconomics:
    """Pre-incentive economics of a single recovery opportunity.

    Attributes
    ----------
    cart_value:
        Product revenue before any discount, excluding shipping.
    cogs:
        Cost of goods for the cart.
    shipping_cost:
        What fulfilment actually costs the merchant.
    shipping_fee_charged:
        What the customer is being asked to pay for shipping.
    return_probability, cancellation_probability:
        Post-purchase risk, used to discount the realised margin.
    """

    cart_value: float
    cogs: float
    shipping_cost: float
    shipping_fee_charged: float
    return_probability: float = 0.0
    cancellation_probability: float = 0.0
    return_handling_cost: float = DEFAULT_RETURN_HANDLING_COST
    cancellation_handling_cost: float = DEFAULT_CANCELLATION_HANDLING_COST

    def __post_init__(self) -> None:
        if self.cart_value <= 0:
            raise FinancialInputError(f"cart_value must be > 0, got {self.cart_value}")
        if self.cogs < 0 or self.shipping_cost < 0 or self.shipping_fee_charged < 0:
            raise FinancialInputError("cost inputs must be non-negative")
        for name in ("return_probability", "cancellation_probability"):
            v = getattr(self, name)
            if not (0.0 <= v <= 1.0) or math.isnan(v):
                raise FinancialInputError(f"{name} must be in [0, 1], got {v}")
        if self.return_probability + self.cancellation_probability > 1.0:
            raise FinancialInputError("return + cancellation probability exceeds 1.0")

    @property
    def base_contribution_margin(self) -> float:
        """Pre-incentive contribution margin, including shipping economics."""
        return (
            self.cart_value
            - self.cogs
            + self.shipping_fee_charged
            - self.shipping_cost
        )

    @property
    def base_margin_percent(self) -> float:
        return 100.0 * self.base_contribution_margin / self.cart_value


@dataclass(frozen=True)
class ActionValuation:
    """Full financial breakdown of one candidate action."""

    action: Action
    recovery_probability: float
    incentive_cost: float
    fixed_cost: float
    expected_return_loss: float
    expected_cancellation_loss: float
    margin_if_recovered: float
    expected_value: float
    delta_ev: float
    delta_recovery_probability: float
    recovered_gmv_net_of_discount: float
    expected_recovered_gmv: float
    maximum_downside: float

    def as_dict(self) -> dict:
        d = asdict(self)
        d["action"] = self.action.value
        return d


def _validate_probability(p: float, label: str) -> float:
    if p is None or isinstance(p, bool) or math.isnan(float(p)) or math.isinf(float(p)):
        raise FinancialInputError(f"{label} must be a finite number, got {p!r}")
    p = float(p)
    if not (0.0 <= p <= 1.0):
        raise FinancialInputError(f"{label} must be in [0, 1], got {p}")
    return p


def incentive_cost(econ: OpportunityEconomics, action: Action | str) -> float:
    """Incentive cost realised ONLY if the transaction is recovered."""
    s = spec_for(action)
    cost = econ.cart_value * s.discount_percent / 100.0
    if s.waives_shipping_fee:
        # The merchant forgoes the shipping fee revenue; the fulfilment cost is
        # already inside base_contribution_margin and is unaffected.
        cost += econ.shipping_fee_charged
    return cost


def fixed_action_cost(action: Action | str) -> float:
    """Operational cost incurred whether or not the customer converts."""
    return spec_for(action).fixed_cost


def recovered_gmv_net_of_discount(econ: OpportunityEconomics, action: Action | str) -> float:
    """Reported GMV must be net of discount (spec Section 7)."""
    s = spec_for(action)
    return econ.cart_value * (1.0 - s.discount_percent / 100.0)


def valuate_action(
    econ: OpportunityEconomics,
    action: Action | str,
    recovery_probability: float,
    baseline_probability: float | None = None,
    baseline_expected_value: float | None = None,
) -> ActionValuation:
    """Compute the full financial picture for one action.

    `baseline_*` describe DO_NOTHING and are supplied by :func:`rank_actions`.
    When omitted, dEV is reported against zero so a single action can still be
    valued in isolation (used by unit tests).
    """
    action = Action(action)
    p = _validate_probability(recovery_probability, "recovery_probability")

    inc = incentive_cost(econ, action)
    fixed = fixed_action_cost(action)

    margin_pre_risk = econ.base_contribution_margin - inc

    # Expected post-purchase losses, expressed exactly as in the spec objective.
    exp_return_loss = econ.return_probability * (
        margin_pre_risk + econ.return_handling_cost
    )
    exp_cancel_loss = econ.cancellation_probability * (
        margin_pre_risk + econ.cancellation_handling_cost
    )

    margin_if_recovered = margin_pre_risk - exp_return_loss - exp_cancel_loss
    ev = p * margin_if_recovered - fixed

    net_gmv = recovered_gmv_net_of_discount(econ, action)

    return ActionValuation(
        action=action,
        recovery_probability=p,
        incentive_cost=inc,
        fixed_cost=fixed,
        expected_return_loss=exp_return_loss,
        expected_cancellation_loss=exp_cancel_loss,
        margin_if_recovered=margin_if_recovered,
        expected_value=ev,
        delta_ev=ev - (baseline_expected_value or 0.0),
        delta_recovery_probability=p - (baseline_probability if baseline_probability is not None else 0.0),
        recovered_gmv_net_of_discount=net_gmv,
        expected_recovered_gmv=p * net_gmv,
        maximum_downside=_worst_case_downside(econ, inc, fixed),
    )


def _worst_case_downside(econ: OpportunityEconomics, inc: float, fixed: float) -> float:
    """Maximum authorised financial downside for a single action (Section 60).

    The bounded loss the merchant can suffer is the operational cost plus, in
    the event the recovered order is returned, the incentive granted and the
    return handling cost.
    """
    return fixed + inc + econ.return_handling_cost * econ.return_probability


def rank_actions(
    econ: OpportunityEconomics,
    probabilities: dict[Action | str, float],
    opportunity_type: str = "CHECKOUT_ABANDONMENT",
) -> list[ActionValuation]:
    """Value every eligible action and sort by dEV vs DO_NOTHING, descending.

    DO_NOTHING must be present in `probabilities`: incremental value is
    undefined without the no-intervention counterfactual.
    """
    probs = {Action(k): v for k, v in probabilities.items()}
    if Action.DO_NOTHING not in probs:
        raise FinancialInputError("probabilities must include DO_NOTHING")

    allowed = set(eligible_actions(opportunity_type))
    unknown = set(probs) - allowed
    if unknown:
        raise FinancialInputError(
            f"actions not eligible for {opportunity_type}: {sorted(a.value for a in unknown)}"
        )

    baseline = valuate_action(econ, Action.DO_NOTHING, probs[Action.DO_NOTHING])

    valuations = [
        valuate_action(
            econ, a, p,
            baseline_probability=baseline.recovery_probability,
            baseline_expected_value=baseline.expected_value,
        )
        for a, p in probs.items()
    ]
    valuations.sort(key=lambda v: (-v.delta_ev, v.action.value))
    return valuations


def select_best_action(
    econ: OpportunityEconomics,
    probabilities: dict[Action | str, float],
    opportunity_type: str = "CHECKOUT_ABANDONMENT",
) -> ActionValuation:
    """Pick the highest-dEV action; fall back to DO_NOTHING when no dEV > 0.

    This is the behaviour required by Section 6: restraint is the default, not
    a special case.
    """
    ranked = rank_actions(econ, probabilities, opportunity_type)
    best = ranked[0]
    if best.delta_ev <= 0 or best.action is Action.DO_NOTHING:
        return next(v for v in ranked if v.action is Action.DO_NOTHING)
    return best
