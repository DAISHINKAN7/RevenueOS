"""Finite recovery action space (spec Section 9).

The action space is CLOSED. Neither the LLM nor the optimizer may invent new
financial actions; they may only select from `ALL_ACTIONS`.

Each action carries deterministic cost semantics used by the financial engine:

* ``discount_percent``      -> incentive cost realised ONLY on recovery
* ``waives_shipping_fee``   -> incentive cost realised ONLY on recovery
* ``fixed_cost``            -> operational cost incurred REGARDLESS of outcome
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Action(str, Enum):
    DO_NOTHING = "DO_NOTHING"
    FREE_SHIPPING = "FREE_SHIPPING"
    SMALL_DISCOUNT = "SMALL_DISCOUNT"
    MEDIUM_DISCOUNT = "MEDIUM_DISCOUNT"
    PAYMENT_METHOD_SWITCH = "PAYMENT_METHOD_SWITCH"
    IMMEDIATE_RETRY = "IMMEDIATE_RETRY"
    DELAYED_RETRY = "DELAYED_RETRY"
    PAYMENT_LINK = "PAYMENT_LINK"
    HUMAN_ESCALATION = "HUMAN_ESCALATION"


@dataclass(frozen=True)
class ActionSpec:
    """Deterministic cost semantics for one action."""

    action: Action
    discount_percent: float = 0.0
    waives_shipping_fee: bool = False
    fixed_cost: float = 0.0
    requires_payment_failure: bool = False
    description: str = ""


# Fixed costs are INR operational costs (messaging, API, agent handling time).
ACTION_SPECS: dict[Action, ActionSpec] = {
    Action.DO_NOTHING: ActionSpec(
        Action.DO_NOTHING, fixed_cost=0.0,
        description="No intervention. Baseline for incremental value.",
    ),
    Action.FREE_SHIPPING: ActionSpec(
        Action.FREE_SHIPPING, waives_shipping_fee=True, fixed_cost=2.0,
        description="Waive the shipping fee charged to the customer.",
    ),
    Action.SMALL_DISCOUNT: ActionSpec(
        Action.SMALL_DISCOUNT, discount_percent=5.0, fixed_cost=2.0,
        description="5% order discount.",
    ),
    Action.MEDIUM_DISCOUNT: ActionSpec(
        Action.MEDIUM_DISCOUNT, discount_percent=10.0, fixed_cost=2.0,
        description="10% order discount. Usually exceeds autonomous policy limits.",
    ),
    Action.PAYMENT_METHOD_SWITCH: ActionSpec(
        Action.PAYMENT_METHOD_SWITCH, fixed_cost=2.0,
        description="Suggest an alternative payment instrument.",
    ),
    Action.IMMEDIATE_RETRY: ActionSpec(
        Action.IMMEDIATE_RETRY, fixed_cost=1.0, requires_payment_failure=True,
        description="Retry the payment immediately.",
    ),
    Action.DELAYED_RETRY: ActionSpec(
        Action.DELAYED_RETRY, fixed_cost=1.0, requires_payment_failure=True,
        description="Retry the payment after a delay window.",
    ),
    Action.PAYMENT_LINK: ActionSpec(
        Action.PAYMENT_LINK, fixed_cost=3.0,
        description="Send a hosted payment link.",
    ),
    Action.HUMAN_ESCALATION: ActionSpec(
        Action.HUMAN_ESCALATION, fixed_cost=130.0,
        description="Route to a human agent. High fixed cost, no incentive cost.",
    ),
}

ALL_ACTIONS: tuple[Action, ...] = tuple(ACTION_SPECS)


def spec_for(action: Action | str) -> ActionSpec:
    return ACTION_SPECS[Action(action)]


def eligible_actions(opportunity_type: str) -> tuple[Action, ...]:
    """Actions structurally valid for an opportunity type.

    Retry actions are meaningless without a prior payment attempt, so they are
    excluded for pure checkout abandonment.
    """
    is_failure = opportunity_type == "PAYMENT_FAILURE"
    return tuple(
        a for a in ALL_ACTIONS
        if is_failure or not ACTION_SPECS[a].requires_payment_failure
    )
