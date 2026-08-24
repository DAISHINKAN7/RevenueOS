"""Historical logging policy with stored propensities (spec Sections 35-37).

This is the single most important design decision in the data layer.

Why propensities matter
-----------------------
If historical merchants always gave free shipping to high-LTV customers and
those customers converted, observing (high LTV + free shipping + conversion)
does not establish that free shipping caused anything. The logged policy is
confounded with the context.

Two mitigations are implemented here:

1. **Stochastic assignment with recorded P(action | context).** Every logged
   row stores the exact probability with which its action was chosen, which is
   what makes IPS / SNIPS / doubly robust estimation valid.
2. **A randomised exploration cohort** (default 25% of opportunities) where the
   action is drawn from a fixed, context-independent distribution over the
   eligible action set, guaranteeing support for action/context combinations
   the greedy historical policy would never produce.

Stratified exploration (revision 1.1.0)
---------------------------------------
Uniform exploration produced only ~19 held-out `DELAYED_RETRY` observations,
because retry actions are eligible only for payment-failure opportunities and
were then split nine ways. Exploration now uses `EXPLORATION_WEIGHTS`, which
over-samples the retry actions *within the contexts where they are already
legitimately eligible*. Retry actions are still never assigned to checkout
abandonment. The weights are fixed and context-independent, so the exploration
distribution remains a valid, exactly-known logging policy.

Every action retains at least `min_propensity` probability so importance
weights stay finite.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml.actions import Action, eligible_actions

# Context-free base preference of the historical merchant (Section 35).
# Deliberately mediocre: heavy on doing nothing and on blanket discounting.
BASE_PREFERENCE = {
    Action.DO_NOTHING: 0.36,
    Action.SMALL_DISCOUNT: 0.22,
    Action.FREE_SHIPPING: 0.12,
    Action.MEDIUM_DISCOUNT: 0.08,
    Action.IMMEDIATE_RETRY: 0.09,
    Action.DELAYED_RETRY: 0.06,
    Action.PAYMENT_METHOD_SWITCH: 0.04,
    Action.PAYMENT_LINK: 0.02,
    Action.HUMAN_ESCALATION: 0.01,
}


# Exploration weights over the eligible set. Retry actions are over-sampled
# because they are eligible only for payment failures and would otherwise carry
# too little held-out support for per-action off-policy estimation.
# These are RELATIVE weights, renormalised over whichever actions are eligible.
EXPLORATION_WEIGHTS = {
    Action.DO_NOTHING: 1.0,
    Action.SMALL_DISCOUNT: 1.0,
    Action.MEDIUM_DISCOUNT: 1.0,
    Action.FREE_SHIPPING: 1.0,
    Action.PAYMENT_METHOD_SWITCH: 1.0,
    Action.PAYMENT_LINK: 1.0,
    Action.HUMAN_ESCALATION: 1.0,
    Action.IMMEDIATE_RETRY: 2.2,
    Action.DELAYED_RETRY: 2.6,
}


def _context_multipliers(row_ctx: dict, actions: tuple[Action, ...], i: int) -> np.ndarray:
    """Heuristic, imperfect merchant reasoning — NOT the optimal policy.

    These rules mimic how a real merchant would plausibly behave: discount big
    carts, retry on timeouts, escalate rarely. They are correlated with context,
    which is exactly the confounding that propensity logging corrects for.
    """
    mult = np.ones(len(actions))
    cart = row_ctx["cart_value"][i]
    reason = row_ctx["failure_reason"][i]
    is_failure = row_ctx["opportunity_type"][i] == "PAYMENT_FAILURE"
    fee = row_ctx["shipping_fee_charged"][i]

    for j, a in enumerate(actions):
        if a in (Action.SMALL_DISCOUNT, Action.MEDIUM_DISCOUNT):
            mult[j] *= 1.6 if cart > 5000 else 0.9
            if is_failure:
                mult[j] *= 0.5          # merchants discount abandonment, not failures
        elif a is Action.FREE_SHIPPING:
            mult[j] *= 1.8 if fee > 0 else 0.15
        elif a is Action.IMMEDIATE_RETRY:
            mult[j] *= 2.4 if reason in ("BANK_TIMEOUT", "UPI_TIMEOUT", "NETWORK_ERROR") else 0.8
        elif a is Action.DELAYED_RETRY:
            mult[j] *= 1.9 if reason in ("INSUFFICIENT_FUNDS", "BANK_TIMEOUT") else 0.7
        elif a is Action.PAYMENT_METHOD_SWITCH:
            mult[j] *= 2.2 if reason in ("CARD_DECLINED", "AUTHENTICATION_FAILURE") else 0.9
        elif a is Action.HUMAN_ESCALATION:
            mult[j] *= 2.0 if cart > 15000 else 0.4
        elif a is Action.DO_NOTHING:
            mult[j] *= 0.7 if cart > 8000 else 1.25
    return mult


def assign_actions(
    ctx: dict,
    rng: np.random.Generator,
    exploration_rate: float,
    min_propensity: float,
) -> pd.DataFrame:
    """Choose one logged action per opportunity and record its exact propensity.

    Returns a frame with `action_taken`, `action_propensity`, `is_exploration`
    and the full propensity vector (needed for doubly robust estimation).
    """
    n = len(ctx["cart_value"])
    is_exploration = rng.random(n) < exploration_rate

    actions_by_type = {
        "CHECKOUT_ABANDONMENT": eligible_actions("CHECKOUT_ABANDONMENT"),
        "PAYMENT_FAILURE": eligible_actions("PAYMENT_FAILURE"),
    }

    taken, propensity, is_expl = [], [], []
    prop_vectors = []

    for i in range(n):
        acts = actions_by_type[ctx["opportunity_type"][i]]

        if is_exploration[i]:
            # Stratified randomisation over eligible actions: fixed weights,
            # renormalised over the eligible set. Context-independent, so the
            # propensity is exactly known.
            p = np.array([EXPLORATION_WEIGHTS[a] for a in acts])
            p = p / p.sum()
        else:
            base = np.array([BASE_PREFERENCE[a] for a in acts])
            p = base * _context_multipliers(ctx, acts, i)
            p = p / p.sum()
            # Floor every action so importance weights stay finite (Section 36).
            p = np.maximum(p, min_propensity)
            p = p / p.sum()

        j = rng.choice(len(acts), p=p)
        taken.append(acts[j].value)
        propensity.append(float(p[j]))
        is_expl.append(bool(is_exploration[i]))
        prop_vectors.append({a.value: float(pv) for a, pv in zip(acts, p)})

    return pd.DataFrame({
        "action_taken": taken,
        "action_propensity": propensity,
        "is_exploration": is_expl,
        "propensity_vector": prop_vectors,
    })
