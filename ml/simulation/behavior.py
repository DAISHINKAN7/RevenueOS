"""Behavioural response surface (spec Sections 32, 33).

This module defines the hidden ground truth:

    P(recovery | context, action)

for EVERY action, not just the one that was historically taken. That full
surface is what makes counterfactual (oracle) evaluation possible, and it is
never exposed to the model.

Design intent
-------------
The surface must contain learnable structure without being trivially learnable:

* Latent traits drive action-specific uplift, and each trait has a *matched*
  intervention that dominates in its own regime.
* Failure reason gates which families of intervention can work at all.
  Discounts do essentially nothing for infrastructure failures: a bank timeout
  is not a pricing objection.
* Loyal / high-LTV customers carry a high unaided baseline, so intervention
  produces little incremental value for them.
* Hidden environment windows shift outcomes with no corresponding feature.
* Logit noise (`response_logit_noise_sd`) caps achievable accuracy.

If a trained model reaches near-perfect calibration against this surface, the
noise term is too small — that is a simulator bug, not a success.

Revision 1.1.0
--------------
Discount uplift was previously strong enough that MEDIUM_DISCOUNT was the
probability-maximising action for ~66% of opportunities, making the surface
discount-dominant and weakening the product thesis. Targeted changes:

1. Discount uplift reduced; diminishing-returns multiplier tightened 1.75 -> 1.35.
2. Discounts heavily damped for infrastructure failure reasons.
3. Matched interventions (free shipping, payment switch, delayed retry)
   strengthened so they win decisively in their own regimes.
4. Unaided baseline raised for loyal customers so DO_NOTHING is competitive.
"""

from __future__ import annotations

import numpy as np

from ml.actions import Action

# Baseline no-intervention recovery propensity, in logit space.
BASE_LOGIT = -1.15

# Global damping on ALL action uplifts (revision 1.1.0).
#
# The uncalibrated surface produced mean uplifts of roughly +0.25 absolute
# probability for a recovery nudge, which is far beyond anything observed in
# real recovery programmes and made intervention unconditionally profitable —
# DO_NOTHING was economics-optimal in 0% of cases. Damping brings typical
# uplift into a defensible 0.03-0.12 band, which restores a genuine
# cost/benefit trade-off and lets restraint emerge from the economics rather
# than being hard-coded.
UPLIFT_SCALE = 0.42

# Intervention fatigue (revision 1.1.0).
#
# Contacting a customer who was already going to return is not free even when
# the message costs ~nothing: it trains discount-seeking, dilutes the brand and
# burns list tolerance. Modelled as a logit penalty proportional to brand
# loyalty and applied to every contact-based action. For highly loyal customers
# this drives net uplift toward zero or slightly negative, which is what makes
# DO_NOTHING genuinely win on economics rather than never being selected.
FATIGUE_PENALTY = 0.85

# Actions that involve contacting the customer (DO_NOTHING and silent retries
# do not).
CONTACT_ACTIONS = frozenset({
    Action.FREE_SHIPPING, Action.SMALL_DISCOUNT, Action.MEDIUM_DISCOUNT,
    Action.PAYMENT_METHOD_SWITCH, Action.PAYMENT_LINK, Action.HUMAN_ESCALATION,
})

# Half-life (minutes) of recovery probability after abandonment, by cause.
# Checkout abandonment decays fast; bank-side failures stay recoverable longer.
DECAY_HALFLIFE_MINUTES = {
    "CHECKOUT_ABANDONMENT": 90.0,
    "PAYMENT_FAILURE": 240.0,
}

# Failure reasons caused by infrastructure rather than customer intent.
# A price cut does not fix a bank timeout, so discounts are damped here.
INFRASTRUCTURE_FAILURES = frozenset({
    "BANK_TIMEOUT", "NETWORK_ERROR", "UPI_TIMEOUT", "AUTHENTICATION_FAILURE",
})
DISCOUNT_DAMPING_INFRASTRUCTURE = 0.30

# INSUFFICIENT_FUNDS is a genuine affordability constraint, so a discount can
# legitimately help — but only partially, since the shortfall may exceed it.
DISCOUNT_DAMPING_FUNDS = 0.70

# Retry viability by synthetic failure reason: (immediate, delayed).
# Encodes the domain logic in Section 33.
RETRY_UPLIFT = {
    "BANK_TIMEOUT":           (0.45, 2.55),
    "NETWORK_ERROR":          (1.05, 1.95),
    "UPI_TIMEOUT":            (0.70, 2.15),
    "INSUFFICIENT_FUNDS":     (-0.45, 1.55),
    "AUTHENTICATION_FAILURE": (0.20, 0.65),
    "CARD_DECLINED":          (-0.25, 0.20),
    "USER_CANCELLED":         (-0.35, -0.15),
    "UNKNOWN":                (0.25, 0.75),
    "NONE":                   (0.0, 0.0),
}

# Payment-method switch is powerful exactly where the instrument was the problem.
SWITCH_UPLIFT = {
    "CARD_DECLINED": 2.45,
    "AUTHENTICATION_FAILURE": 2.10,
    "UPI_TIMEOUT": 1.70,
    "BANK_TIMEOUT": 1.15,
    "INSUFFICIENT_FUNDS": 0.75,
    "NETWORK_ERROR": 0.85,
    "USER_CANCELLED": 0.15,
    "UNKNOWN": 0.60,
    "NONE": 0.35,
}

# Payment link helps most where the original attempt hit instrument friction.
LINK_UPLIFT = {
    "AUTHENTICATION_FAILURE": 1.55,
    "CARD_DECLINED": 1.35,
    "UPI_TIMEOUT": 1.25,
    "NETWORK_ERROR": 1.15,
    "BANK_TIMEOUT": 0.85,
    "INSUFFICIENT_FUNDS": 0.55,
    "USER_CANCELLED": 0.30,
    "UNKNOWN": 0.70,
    "NONE": 0.60,
}


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def time_decay(minutes: np.ndarray, opportunity_type: np.ndarray) -> np.ndarray:
    """Multiplicative decay applied in probability space (Section 33)."""
    halflife = np.where(
        opportunity_type == "PAYMENT_FAILURE",
        DECAY_HALFLIFE_MINUTES["PAYMENT_FAILURE"],
        DECAY_HALFLIFE_MINUTES["CHECKOUT_ABANDONMENT"],
    )
    return np.power(0.5, np.asarray(minutes, dtype=float) / halflife)


def base_logit(ctx: dict) -> np.ndarray:
    """No-intervention recovery logit before any action effect."""
    n = len(ctx["cart_value"])
    z = np.full(n, BASE_LOGIT)

    # Loyal / high-intent customers come back on their own. Strengthened in
    # 1.1.0 so DO_NOTHING is genuinely competitive for them and little
    # incremental value is left for any intervention to capture.
    z += 2.30 * ctx["hidden_brand_loyalty"]
    z += 0.65 * ctx["hidden_impulsivity"]

    # Payment friction suppresses unaided recovery.
    z -= 0.95 * ctx["hidden_payment_friction"]

    # Large carts are considered more carefully.
    z -= 0.28 * np.log1p(ctx["cart_value"] / 3000.0)

    # A visible shipping fee that is large relative to the cart hurts.
    fee_ratio = ctx["shipping_fee_charged"] / np.maximum(ctx["cart_value"], 1.0)
    z -= 2.20 * ctx["hidden_shipping_sensitivity"] * np.clip(fee_ratio, 0, 0.25)

    # Hidden environment: no matching feature exists for these.
    z -= 0.60 * ctx["in_competitor_sale"]
    z += 0.35 * ctx["is_payday"]
    z -= 0.30 * ctx["in_bank_outage"]

    # Later attempts are progressively harder.
    z -= 0.40 * np.maximum(0, ctx["attempt_number"] - 1)
    return z


def _discount_damping(reason: np.ndarray) -> np.ndarray:
    """Discounts do not fix infrastructure problems."""
    d = np.ones(len(reason))
    d[np.isin(reason, list(INFRASTRUCTURE_FAILURES))] = DISCOUNT_DAMPING_INFRASTRUCTURE
    d[reason == "INSUFFICIENT_FUNDS"] = DISCOUNT_DAMPING_FUNDS
    d[reason == "CARD_DECLINED"] = 0.45
    d[reason == "USER_CANCELLED"] = 0.85
    return d


def action_logit_uplift(action: Action, ctx: dict) -> np.ndarray:
    """Action-specific uplift in logit space."""
    n = len(ctx["cart_value"])
    reason = ctx["failure_reason"]

    if action is Action.DO_NOTHING:
        return np.zeros(n)

    if action is Action.FREE_SHIPPING:
        fee_ratio = ctx["shipping_fee_charged"] / np.maximum(ctx["cart_value"], 1.0)
        # Worthless if no fee was being charged in the first place.
        has_fee = (ctx["shipping_fee_charged"] > 0).astype(float)
        # Strengthened trait coupling: a convenience-sensitive customer facing a
        # visible fee is the regime where this action should dominate.
        return has_fee * (0.35 + 3.90 * ctx["hidden_shipping_sensitivity"]
                          + 9.0 * np.clip(fee_ratio, 0, 0.20))

    if action in (Action.SMALL_DISCOUNT, Action.MEDIUM_DISCOUNT):
        # Diminishing returns tightened: doubling the discount adds ~35%, not 75%.
        strength = 1.0 if action is Action.SMALL_DISCOUNT else 1.35
        raw = 0.15 + 1.85 * ctx["hidden_price_sensitivity"]
        return strength * raw * _discount_damping(reason)

    if action is Action.PAYMENT_METHOD_SWITCH:
        base = np.array([SWITCH_UPLIFT.get(r, 0.35) for r in reason])
        return base * (0.55 + 1.25 * ctx["hidden_payment_friction"])

    if action in (Action.IMMEDIATE_RETRY, Action.DELAYED_RETRY):
        idx = 0 if action is Action.IMMEDIATE_RETRY else 1
        base = np.array([RETRY_UPLIFT.get(r, (0.0, 0.0))[idx] for r in reason])
        base = base * (0.60 + 0.90 * ctx["hidden_retry_tolerance"])
        if action is Action.IMMEDIATE_RETRY:
            # An immediate retry during a bank outage mostly fails again.
            base = base - 0.95 * ctx["in_bank_outage"]
        else:
            base = base + 0.70 * ctx["in_bank_outage"]
        return base

    if action is Action.PAYMENT_LINK:
        base = np.array([LINK_UPLIFT.get(r, 0.60) for r in reason])
        return base * (0.70 + 0.60 * ctx["hidden_impulsivity"]) \
            - 0.30 * ctx["hidden_payment_friction"]

    if action is Action.HUMAN_ESCALATION:
        # Effective almost everywhere — but its high fixed cost makes it
        # uneconomic on small carts, which is exactly the trade-off the optimiser must
        # discover on its own.
        return np.full(n, 0.95)

    raise ValueError(f"unhandled action {action}")


def response_surface(
    ctx: dict, actions: tuple[Action, ...], rng: np.random.Generator, noise_sd: float
) -> dict[Action, np.ndarray]:
    """Full counterfactual surface: {action -> P(recovery)} for every row.

    A single shared noise draw per row is applied to all actions so the noise
    represents unobserved *context*, not independent per-action measurement
    error. This keeps relative action ordering meaningful while making the
    absolute level unpredictable.
    """
    n = len(ctx["cart_value"])
    shared_noise = rng.normal(0.0, noise_sd, size=n)
    z0 = base_logit(ctx) + shared_noise
    decay = time_decay(ctx["minutes_since_event"], ctx["opportunity_type"])

    out: dict[Action, np.ndarray] = {}
    fatigue = FATIGUE_PENALTY * ctx["hidden_brand_loyalty"]
    for a in actions:
        z = z0 + UPLIFT_SCALE * action_logit_uplift(a, ctx)
        if a in CONTACT_ACTIONS:
            z = z - fatigue
        p = _sigmoid(z) * decay
        out[a] = np.clip(p, 0.001, 0.985)
    return out
