"""Customer generation with a strict hidden/observable split.

Two frames come out of here:

* ``customers``        - observable, safe to feed the model
* ``customers_hidden`` - latent traits used ONLY by the response surface

The split is the reason evaluation is meaningful. If `hidden_price_sensitivity`
leaked into the feature matrix, the model would trivially recover the response
surface and every metric would be fiction.

Observable history features are deliberately NOT copies of the latent traits.
They are finite event counts (offers seen / offers redeemed), so a customer with
3 coupon offers carries genuine sampling noise and a new customer carries almost
no information (spec Section 22).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SEGMENTS = [
    "LOYAL", "PRICE_SENSITIVE", "CONVENIENCE_SENSITIVE",
    "HIGH_LTV", "DEAL_SEEKER", "PAYMENT_FRICTION", "AT_RISK", "NEW_CUSTOMER",
]
SEGMENT_WEIGHTS = [0.14, 0.18, 0.14, 0.09, 0.12, 0.11, 0.10, 0.12]

PAYMENT_METHODS = ["UPI", "CARD", "NETBANKING", "WALLET", "COD"]

# Segment -> (price_sens, shipping_sens, payment_friction, retry_tolerance,
#             brand_loyalty, impulsivity) expressed as Beta(a, b) means.
SEGMENT_TRAITS = {
    "LOYAL":                 (0.25, 0.30, 0.20, 0.70, 0.85, 0.45),
    "PRICE_SENSITIVE":       (0.80, 0.45, 0.30, 0.45, 0.30, 0.40),
    "CONVENIENCE_SENSITIVE": (0.35, 0.82, 0.30, 0.50, 0.55, 0.55),
    "HIGH_LTV":              (0.20, 0.35, 0.25, 0.75, 0.75, 0.60),
    "DEAL_SEEKER":           (0.88, 0.60, 0.35, 0.40, 0.20, 0.65),
    "PAYMENT_FRICTION":      (0.45, 0.40, 0.85, 0.35, 0.45, 0.45),
    "AT_RISK":               (0.55, 0.50, 0.45, 0.30, 0.20, 0.35),
    "NEW_CUSTOMER":          (0.50, 0.55, 0.40, 0.50, 0.35, 0.55),
}

CITY_TIER_P = [0.45, 0.35, 0.20]


def _beta_around(rng: np.random.Generator, mean: float, n: int, concentration: float = 8.0):
    """Sample Beta values centred on `mean` with moderate spread."""
    a = max(0.15, mean * concentration)
    b = max(0.15, (1 - mean) * concentration)
    return rng.beta(a, b, size=n)


def generate_customers(
    n_customers: int, rng: np.random.Generator, start_date: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    start = pd.Timestamp(start_date)
    segments = rng.choice(SEGMENTS, size=n_customers, p=SEGMENT_WEIGHTS)

    traits = np.array([SEGMENT_TRAITS[s] for s in segments])
    hidden = {
        "hidden_price_sensitivity": np.zeros(n_customers),
        "hidden_shipping_sensitivity": np.zeros(n_customers),
        "hidden_payment_friction": np.zeros(n_customers),
        "hidden_retry_tolerance": np.zeros(n_customers),
        "hidden_brand_loyalty": np.zeros(n_customers),
        "hidden_impulsivity": np.zeros(n_customers),
    }
    for j, key in enumerate(hidden):
        for s in SEGMENTS:
            mask = segments == s
            if mask.any():
                hidden[key][mask] = _beta_around(rng, traits[mask][0, j], int(mask.sum()))

    # Tenure drives how much observable history exists.
    tenure_days = rng.integers(1, 900, size=n_customers)
    tenure_days = np.where(segments == "NEW_CUSTOMER", rng.integers(1, 60, size=n_customers), tenure_days)

    activity = np.clip(rng.gamma(1.6, 1.0, size=n_customers), 0.05, None)
    activity *= np.where(np.isin(segments, ["HIGH_LTV", "LOYAL"]), 2.2, 1.0)
    orders_lifetime = rng.poisson(activity * tenure_days / 90.0)

    aov = np.exp(rng.normal(np.log(2800), 0.72, size=n_customers))
    aov *= np.where(segments == "HIGH_LTV", 2.1, 1.0)

    customer_ids = np.array([f"C{i + 1:06d}" for i in range(n_customers)])

    # ---- finite-count response history (Section 22) ----------------------
    # Offers seen scale with tenure; redemption is a Binomial draw against the
    # LATENT trait, so the observable rate is a noisy estimate of it rather
    # than a copy.
    coupon_seen = rng.poisson(np.clip(orders_lifetime * 0.55, 0, 14))
    coupon_redeemed = rng.binomial(coupon_seen, np.clip(hidden["hidden_price_sensitivity"] * 0.8, 0.01, 0.95))

    ship_seen = rng.poisson(np.clip(orders_lifetime * 0.40, 0, 12))
    ship_redeemed = rng.binomial(ship_seen, np.clip(hidden["hidden_shipping_sensitivity"] * 0.8, 0.01, 0.95))

    retry_seen = rng.poisson(np.clip(orders_lifetime * 0.25, 0, 10))
    retry_success = rng.binomial(retry_seen, np.clip(hidden["hidden_retry_tolerance"] * 0.75, 0.01, 0.95))

    prev_failures = rng.poisson(np.clip(hidden["hidden_payment_friction"] * orders_lifetime * 0.5, 0, 12))

    def safe_rate(num, den):
        return np.where(den > 0, num / np.maximum(den, 1), np.nan)

    observable = pd.DataFrame({
        "customer_id": customer_ids,
        "signup_date": [start - pd.Timedelta(days=int(d)) for d in tenure_days],
        "customer_segment": segments,
        "city_tier": rng.choice([1, 2, 3], size=n_customers, p=CITY_TIER_P),
        "tenure_days": tenure_days,
        "orders_lifetime": orders_lifetime,
        "orders_last_90d": rng.binomial(orders_lifetime, 0.35),
        "orders_last_30d": rng.binomial(orders_lifetime, 0.14),
        "average_order_value": np.round(aov, 2),
        "lifetime_value": np.round(aov * orders_lifetime, 2),
        "days_since_last_purchase": np.where(
            orders_lifetime > 0, rng.integers(1, 240, size=n_customers), -1
        ),
        "previous_checkout_abandonments": rng.poisson(np.clip(orders_lifetime * 0.6, 0, 20)),
        "previous_payment_failures": prev_failures,
        "coupon_offers_seen": coupon_seen,
        "coupon_offers_redeemed": coupon_redeemed,
        "coupon_response_rate": safe_rate(coupon_redeemed, coupon_seen),
        "free_shipping_offers_seen": ship_seen,
        "free_shipping_offers_redeemed": ship_redeemed,
        "free_shipping_response_rate": safe_rate(ship_redeemed, ship_seen),
        "retry_offers_seen": retry_seen,
        "retry_offers_succeeded": retry_success,
        "retry_response_rate": safe_rate(retry_success, retry_seen),
        "preferred_payment_method": rng.choice(PAYMENT_METHODS, size=n_customers,
                                               p=[0.42, 0.26, 0.12, 0.10, 0.10]),
        "historical_return_count": rng.poisson(np.clip(orders_lifetime * 0.08, 0, 10)),
        "historical_cancellation_count": rng.poisson(np.clip(orders_lifetime * 0.05, 0, 10)),
    })
    observable["historical_return_rate"] = safe_rate(
        observable["historical_return_count"], observable["orders_lifetime"]
    )
    observable["historical_cancellation_rate"] = safe_rate(
        observable["historical_cancellation_count"], observable["orders_lifetime"]
    )

    hidden_df = pd.DataFrame({"customer_id": customer_ids, **hidden})
    hidden_df["hidden_return_propensity"] = np.clip(rng.beta(2, 30, size=n_customers), 0, 0.5)
    hidden_df["hidden_cancellation_propensity"] = np.clip(rng.beta(1.5, 40, size=n_customers), 0, 0.35)

    return observable, hidden_df
