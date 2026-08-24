"""Sessions, checkouts and payment attempts.

The funnel is: session -> (maybe) checkout -> (maybe) abandon OR payment attempt
-> (maybe) failure. Abandonment and failure both produce recovery opportunities.

Customer activity is concentrated (a minority of customers drive most sessions),
matching the concentration curve observed in public retail transaction data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml.simulation.environment import HiddenEnvironment
from ml.simulation.products import shipping_cost_for

DEVICES = ["MOBILE_APP", "MOBILE_WEB", "DESKTOP"]
TRAFFIC = ["ORGANIC", "PAID_SEARCH", "SOCIAL", "EMAIL", "DIRECT", "REFERRAL"]
NETWORKS = ["WIFI", "MOBILE_4G", "MOBILE_5G", "MOBILE_3G"]

FAILURE_REASONS = [
    "INSUFFICIENT_FUNDS", "BANK_TIMEOUT", "NETWORK_ERROR",
    "AUTHENTICATION_FAILURE", "UPI_TIMEOUT", "CARD_DECLINED",
    "USER_CANCELLED", "UNKNOWN",
]
FAILURE_BASE_P = np.array([0.18, 0.14, 0.10, 0.11, 0.16, 0.15, 0.10, 0.06])

ABANDON_STAGES = ["CART", "ADDRESS", "SHIPPING", "PAYMENT_SELECTION", "PAYMENT_PROCESSING"]
ABANDON_STAGE_P = [0.30, 0.14, 0.18, 0.24, 0.14]

# Free-shipping threshold: carts above this ship free, so a shipping incentive
# is worthless on them. This creates genuine action-eligibility structure.
FREE_SHIPPING_THRESHOLD = 4_999.0


def generate_sessions(
    cfg,
    rng: np.random.Generator,
    customers: pd.DataFrame,
    customers_hidden: pd.DataFrame,
    products: pd.DataFrame,
    env: HiddenEnvironment,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = cfg.n_sessions
    start = pd.Timestamp(cfg.start_date)

    # Concentrated activity: Zipf-like customer selection.
    weights = rng.gamma(1.3, 1.0, size=len(customers))
    weights /= weights.sum()
    cust_idx = rng.choice(len(customers), size=n, p=weights)

    # Timestamps with diurnal structure (evening peak).
    day_offset = rng.integers(0, cfg.days, size=n)
    hour = rng.choice(
        24, size=n,
        p=_normalise([0.4, 0.25, 0.15, 0.1, 0.1, 0.2, 0.5, 0.9, 1.2, 1.4, 1.5, 1.5,
                      1.4, 1.3, 1.3, 1.4, 1.6, 1.9, 2.3, 2.6, 2.5, 2.0, 1.3, 0.8]),
    )
    minute = rng.integers(0, 60, size=n)
    ts = start + pd.to_timedelta(day_offset, "D") + pd.to_timedelta(hour, "h") + pd.to_timedelta(minute, "m")

    cust = customers.iloc[cust_idx].reset_index(drop=True)
    hid = customers_hidden.iloc[cust_idx].reset_index(drop=True)

    # Cart construction: 1-4 items drawn around the customer's typical AOV.
    n_items = rng.choice([1, 2, 3, 4], size=n, p=[0.55, 0.27, 0.12, 0.06])
    prod_idx = rng.integers(0, len(products), size=n)
    unit_price = products["selling_price"].values[prod_idx]
    unit_cogs = products["cost_of_goods"].values[prod_idx]
    unit_weight = products["weight_kg"].values[prod_idx]
    unit_ship = products["base_shipping_cost"].values[prod_idx]
    unit_return = products["historical_return_rate"].values[prod_idx]

    cart_value = np.round(unit_price * n_items, 2)
    cart_cogs = np.round(unit_cogs * n_items, 2)
    cart_weight = np.round(unit_weight * n_items, 3)

    courier = env.in_courier_disruption(ts)
    shipping_cost = np.array([
        shipping_cost_for(bs * ni, int(ct), bool(cd))
        for bs, ni, ct, cd in zip(unit_ship, n_items, cust["city_tier"].values, courier)
    ])
    shipping_fee_charged = np.where(cart_value >= FREE_SHIPPING_THRESHOLD, 0.0,
                                    np.round(shipping_cost * 0.85, 2))

    sessions = pd.DataFrame({
        "session_id": [f"S{i + 1:07d}" for i in range(n)],
        "customer_id": cust["customer_id"].values,
        "timestamp": ts,
        "device_type": rng.choice(DEVICES, size=n, p=[0.46, 0.30, 0.24]),
        "traffic_source": rng.choice(TRAFFIC, size=n, p=[0.28, 0.18, 0.16, 0.12, 0.18, 0.08]),
        "network_context": rng.choice(NETWORKS, size=n, p=[0.40, 0.34, 0.18, 0.08]),
        "pages_viewed": rng.poisson(6, size=n) + 1,
        "products_viewed": rng.poisson(3, size=n) + 1,
        "session_duration_seconds": np.round(rng.gamma(2.2, 110, size=n), 0),
        "cart_additions": n_items,
        "cart_value": cart_value,
        "cart_cogs": cart_cogs,
        "cart_weight_kg": cart_weight,
        "shipping_cost": shipping_cost,
        "shipping_fee_shown": shipping_fee_charged,
        "product_return_rate": np.round(unit_return, 4),
        "hour_of_day": hour,
        "day_of_week": ts.dayofweek,
        "coupon_attempted": rng.random(n) < (0.12 + 0.35 * hid["hidden_price_sensitivity"].values),
    })

    # ---- checkout start --------------------------------------------------
    p_start = np.clip(
        cfg.p_checkout_start
        + 0.18 * hid["hidden_impulsivity"].values
        - 0.10 * env.in_competitor_sale(ts),
        0.05, 0.95,
    )
    sessions["checkout_started"] = rng.random(n) < p_start

    ck = sessions[sessions["checkout_started"]].reset_index(drop=True)
    ck_hidden = hid[sessions["checkout_started"].values].reset_index(drop=True)
    m = len(ck)

    # ---- abandon vs attempt payment --------------------------------------
    fee_ratio = ck["shipping_fee_shown"] / np.maximum(ck["cart_value"], 1)
    p_abandon = np.clip(
        cfg.p_abandon_before_payment
        + 0.30 * ck_hidden["hidden_shipping_sensitivity"].values * np.clip(fee_ratio, 0, 0.3) * 3
        + 0.18 * ck_hidden["hidden_price_sensitivity"].values * (ck["cart_value"].values > 6000)
        - 0.22 * ck_hidden["hidden_brand_loyalty"].values
        + 0.12 * env.in_competitor_sale(ck["timestamp"]),
        0.05, 0.92,
    )
    abandoned = rng.random(m) < p_abandon

    checkouts = pd.DataFrame({
        "checkout_id": [f"CK{i + 1:07d}" for i in range(m)],
        "session_id": ck["session_id"].values,
        "customer_id": ck["customer_id"].values,
        "cart_value": ck["cart_value"].values,
        "cart_cogs": ck["cart_cogs"].values,
        "shipping_cost": ck["shipping_cost"].values,
        "shipping_fee_charged": ck["shipping_fee_shown"].values,
        "product_return_rate": ck["product_return_rate"].values,
        "checkout_started_at": ck["timestamp"].values,
        "abandoned": abandoned,
        "abandonment_stage": np.where(
            abandoned, rng.choice(ABANDON_STAGES, size=m, p=ABANDON_STAGE_P), None
        ),
        "payment_attempted": ~abandoned,
        "device_type": ck["device_type"].values,
        "traffic_source": ck["traffic_source"].values,
        "network_context": ck["network_context"].values,
        "hour_of_day": ck["hour_of_day"].values,
        "day_of_week": ck["day_of_week"].values,
        "coupon_attempted": ck["coupon_attempted"].values,
    })
    checkouts["base_contribution_margin"] = np.round(
        checkouts["cart_value"] - checkouts["cart_cogs"]
        + checkouts["shipping_fee_charged"] - checkouts["shipping_cost"], 2
    )
    checkouts["abandoned_at"] = np.where(
        checkouts["abandoned"],
        checkouts["checkout_started_at"] + pd.to_timedelta(rng.integers(2, 25, size=m), "m"),
        pd.NaT,
    )

    # ---- payment attempts -------------------------------------------------
    pay_mask = checkouts["payment_attempted"].values
    pc = checkouts[pay_mask].reset_index(drop=True)
    pc_hidden = ck_hidden[pay_mask].reset_index(drop=True)
    k = len(pc)

    outage = env.in_bank_outage(pc["checkout_started_at"])
    method = np.where(
        rng.random(k) < 0.75,
        customers.set_index("customer_id").loc[pc["customer_id"], "preferred_payment_method"].values,
        rng.choice(["UPI", "CARD", "NETBANKING", "WALLET"], size=k),
    )
    p_fail = np.clip(
        cfg.p_payment_failure
        + 0.30 * pc_hidden["hidden_payment_friction"].values
        + 0.28 * outage
        + 0.04 * (pc["network_context"].values == "MOBILE_3G"),
        0.01, 0.90,
    )
    failed = rng.random(k) < p_fail

    reason_p = np.tile(FAILURE_BASE_P, (k, 1))
    reason_p[outage] *= np.array([0.5, 4.0, 1.2, 0.8, 1.5, 0.8, 0.6, 1.0])
    reason_p /= reason_p.sum(axis=1, keepdims=True)
    reasons = np.array([
        rng.choice(FAILURE_REASONS, p=reason_p[i]) if failed[i] else "NONE"
        for i in range(k)
    ])

    payments = pd.DataFrame({
        "payment_attempt_id": [f"PA{i + 1:07d}" for i in range(k)],
        "checkout_id": pc["checkout_id"].values,
        "customer_id": pc["customer_id"].values,
        "timestamp": pc["checkout_started_at"].values + pd.to_timedelta(rng.integers(1, 12, size=k), "m"),
        "amount": pc["cart_value"].values + pc["shipping_fee_charged"].values,
        "payment_method": method,
        "status": np.where(failed, "FAILED", "CAPTURED"),
        "failure_reason": reasons,
        "retry_number": 0,
        "razorpay_order_id": None,
        "razorpay_payment_id": None,
    })

    return sessions, checkouts, payments


def _normalise(x):
    a = np.asarray(x, dtype=float)
    return a / a.sum()
