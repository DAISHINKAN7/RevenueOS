"""Synthetic product catalogue for NovaCart (consumer electronics / lifestyle).

Price points are constructed for an Indian consumer-electronics merchant. Only
the *shape* of the order-value distribution is calibrated against public retail
data (see `ml/simulation/calibration.py`); category mix and absolute price
points are designed, not borrowed. This is stated in the data card.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# (category, subcategory, price_low, price_high, margin_low, margin_high, weight_kg_low, weight_kg_high)
CATEGORIES = [
    ("Audio", "True Wireless Earbuds", 1_499, 9_999, 0.30, 0.46, 0.05, 0.35),
    ("Audio", "Over-Ear Headphones", 2_499, 14_999, 0.28, 0.44, 0.25, 0.60),
    ("Audio", "Bluetooth Speakers", 1_199, 12_999, 0.26, 0.42, 0.40, 2.20),
    ("Wearables", "Smartwatches", 1_999, 24_999, 0.22, 0.40, 0.08, 0.30),
    ("Wearables", "Fitness Bands", 999, 4_999, 0.30, 0.48, 0.04, 0.15),
    ("Mobile Accessories", "Power Banks", 799, 4_499, 0.24, 0.38, 0.20, 0.70),
    ("Mobile Accessories", "Chargers & Cables", 299, 2_499, 0.34, 0.55, 0.05, 0.40),
    ("Mobile Accessories", "Cases & Protection", 199, 1_799, 0.42, 0.62, 0.03, 0.20),
    ("Home", "Smart Lighting", 699, 5_999, 0.28, 0.45, 0.15, 1.20),
    ("Home", "Kitchen Appliances", 1_999, 18_999, 0.20, 0.34, 1.50, 8.00),
    ("Computing", "Keyboards & Mice", 899, 8_999, 0.26, 0.42, 0.30, 1.40),
    ("Computing", "Storage & Drives", 999, 11_999, 0.18, 0.32, 0.05, 0.45),
]

ADJECTIVES = ["Nova", "Pulse", "Aero", "Lumen", "Vertex", "Quanta", "Orbit", "Zenith"]
SUFFIXES = ["Pro", "Lite", "Max", "Air", "Core", "Plus", "X", "Neo", "SE", "Ultra"]

# Shipping is NOT constant (spec Section 24): it scales with weight and zone,
# so free shipping is sometimes excellent and sometimes economically absurd.
BASE_SHIPPING_FLOOR = 35.0
SHIPPING_PER_KG = 42.0


def generate_products(n_products: int, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for i in range(n_products):
        cat, sub, p_lo, p_hi, m_lo, m_hi, w_lo, w_hi = CATEGORIES[i % len(CATEGORIES)]

        # Log-uniform pricing keeps a realistic long right tail.
        price = float(np.exp(rng.uniform(np.log(p_lo), np.log(p_hi))))
        price = round(price / 10) * 10 - 1  # ...₹2,499-style price points
        margin_rate = float(rng.uniform(m_lo, m_hi))
        cogs = round(price * (1 - margin_rate), 2)
        weight = float(rng.uniform(w_lo, w_hi))

        rows.append({
            "product_id": f"P{i + 1:04d}",
            "name": f"{rng.choice(ADJECTIVES)} {sub.split()[0]} {rng.choice(SUFFIXES)}",
            "category": cat,
            "subcategory": sub,
            "selling_price": price,
            "cost_of_goods": cogs,
            "gross_margin": round(price - cogs, 2),
            "gross_margin_percent": round(100 * (price - cogs) / price, 2),
            "weight_kg": round(weight, 3),
            "base_shipping_cost": round(BASE_SHIPPING_FLOOR + SHIPPING_PER_KG * weight, 2),
            "inventory_level": int(rng.integers(5, 500)),
            "historical_return_rate": round(float(rng.beta(2, 40)), 4),
            "historical_conversion_rate": round(float(rng.beta(5, 45)), 4),
            "bundle_group": f"B{(i % 8) + 1}",
            "cross_sell_group": f"X{(i % 5) + 1}",
        })

    return pd.DataFrame(rows)


# Zone multipliers by city tier (spec Section 24).
ZONE_MULTIPLIER = {1: 1.00, 2: 1.25, 3: 1.60}


def shipping_cost_for(base_cost: float, city_tier: int, courier_disrupted: bool = False) -> float:
    """Actual fulfilment cost for one cart."""
    cost = base_cost * ZONE_MULTIPLIER[city_tier]
    if courier_disrupted:
        cost *= 1.35
    return round(cost, 2)
