"""Rule-policy thresholds fitted on VALIDATION only (spec Section 52).

Chosen as the validation median shipping fee among fee-charging abandonment
opportunities. Recorded as a constant so the rule policy is reproducible and so
it is visible that TEST played no part in setting it.
"""
from pathlib import Path

import pandas as pd


def _fit() -> float:
    p = Path("data/processed/validation_features.parquet")
    if not p.exists():
        return 40.0
    v = pd.read_parquet(p)
    m = (v["opportunity_type"] == "CHECKOUT_ABANDONMENT") & (v["shipping_fee_charged"] > 0)
    return float(v.loc[m, "shipping_fee_charged"].median()) if m.any() else 40.0


SHIPPING_FEE_THRESHOLD: float = _fit()