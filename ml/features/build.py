"""Feature pipeline for action-conditioned recovery prediction.

    python -m ml.features.build

Builds one row per (opportunity, action) pair. For TRAINING, only the
historically observed action is emitted — expanding across all actions using
oracle labels would be training on counterfactual ground truth, which is
forbidden (spec Section 19).

At decision time the same `build_candidate_rows` function emits one row per
*eligible* action so the model can score each candidate.

This module must never import anything under `ml/evaluation/oracle_eval.py`
or read `oracle.parquet`. Enforced by `tests/test_phase3.py`.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ml.actions import Action, eligible_actions, spec_for
from ml.config import DEFAULT_CONFIG
from ml.financial_engine import OpportunityEconomics, incentive_cost, fixed_action_cost

FEATURE_PIPELINE_VERSION = "1.0.0"

DATA = Path("data/generated")
OUT = Path("data/processed")
RESULTS = Path("evaluation/results")

# Columns that must never reach the model (spec Section 16).
FORBIDDEN_PATTERNS = (
    "hidden_", "oracle_", "future_", "p_recovery__",
    "true_recovery_probability", "true_uplift", "best_action",
    "counterfactual", "realized_outcome", "converted_after_intervention",
    "recovered_revenue", "recovered_contribution",
)

# Empirical-Bayes prior strength for sparse historical rates (Section 8).
# Chosen a priori (not tuned on any split): with alpha=5 a customer needs ~5
# observations before their own history outweighs the population rate.
SMOOTHING_ALPHA = 5.0


class LeakageError(AssertionError):
    """Raised when a forbidden column reaches the feature matrix."""


FEATURE_GROUPS: dict[str, list[str]] = {
    "customer_history": [
        "orders_lifetime", "orders_last_30d", "orders_last_90d",
        "average_order_value", "lifetime_value", "days_since_last_purchase",
        "previous_checkout_abandonments", "previous_payment_failures",
        "historical_return_rate", "historical_cancellation_rate",
        "tenure_days", "city_tier",
    ],
    "action_history": [
        "coupon_offers_seen", "coupon_offers_redeemed", "coupon_rate_smoothed",
        "free_shipping_offers_seen", "free_shipping_offers_redeemed",
        "free_shipping_rate_smoothed", "retry_offers_seen", "retry_offers_succeeded",
        "retry_rate_smoothed", "coupon_history_missing", "shipping_history_missing",
        "retry_history_missing",
    ],
    "checkout_context": [
        "cart_value", "base_contribution_margin", "base_margin_pct",
        "shipping_fee_charged", "shipping_cost", "shipping_fee_to_cart_ratio",
        "shipping_cost_to_margin_ratio", "product_return_rate", "coupon_attempted",
        "abandonment_stage", "device_type", "traffic_source", "network_context",
    ],
    "payment_context": [
        "payment_method", "failure_reason", "attempt_number", "opportunity_type",
    ],
    "temporal_context": [
        "minutes_since_event", "log_minutes_since_event",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend",
    ],
    "action_features": [
        "action", "action_discount_percent", "action_discount_amount",
        "action_waives_shipping", "action_fixed_cost",
        "action_incentive_cost", "action_cost_to_margin_ratio",
        "action_cost_to_cart_ratio", "action_is_retry", "action_is_contact",
    ],
}

CATEGORICAL = [
    "abandonment_stage", "device_type", "traffic_source", "network_context",
    "payment_method", "failure_reason", "opportunity_type", "action",
]

ALL_FEATURES = [f for g in FEATURE_GROUPS.values() for f in g]

RETRY_ACTIONS = {Action.IMMEDIATE_RETRY, Action.DELAYED_RETRY}
CONTACT_ACTIONS = {
    Action.FREE_SHIPPING, Action.SMALL_DISCOUNT, Action.MEDIUM_DISCOUNT,
    Action.PAYMENT_METHOD_SWITCH, Action.PAYMENT_LINK, Action.HUMAN_ESCALATION,
}


# --------------------------------------------------------------- eligibility
def context_eligible_actions(row) -> list[Action]:
    """Deterministic eligibility (spec Section 15).

    Narrower than the structural `eligible_actions`: absurd candidates are
    excluded here rather than relying on the model to reject them.
    """
    acts = list(eligible_actions(row["opportunity_type"]))
    out = []
    for a in acts:
        if a is Action.FREE_SHIPPING and float(row["shipping_fee_charged"]) <= 0:
            continue  # nothing to waive
        if a is Action.PAYMENT_METHOD_SWITCH and row["opportunity_type"] != "PAYMENT_FAILURE":
            # No instrument-level friction evidence without a failed attempt.
            continue
        out.append(a)
    return out


# ------------------------------------------------------------------ features
def _smooth(successes, observations, global_rate, alpha=SMOOTHING_ALPHA):
    """Empirical-Bayes smoothed rate (Section 8)."""
    s = np.nan_to_num(np.asarray(successes, dtype=float))
    n = np.nan_to_num(np.asarray(observations, dtype=float))
    return (s + alpha * global_rate) / (n + alpha)


def build_base_features(opp: pd.DataFrame, customers: pd.DataFrame) -> pd.DataFrame:
    """Opportunity-level features, independent of the candidate action."""
    df = opp.merge(customers, on="customer_id", how="left", suffixes=("", "_cust"))

    # --- smoothed historical response rates with population priors ---
    for name, seen, red in [
        ("coupon", "coupon_offers_seen", "coupon_offers_redeemed"),
        ("free_shipping", "free_shipping_offers_seen", "free_shipping_offers_redeemed"),
        ("retry", "retry_offers_seen", "retry_offers_succeeded"),
    ]:
        g = df[red].sum() / max(df[seen].sum(), 1)
        df[f"{name}_rate_smoothed"] = _smooth(df[red], df[seen], g)
        # Cold start is stated explicitly, never imputed as certainty (Section 9).
        df[f"{'coupon' if name == 'coupon' else 'shipping' if name == 'free_shipping' else 'retry'}_history_missing"] = (
            df[seen].fillna(0) == 0
        ).astype(int)

    df["base_margin_pct"] = 100 * df["base_contribution_margin"] / df["cart_value"].clip(lower=1)
    df["shipping_fee_to_cart_ratio"] = df["shipping_fee_charged"] / df["cart_value"].clip(lower=1)
    df["shipping_cost_to_margin_ratio"] = (
        df["shipping_cost"] / df["base_contribution_margin"].abs().clip(lower=1)
    )

    df["log_minutes_since_event"] = np.log1p(df["minutes_since_event"].astype(float))
    h = df["hour_of_day"].astype(float)
    d = df["day_of_week"].astype(float)
    df["hour_sin"] = np.sin(2 * np.pi * h / 24)
    df["hour_cos"] = np.cos(2 * np.pi * h / 24)
    df["dow_sin"] = np.sin(2 * np.pi * d / 7)
    df["dow_cos"] = np.cos(2 * np.pi * d / 7)
    df["is_weekend"] = (d >= 5).astype(int)

    df["coupon_attempted"] = df["coupon_attempted"].fillna(False).astype(int)
    df["abandonment_stage"] = df["abandonment_stage"].fillna("NO_ABANDONMENT_STAGE")
    df["failure_reason"] = df["failure_reason"].fillna("NO_PAYMENT_FAILURE")
    df["failure_reason"] = df["failure_reason"].replace({"NONE": "NO_PAYMENT_FAILURE"})
    df["payment_method"] = df["payment_method"].replace({"NONE": "NO_PAYMENT_METHOD"})
    return df


def add_action_features(df: pd.DataFrame, action_col: str = "action") -> pd.DataFrame:
    """Per-candidate-action economics. Recomputed for every scored action."""
    df = df.copy()
    specs = {a.value: spec_for(a) for a in Action}

    df["action_discount_percent"] = df[action_col].map(lambda a: specs[a].discount_percent)
    df["action_waives_shipping"] = df[action_col].map(lambda a: int(specs[a].waives_shipping_fee))
    df["action_fixed_cost"] = df[action_col].map(lambda a: specs[a].fixed_cost)
    df["action_discount_amount"] = df["cart_value"] * df["action_discount_percent"] / 100.0
    df["action_incentive_cost"] = (
        df["action_discount_amount"] + df["action_waives_shipping"] * df["shipping_fee_charged"]
    )
    df["action_cost_to_margin_ratio"] = (
        df["action_incentive_cost"] / df["base_contribution_margin"].abs().clip(lower=1)
    )
    df["action_cost_to_cart_ratio"] = df["action_incentive_cost"] / df["cart_value"].clip(lower=1)
    df["action_is_retry"] = df[action_col].map(lambda a: int(Action(a) in RETRY_ACTIONS))
    df["action_is_contact"] = df[action_col].map(lambda a: int(Action(a) in CONTACT_ACTIONS))
    return df


def build_candidate_rows(base_row: pd.Series) -> pd.DataFrame:
    """One feature row per eligible action, for decision-time scoring."""
    acts = context_eligible_actions(base_row)
    rows = pd.DataFrame([base_row] * len(acts)).reset_index(drop=True)
    rows["action"] = [a.value for a in acts]
    return add_action_features(rows)


def assert_no_leakage(frame: pd.DataFrame) -> None:
    """Reject any forbidden column reaching the model (Section 16)."""
    bad = [c for c in frame.columns
           if any(p in c.lower() for p in FORBIDDEN_PATTERNS)]
    if bad:
        raise LeakageError(f"forbidden columns in feature matrix: {bad}")


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    opp = pd.read_parquet(DATA / "opportunities.parquet")
    iv = pd.read_parquet(DATA / "interventions.parquet")
    customers = pd.read_parquet(DATA / "customers.parquet")
    manifest = json.loads((DATA / "manifest.json").read_text())

    base = build_base_features(opp, customers)

    # Attach ONLY the logged action + logged outcome (Section 19).
    logged = iv[["opportunity_id", "action_taken", "action_propensity",
                 "is_exploration", "converted_after_intervention",
                 "fixed_action_cost", "incentive_cost_if_recovered",
                 "recovered_contribution"]]
    df = base.merge(logged, on="opportunity_id", how="inner")
    df["action"] = df["action_taken"]
    df = add_action_features(df)
    df["outcome"] = df["converted_after_intervention"].astype(int)

    keep = (["opportunity_id", "detected_at", "split", "customer_id",
             "customer_segment", "action_propensity", "is_exploration",
             "outcome", "cart_cogs"] + ALL_FEATURES)
    keep = [c for c in dict.fromkeys(keep) if c in df.columns]
    out = df[keep].sort_values("detected_at").reset_index(drop=True)

    # Feature matrix must be clean; identifiers/labels are carried separately.
    assert_no_leakage(out[ALL_FEATURES])

    paths = {}
    for split, name in [("TRAIN", "train"), ("VALIDATION", "validation"), ("TEST", "test")]:
        sub = out[out["split"] == split].reset_index(drop=True)
        p = OUT / f"{name}_features.parquet"
        sub.to_parquet(p, index=False)
        paths[name] = (p, len(sub))
        print(f"{name:<11} {len(sub):>7,} rows -> {p}")

    schema = {
        "feature_pipeline_version": FEATURE_PIPELINE_VERSION,
        "features": ALL_FEATURES,
        "categorical": CATEGORICAL,
        "feature_groups": FEATURE_GROUPS,
        "smoothing_alpha": SMOOTHING_ALPHA,
    }
    Path("ml/artifacts").mkdir(parents=True, exist_ok=True)
    Path("ml/artifacts/feature_schema.json").write_text(json.dumps(schema, indent=2))

    def bounds(name):
        sub = out[out["split"] == name]
        return str(sub["detected_at"].min()), str(sub["detected_at"].max())

    tr, va, te = bounds("TRAIN"), bounds("VALIDATION"), bounds("TEST")
    provenance = {
        "simulator_version": manifest["simulator_version"],
        "logging_policy_version": manifest["logging_policy_version"],
        "seed": manifest["seed"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "train_start": tr[0], "train_end": tr[1],
        "validation_start": va[0], "validation_end": va[1],
        "test_start": te[0], "test_end": te[1],
        "train_rows": paths["train"][1],
        "validation_rows": paths["validation"][1],
        "test_rows": paths["test"][1],
        "feature_pipeline_version": FEATURE_PIPELINE_VERSION,
        "oracle_file": str(DATA / "oracle.parquet"),
        "oracle_access_policy": "evaluation_only",
        "hashes": {
            "train": _hash(paths["train"][0]),
            "validation": _hash(paths["validation"][0]),
            "test": _hash(paths["test"][0]),
            "oracle": _hash(DATA / "oracle.parquet"),
            "feature_schema": _hash(Path("ml/artifacts/feature_schema.json")),
        },
    }
    (RESULTS / "data_provenance.json").write_text(json.dumps(provenance, indent=2))
    print(f"\nFeatures: {len(ALL_FEATURES)} | wrote {RESULTS/'data_provenance.json'}")


if __name__ == "__main__":
    main()