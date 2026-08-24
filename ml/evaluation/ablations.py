"""Feature-group ablations (spec Sections 79-83).

Four ablations, no more. Each retrains XGBoost on TRAIN with one feature group
removed, then reports held-out probability quality and oracle dP error.

Note on the economic-context ablation: economics still exist in the financial
engine regardless. This ablation only asks whether the *response model* needs
margin/shipping features to predict recovery.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, average_precision_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from ml.actions import Action
from ml.features.build import ALL_FEATURES, CATEGORICAL, FEATURE_GROUPS, add_action_features
from ml.models.train import make_preprocessor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROC = Path("data/processed")
SEED = 42

ABLATIONS = {
    "full": [],
    "no_action_history": FEATURE_GROUPS["action_history"],
    "no_customer_history": FEATURE_GROUPS["customer_history"],
    "no_economic_context": ["base_contribution_margin", "base_margin_pct",
                            "shipping_cost", "shipping_cost_to_margin_ratio",
                            "action_cost_to_margin_ratio"],
    "no_time_context": ["minutes_since_event", "log_minutes_since_event"],
}


def _prep(numeric, categorical):
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20), categorical),
    ])


def run_ablations(test: pd.DataFrame, true_p: pd.DataFrame,
                  elig: pd.DataFrame) -> pd.DataFrame:
    train = pd.read_parquet(PROC / "train_features.parquet")
    rows = []

    for name, drop in ABLATIONS.items():
        feats = [f for f in ALL_FEATURES if f not in set(drop)]
        num = [f for f in feats if f not in CATEGORICAL]
        cat = [f for f in feats if f in CATEGORICAL]

        pipe = Pipeline([
            ("prep", _prep(num, cat)),
            ("clf", XGBClassifier(n_estimators=250, max_depth=3, learning_rate=0.05,
                                  min_child_weight=20, reg_lambda=10.0,
                                  random_state=SEED, n_jobs=4,
                                  eval_metric="logloss", tree_method="hist")),
        ])
        pipe.fit(train[feats], train["outcome"].values)

        p = pipe.predict_proba(test[feats])[:, 1]
        y = test["outcome"].values

        # oracle dP MAE across the main actions
        maes = []
        base_rows = test.copy()
        base_rows["action"] = Action.DO_NOTHING.value
        p0 = pipe.predict_proba(add_action_features(base_rows)[feats])[:, 1]
        for a in [Action.FREE_SHIPPING, Action.SMALL_DISCOUNT,
                  Action.MEDIUM_DISCOUNT, Action.DELAYED_RETRY]:
            m = elig[a.value].values & np.isfinite(true_p[a.value].values)
            if m.sum() < 30:
                continue
            r = test.loc[m].copy()
            r["action"] = a.value
            pa = pipe.predict_proba(add_action_features(r)[feats])[:, 1]
            pred_d = pa - p0[m]
            true_d = true_p[a.value].values[m] - true_p[Action.DO_NOTHING.value].values[m]
            maes.append(np.abs(pred_d - true_d).mean())

        rows.append({
            "ablation": name,
            "n_features": len(feats),
            "brier": float(brier_score_loss(y, p)),
            "log_loss": float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))),
            "pr_auc": float(average_precision_score(y, p)),
            "oracle_dP_mae": float(np.mean(maes)) if maes else np.nan,
        })

    df = pd.DataFrame(rows)
    base = df[df["ablation"] == "full"].iloc[0]
    df["brier_delta_vs_full"] = df["brier"] - base["brier"]
    df["dP_mae_delta_vs_full"] = df["oracle_dP_mae"] - base["oracle_dP_mae"]
    return df