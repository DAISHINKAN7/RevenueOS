"""Train and calibrate the action-conditioned recovery model.

    python -m ml.models.train

Trains, in order:

    0. Global mean              (calibration floor)
    1. Segment lookup table     (does ML beat a smoothed lookup?)
    2. Logistic regression      (linear baseline)
    3. XGBoost                  (production candidate)

then compares three calibration treatments (none / Platt / isotonic) fitted on
VALIDATION only, selects one, and freezes everything.

TEST is not read anywhere in this module.

Selection objective (spec Section 24) is calibration-first: Brier and log loss
dominate, AUC is reported but does not drive the choice. Decisions are expected-
value comparisons, so probability *magnitude* matters more than ranking.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import IsotonicRegression
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, brier_score_loss, log_loss, roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from ml.features.build import ALL_FEATURES, CATEGORICAL, assert_no_leakage

PROC = Path("data/processed")
ART = Path("ml/artifacts")
RESULTS = Path("evaluation/results")
SEED = 42

NUMERIC = [f for f in ALL_FEATURES if f not in CATEGORICAL]


def expected_calibration_error(y, p, bins: int = 15) -> float:
    """Binned |accuracy - confidence|, weighted by bin mass."""
    idx = np.clip((np.asarray(p) * bins).astype(int), 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        m = idx == b
        if m.sum():
            ece += (m.sum() / len(p)) * abs(np.mean(y[m]) - np.mean(np.asarray(p)[m]))
    return float(ece)


def metrics(y, p) -> dict:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p)),
        "ece": expected_calibration_error(np.asarray(y), p),
    }


def make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20), CATEGORICAL),
    ])


# ------------------------------------------------------------- baseline 1
class SegmentBaseline:
    """Smoothed lookup table over (opportunity_type, failure_reason, action).

    The honest question this answers: does gradient boosting actually beat a
    well-smoothed contingency table? If not, the ML is decoration.
    """

    KEYS = ["opportunity_type", "failure_reason", "action"]

    def __init__(self, alpha: float = 20.0):
        self.alpha = alpha
        self.table: dict = {}
        self.global_rate = 0.5

    def fit(self, X: pd.DataFrame, y):
        self.global_rate = float(np.mean(y))
        d = X[self.KEYS].copy()
        d["y"] = np.asarray(y)
        g = d.groupby(self.KEYS)["y"].agg(["sum", "count"])
        self.table = {
            k: (r["sum"] + self.alpha * self.global_rate) / (r["count"] + self.alpha)
            for k, r in g.iterrows()
        }
        return self

    def predict_proba_1d(self, X: pd.DataFrame) -> np.ndarray:
        keys = list(map(tuple, X[self.KEYS].values))
        return np.array([self.table.get(k, self.global_rate) for k in keys])


def load(split: str) -> pd.DataFrame:
    return pd.read_parquet(PROC / f"{split}_features.parquet")


def main() -> None:
    ART.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    train, val = load("train"), load("validation")
    for d in (train, val):
        assert_no_leakage(d[ALL_FEATURES])

    Xtr, ytr = train[ALL_FEATURES], train["outcome"].values
    Xva, yva = val[ALL_FEATURES], val["outcome"].values

    prevalence = float(ytr.mean())
    print(f"recovery prevalence: train {prevalence:.3f} | val {yva.mean():.3f}")
    print(f"features {len(ALL_FEATURES)} | train {len(train):,} | val {len(val):,}\n")

    results: dict[str, dict] = {}

    # ---- baseline 0: global mean -----------------------------------------
    results["global_mean"] = metrics(yva, np.full(len(yva), prevalence))

    # ---- baseline 1: segment lookup --------------------------------------
    seg = SegmentBaseline().fit(Xtr, ytr)
    results["segment_lookup"] = metrics(yva, seg.predict_proba_1d(Xva))

    # ---- baseline 2: logistic regression ---------------------------------
    best_lr, best_lr_score = None, np.inf
    for C in [0.05, 0.2, 1.0, 5.0]:
        pipe = Pipeline([("prep", make_preprocessor()),
                         ("clf", LogisticRegression(C=C, max_iter=2000, random_state=SEED))])
        pipe.fit(Xtr, ytr)
        b = brier_score_loss(yva, pipe.predict_proba(Xva)[:, 1])
        if b < best_lr_score:
            best_lr, best_lr_score, best_C = pipe, b, C
    results["logistic_regression"] = metrics(yva, best_lr.predict_proba(Xva)[:, 1])
    print(f"logistic regression: best C={best_C}")

    # ---- baseline 3: XGBoost ---------------------------------------------
    # Small randomised search. Not AutoML: 12 candidates, selected on validation
    # Brier, which keeps the search honest about probability quality.
    space = {
        "n_estimators": [250, 400, 600],
        "max_depth": [3, 4, 5, 6],
        "learning_rate": [0.03, 0.05, 0.08],
        "subsample": [0.7, 0.85, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "min_child_weight": [1, 5, 20],
        "reg_lambda": [1.0, 3.0, 10.0],
        "reg_alpha": [0.0, 0.5],
    }
    best_xgb, best_params, best_b = None, None, np.inf
    trials = []
    for i in range(12):
        params = {k: rng.choice(v).item() for k, v in space.items()}
        pipe = Pipeline([
            ("prep", make_preprocessor()),
            ("clf", XGBClassifier(**params, random_state=SEED, n_jobs=4,
                                  eval_metric="logloss", tree_method="hist")),
        ])
        pipe.fit(Xtr, ytr)
        p = pipe.predict_proba(Xva)[:, 1]
        b = brier_score_loss(yva, p)
        trials.append({"trial": i, "brier": float(b), **params})
        if b < best_b:
            best_xgb, best_params, best_b = pipe, params, b
    print(f"xgboost: best validation Brier {best_b:.5f}")
    print(f"         params {best_params}\n")

    p_raw = best_xgb.predict_proba(Xva)[:, 1]
    results["xgboost_raw"] = metrics(yva, p_raw)

    # ---- calibration: fitted on VALIDATION only --------------------------
    # Protocol: the XGBoost model is fitted on TRAIN; the calibrator is fitted
    # on VALIDATION predictions. TEST is untouched. Because the calibrator sees
    # all of VALIDATION, validation calibration metrics are optimistic — the
    # honest read of calibration quality is the TEST number reported later.
    platt = LogisticRegression(max_iter=1000)
    platt.fit(p_raw.reshape(-1, 1), yva)
    p_platt = platt.predict_proba(p_raw.reshape(-1, 1))[:, 1]

    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(p_raw, yva)
    p_iso = iso.predict(p_raw)

    results["xgboost_platt"] = metrics(yva, p_platt)
    results["xgboost_isotonic"] = metrics(yva, p_iso)

    # ---- selection --------------------------------------------------------
    # Calibration-first: rank candidates by Brier, break ties on ECE. Isotonic
    # is not assumed best (Section 28); if raw is already calibrated it wins.
    cands = {k: results[k] for k in ("xgboost_raw", "xgboost_platt", "xgboost_isotonic")}
    chosen = min(cands, key=lambda k: (cands[k]["brier"], cands[k]["ece"]))
    calibration_method = {"xgboost_raw": "none", "xgboost_platt": "platt",
                          "xgboost_isotonic": "isotonic"}[chosen]
    calibrator = {"none": None, "platt": platt, "isotonic": iso}[calibration_method]

    print("validation metrics:")
    print(pd.DataFrame(results).T.round(5).to_string())
    print(f"\nselected: xgboost + calibration='{calibration_method}'")

    # ---- freeze -----------------------------------------------------------
    with open(ART / "recovery_model.pkl", "wb") as f:
        pickle.dump(best_xgb, f)
    with open(ART / "calibrator.pkl", "wb") as f:
        pickle.dump(calibrator, f)

    model_hash = hashlib.sha256((ART / "recovery_model.pkl").read_bytes()).hexdigest()[:16]
    meta = {
        "model_type": "XGBClassifier",
        "hyperparameters": best_params,
        "features": ALL_FEATURES,
        "categorical": CATEGORICAL,
        "calibration_method": calibration_method,
        "training_seed": SEED,
        "model_hash": model_hash,
        "prevalence_train": prevalence,
        "class_weighting": "none (prevalence not severely imbalanced)",
        "validation_metrics": results,
        "search_trials": trials,
        "frozen_at": datetime.now(timezone.utc).isoformat(),
    }
    (ART / "model_metadata.json").write_text(json.dumps(meta, indent=2, default=str))

    freeze = {
        "model_hash": model_hash,
        "calibration_method": calibration_method,
        "hyperparameters": best_params,
        "validation_metrics": results,
        "frozen_at": meta["frozen_at"],
        "note": "Frozen before any TEST or oracle access. Oracle evaluation "
                "occurs strictly after this manifest is written.",
    }
    (RESULTS / "model_freeze_manifest.json").write_text(json.dumps(freeze, indent=2, default=str))
    (RESULTS / "baseline_results.json").write_text(json.dumps(results, indent=2))
    print(f"\nfrozen -> {ART}/ and {RESULTS/'model_freeze_manifest.json'}")


if __name__ == "__main__":
    main()