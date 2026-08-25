"""Train and calibrate the action-conditioned recovery model.

    python -m ml.models.train

Trains, in order:

    0. Global mean              (calibration floor)
    1. Segment lookup table     (does ML beat a smoothed lookup?)
    2. Logistic regression      (linear baseline)
    3. XGBoost                  (production candidate)

then compares three calibration treatments (none / Platt / isotonic) and freezes
everything.

Calibration protocol (corrected — pipeline v1.1.0)
--------------------------------------------------
The first model gate fitted the calibrators on VALIDATION and *also* selected
among them on VALIDATION. Isotonic therefore scored ECE = 0.0000, which is not a
measurement of calibration quality — it is isotonic regression reproducing its
own fitting partition. The selection was optimistic and structurally favoured
isotonic.

VALIDATION is now split chronologically:

    first 50% of the validation timeline  -> CALIBRATION     (fits calibrators)
    second 50%                            -> MODEL_SELECTION (chooses between them)

No candidate is ever scored on rows that fitted it. Chronological order is
preserved throughout, so the ordering TRAIN < CALIBRATION < MODEL_SELECTION <
TEST holds.

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

PIPELINE_VERSION = "1.1.0"

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

    # --- chronological split of VALIDATION (corrected protocol) -----------
    val = val.sort_values("detected_at").reset_index(drop=True)
    cut = len(val) // 2
    calib, msel = val.iloc[:cut].copy(), val.iloc[cut:].copy()
    assert calib["detected_at"].max() <= msel["detected_at"].min(), "validation split not chronological"

    Xtr, ytr = train[ALL_FEATURES], train["outcome"].values
    Xca, yca = calib[ALL_FEATURES], calib["outcome"].values
    Xms, yms = msel[ALL_FEATURES], msel["outcome"].values

    prevalence = float(ytr.mean())
    print(f"recovery prevalence: train {prevalence:.3f} | calib {yca.mean():.3f} | msel {yms.mean():.3f}")
    print(f"features {len(ALL_FEATURES)} | train {len(train):,} | "
          f"calibration {len(calib):,} | model_selection {len(msel):,}\n")

    results: dict[str, dict] = {}

    # ---- baseline 0: global mean -----------------------------------------
    results["global_mean"] = metrics(yms, np.full(len(yms), prevalence))

    # ---- baseline 1: segment lookup --------------------------------------
    seg = SegmentBaseline().fit(Xtr, ytr)
    results["segment_lookup"] = metrics(yms, seg.predict_proba_1d(Xms))
    with open(ART / "segment_baseline.pkl", "wb") as f:
        pickle.dump(seg, f)

    # ---- baseline 2: logistic regression ---------------------------------
    # Hyperparameter chosen on MODEL_SELECTION, which no calibrator touches.
    best_lr, best_lr_score, best_C = None, np.inf, None
    for C in [0.05, 0.2, 1.0, 5.0]:
        pipe = Pipeline([("prep", make_preprocessor()),
                         ("clf", LogisticRegression(C=C, max_iter=2000, random_state=SEED))])
        pipe.fit(Xtr, ytr)
        b = brier_score_loss(yms, pipe.predict_proba(Xms)[:, 1])
        if b < best_lr_score:
            best_lr, best_lr_score, best_C = pipe, b, C
    results["logistic_regression"] = metrics(yms, best_lr.predict_proba(Xms)[:, 1])
    with open(ART / "logistic_model.pkl", "wb") as f:
        pickle.dump(best_lr, f)
    print(f"logistic regression: best C={best_C}")

    # ---- baseline 3: XGBoost ---------------------------------------------
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
        b = brier_score_loss(yms, pipe.predict_proba(Xms)[:, 1])
        trials.append({"trial": i, "brier": float(b), **params})
        if b < best_b:
            best_xgb, best_params, best_b = pipe, params, b
    print(f"xgboost: best MODEL_SELECTION Brier {best_b:.5f}")
    print(f"         params {best_params}\n")

    # ---- calibration: FIT on CALIBRATION, SELECT on MODEL_SELECTION ------
    p_ca = best_xgb.predict_proba(Xca)[:, 1]
    p_ms_raw = best_xgb.predict_proba(Xms)[:, 1]

    platt = LogisticRegression(max_iter=1000)
    platt.fit(p_ca.reshape(-1, 1), yca)

    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(p_ca, yca)

    # Every candidate is scored on MODEL_SELECTION, which fitted none of them.
    results["xgboost_raw"] = metrics(yms, p_ms_raw)
    results["xgboost_platt"] = metrics(yms, platt.predict_proba(p_ms_raw.reshape(-1, 1))[:, 1])
    results["xgboost_isotonic"] = metrics(yms, iso.predict(p_ms_raw))

    # For the record: what the OLD (incorrect) same-set procedure would report.
    optimistic = {
        "xgboost_isotonic_same_set_ece": expected_calibration_error(
            yca, IsotonicRegression(out_of_bounds="clip").fit(p_ca, yca).predict(p_ca)),
    }

    # ---- selection --------------------------------------------------------
    cands = {k: results[k] for k in ("xgboost_raw", "xgboost_platt", "xgboost_isotonic")}
    chosen = min(cands, key=lambda k: (cands[k]["brier"], cands[k]["ece"]))
    calibration_method = {"xgboost_raw": "none", "xgboost_platt": "platt",
                          "xgboost_isotonic": "isotonic"}[chosen]
    calibrator = {"none": None, "platt": platt, "isotonic": iso}[calibration_method]

    print("MODEL_SELECTION metrics (no candidate scored on its fitting rows):")
    print(pd.DataFrame(results).T.round(5).to_string())
    print(f"\nsame-set isotonic ECE under the OLD incorrect protocol: "
          f"{optimistic['xgboost_isotonic_same_set_ece']:.5f} (reported for transparency)")
    print(f"\nselected: xgboost + calibration='{calibration_method}'")

    # ---- freeze -----------------------------------------------------------
    with open(ART / "recovery_model.pkl", "wb") as f:
        pickle.dump(best_xgb, f)
    with open(ART / "calibrator.pkl", "wb") as f:
        pickle.dump(calibrator, f)

    model_hash = hashlib.sha256((ART / "recovery_model.pkl").read_bytes()).hexdigest()[:16]
    periods = {
        "base_model_training_period": [str(train["detected_at"].min()), str(train["detected_at"].max())],
        "calibration_period": [str(calib["detected_at"].min()), str(calib["detected_at"].max())],
        "model_selection_period": [str(msel["detected_at"].min()), str(msel["detected_at"].max())],
        "calibration_rows": len(calib),
        "model_selection_rows": len(msel),
    }

    meta = {
        "model_type": "XGBClassifier",
        "pipeline_version": PIPELINE_VERSION,
        **periods,
        "hyperparameters": best_params,
        "features": ALL_FEATURES,
        "categorical": CATEGORICAL,
        "calibration_method": calibration_method,
        "training_seed": SEED,
        "model_hash": model_hash,
        "prevalence_train": prevalence,
        "class_weighting": "none (prevalence not severely imbalanced)",
        "model_selection_metrics": results,
        "optimistic_same_set_reference": optimistic,
        "search_trials": trials,
        "frozen_at": datetime.now(timezone.utc).isoformat(),
    }
    (ART / "model_metadata.json").write_text(json.dumps(meta, indent=2, default=str))

    freeze = {
        "pipeline_version": PIPELINE_VERSION,
        "model_hash": model_hash,
        "calibration_method": calibration_method,
        "hyperparameters": best_params,
        **periods,
        "selection_metrics": results,
        "optimistic_same_set_reference": optimistic,
        "frozen_at": meta["frozen_at"],
        "calibration_protocol": (
            "TRAIN fits base models; CALIBRATION (first 50% of the validation "
            "timeline) fits Platt/isotonic; MODEL_SELECTION (second 50%) chooses "
            "among raw/Platt/isotonic. No candidate is scored on rows that fitted "
            "it. TEST is reporting-only."
        ),
        "protocol_correction_note": (
            "The initial model gate exposed an optimistic calibration-selection "
            "procedure because isotonic calibration was evaluated on its fitting "
            "partition. The pipeline was corrected by separating calibration "
            "fitting from model selection before finalizing the model."
        ),
        "note": "Frozen before any TEST or oracle access.",
    }
    (RESULTS / "model_freeze_manifest.json").write_text(json.dumps(freeze, indent=2, default=str))
    (RESULTS / "baseline_results.json").write_text(json.dumps(results, indent=2))
    print(f"\nfrozen -> {ART}/ and {RESULTS/'model_freeze_manifest.json'}")


if __name__ == "__main__":
    main()