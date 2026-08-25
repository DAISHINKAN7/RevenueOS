"""Production inference service around the frozen ML artifact.

Fails closed on every ambiguity. A missing feature, a version mismatch or a
non-finite probability disables autonomous financial action rather than
guessing — an incorrect probability here becomes an incorrect spend downstream.

This module must never touch the oracle dataset.
"""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from backend.app.domain import (
    FeatureSchemaMismatch, InvalidModelOutput, ModelVersionMismatch,
)

log = logging.getLogger("revenueos.predictor")
ART = Path("ml/artifacts")

EXPECTED_PIPELINE_VERSION = "1.0.0"
EXPECTED_MODEL_TYPE = "XGBClassifier"


@dataclass(frozen=True)
class RecoveryPrediction:
    action: str
    probability: float | None
    model_version: str
    feature_pipeline_version: str
    prediction_timestamp: datetime
    valid: bool
    error: str | None = None

    def as_dict(self) -> dict:
        return {
            "action": self.action,
            "probability": self.probability,
            "model_version": self.model_version,
            "feature_pipeline_version": self.feature_pipeline_version,
            "prediction_timestamp": self.prediction_timestamp.isoformat(),
            "valid": self.valid,
            "error": self.error,
        }


class RecoveryPredictor:
    """Loads and validates the frozen artifact. Returns probabilities only."""

    def __init__(self, artifacts_dir: Path = ART, strict: bool = True):
        self.dir = Path(artifacts_dir)
        self.loaded = False
        self.load_error: str | None = None
        self.model = None
        self.calibrator = None
        self.model_version = "unknown"
        self.feature_pipeline_version = "unknown"
        self.features: list[str] = []
        self.calibration_method = "unknown"
        try:
            self._load()
            self.loaded = True
        except Exception as exc:  # noqa: BLE001 - recorded, then surfaced via health
            self.load_error = f"{type(exc).__name__}: {exc}"
            log.error("model_load_failed", extra={"error": self.load_error})
            if strict:
                raise

    def _load(self) -> None:
        meta_path = self.dir / "model_metadata.json"
        schema_path = self.dir / "feature_schema.json"
        model_path = self.dir / "recovery_model.pkl"
        for p in (meta_path, schema_path, model_path):
            if not p.exists():
                raise ModelVersionMismatch(f"missing artifact {p}")

        meta = json.loads(meta_path.read_text())
        schema = json.loads(schema_path.read_text())

        if meta.get("model_type") != EXPECTED_MODEL_TYPE:
            raise ModelVersionMismatch(
                f"expected {EXPECTED_MODEL_TYPE}, artifact is {meta.get('model_type')}")
        if schema.get("feature_pipeline_version") != EXPECTED_PIPELINE_VERSION:
            raise FeatureSchemaMismatch(
                f"feature pipeline {schema.get('feature_pipeline_version')} "
                f"!= expected {EXPECTED_PIPELINE_VERSION}")
        if set(schema["features"]) != set(meta["features"]):
            raise FeatureSchemaMismatch("schema and metadata feature lists disagree")

        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
        calib_path = self.dir / "calibrator.pkl"
        if calib_path.exists():
            with open(calib_path, "rb") as f:
                self.calibrator = pickle.load(f)

        self.model_version = meta["model_hash"]
        self.calibration_method = meta.get("calibration_method", "none")
        self.feature_pipeline_version = schema["feature_pipeline_version"]
        self.features = list(schema["features"])

    # ------------------------------------------------------------------ api
    def health(self) -> dict:
        return {
            "model_loaded": self.loaded,
            "model_version": self.model_version,
            "feature_pipeline_version": self.feature_pipeline_version,
            "calibration_method": self.calibration_method,
            "load_error": self.load_error,
        }

    def _validate_row(self, row: pd.DataFrame) -> None:
        missing = [f for f in self.features if f not in row.columns]
        if missing:
            # Never impute a missing feature: silently defaulting a monetary
            # feature to zero would change the decision without any signal.
            raise FeatureSchemaMismatch(f"missing features: {missing[:6]}")

    def score_action(self, context_row: pd.Series | dict, action: str) -> RecoveryPrediction:
        now = datetime.now(timezone.utc)
        base = dict(context_row)
        try:
            if not self.loaded:
                raise ModelVersionMismatch(self.load_error or "model not loaded")

            from ml.features.build import add_action_features
            df = pd.DataFrame([base])
            df["action"] = action
            df = add_action_features(df)
            self._validate_row(df)

            p = self.model.predict_proba(df[self.features])[:, 1]
            if self.calibrator is not None:
                p = (self.calibrator.predict_proba(np.asarray(p).reshape(-1, 1))[:, 1]
                     if hasattr(self.calibrator, "predict_proba")
                     else self.calibrator.predict(p))
            val = float(np.asarray(p).ravel()[0])

            if not np.isfinite(val) or val < 0.0 or val > 1.0:
                raise InvalidModelOutput(f"probability out of range: {val}")

            return RecoveryPrediction(action, val, self.model_version,
                                      self.feature_pipeline_version, now, True)
        except Exception as exc:  # noqa: BLE001
            return RecoveryPrediction(
                action, None, self.model_version, self.feature_pipeline_version,
                now, False, f"{type(exc).__name__}: {exc}")

    def score_candidate_actions(self, context_row, actions) -> list[RecoveryPrediction]:
        return [self.score_action(context_row, a) for a in actions]


_predictor: RecoveryPredictor | None = None


def get_predictor() -> RecoveryPredictor:
    global _predictor
    if _predictor is None:
        _predictor = RecoveryPredictor(strict=False)
    return _predictor