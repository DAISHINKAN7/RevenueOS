"""Inference, financial and policy contracts (spec Sections 133-135).

Three deliberately separate layers:

    score_candidate_actions()  -> probabilities only, no money
    evaluate_action_value()    -> money only, no model
    select_recovery_action()   -> chooses, using both

Keeping them apart is the point: the model never decides, and the financial
engine never predicts. Every policy in this module consumes the same two
primitives, so comparisons between policies are apples-to-apples.

This module must not read the oracle.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ml.actions import Action
from ml.features.build import ALL_FEATURES, add_action_features
from ml.financial_engine import OpportunityEconomics, valuate_action

ART = Path("ml/artifacts")


@dataclass(frozen=True)
class ActionScore:
    action: str
    probability: float
    model_version: str
    eligible: bool


class RecoveryPredictor:
    """Loads the frozen model + calibrator. Returns probabilities, nothing else."""

    def __init__(self, model_path=ART / "recovery_model.pkl",
                 calib_path=ART / "calibrator.pkl"):
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
        with open(calib_path, "rb") as f:
            self.calibrator = pickle.load(f)
        self.version = "1.0.0"

    def _calibrate(self, p: np.ndarray) -> np.ndarray:
        if self.calibrator is None:
            return p
        if hasattr(self.calibrator, "predict_proba"):
            return self.calibrator.predict_proba(p.reshape(-1, 1))[:, 1]
        return self.calibrator.predict(p)

    def predict(self, X: pd.DataFrame, calibrated: bool = True) -> np.ndarray:
        p = self.model.predict_proba(X[ALL_FEATURES])[:, 1]
        if calibrated:
            p = self._calibrate(p)
        p = np.asarray(p, dtype=float)
        # Consistency checks (Section 33): never let a bad number reach money.
        if not np.all(np.isfinite(p)):
            bad = ~np.isfinite(p)
            p[bad] = 0.0
        return np.clip(p, 0.0, 1.0)


def eligibility_mask(base: pd.DataFrame, action: Action) -> np.ndarray:
    """Deterministic eligibility, vectorised (Section 15)."""
    is_failure = (base["opportunity_type"] == "PAYMENT_FAILURE").values
    if action in (Action.IMMEDIATE_RETRY, Action.DELAYED_RETRY,
                  Action.PAYMENT_METHOD_SWITCH):
        return is_failure
    if action is Action.FREE_SHIPPING:
        return (base["shipping_fee_charged"] > 0).values
    return np.ones(len(base), dtype=bool)


def score_all_actions(base: pd.DataFrame, predictor: RecoveryPredictor,
                      calibrated: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score every action for every opportunity.

    Returns (probabilities, eligibility) frames indexed like `base`, with one
    column per action. Ineligible cells are NaN.
    """
    probs, elig = {}, {}
    for a in Action:
        m = eligibility_mask(base, a)
        col = np.full(len(base), np.nan)
        if m.any():
            rows = base.loc[m].copy()
            rows["action"] = a.value
            rows = add_action_features(rows)
            col[m] = predictor.predict(rows, calibrated=calibrated)
        probs[a.value] = col
        elig[a.value] = m
    return pd.DataFrame(probs, index=base.index), pd.DataFrame(elig, index=base.index)


def econ_for_row(row) -> OpportunityEconomics:
    return OpportunityEconomics(
        cart_value=float(row["cart_value"]),
        cogs=float(row["cart_cogs"]),
        shipping_cost=float(row["shipping_cost"]),
        shipping_fee_charged=float(row["shipping_fee_charged"]),
    )


def evaluate_all_actions(base: pd.DataFrame, probs: pd.DataFrame,
                         elig: pd.DataFrame) -> pd.DataFrame:
    """dEV for every (opportunity, action). Pure arithmetic, no model."""
    out = {}
    econs = [econ_for_row(r) for _, r in base.iterrows()]
    p_nothing = probs[Action.DO_NOTHING.value].values
    ev_nothing = np.array([
        valuate_action(e, Action.DO_NOTHING, p).expected_value
        for e, p in zip(econs, p_nothing)
    ])
    out[Action.DO_NOTHING.value] = np.zeros(len(base))

    for a in Action:
        if a is Action.DO_NOTHING:
            continue
        col = np.full(len(base), np.nan)
        m = elig[a.value].values & np.isfinite(probs[a.value].values)
        idx = np.where(m)[0]
        for i in idx:
            v = valuate_action(econs[i], a, float(probs[a.value].values[i]))
            col[i] = v.expected_value - ev_nothing[i]
        out[a.value] = col

    dev = pd.DataFrame(out, index=base.index)
    assert np.all(np.isfinite(dev.fillna(0.0).values)), "non-finite dEV produced"
    return dev


# ------------------------------------------------------------------ policies
def policy_do_nothing(base, probs, dev) -> pd.Series:
    return pd.Series([Action.DO_NOTHING.value] * len(base), index=base.index)


def policy_flat_10(base, probs, dev) -> pd.Series:
    return pd.Series([Action.MEDIUM_DISCOUNT.value] * len(base), index=base.index)


def policy_rules(base, probs, dev) -> pd.Series:
    """Transparent rule baseline (Section 52). Thresholds set on VALIDATION."""
    from ml.evaluation.rule_thresholds import SHIPPING_FEE_THRESHOLD
    out = []
    for _, r in base.iterrows():
        reason = r["failure_reason"]
        if r["opportunity_type"] == "PAYMENT_FAILURE":
            if reason in ("BANK_TIMEOUT", "INSUFFICIENT_FUNDS"):
                out.append(Action.DELAYED_RETRY.value)
            elif reason in ("UPI_TIMEOUT", "CARD_DECLINED", "AUTHENTICATION_FAILURE"):
                out.append(Action.PAYMENT_METHOD_SWITCH.value)
            elif reason == "NETWORK_ERROR":
                out.append(Action.IMMEDIATE_RETRY.value)
            else:
                out.append(Action.DO_NOTHING.value)
        elif r["shipping_fee_charged"] > SHIPPING_FEE_THRESHOLD:
            out.append(Action.FREE_SHIPPING.value)
        elif r["abandonment_stage"] in ("CART", "PAYMENT_SELECTION"):
            out.append(Action.SMALL_DISCOUNT.value)
        else:
            out.append(Action.DO_NOTHING.value)
    return pd.Series(out, index=base.index)


def policy_model_conversion_max(base, probs, dev) -> pd.Series:
    """Naive comparator: pick the highest predicted recovery probability."""
    return probs.idxmax(axis=1)


def policy_revenueos(base, probs, dev, min_delta_ev: float = 0.0,
                     min_margin: float = 0.0) -> pd.Series:
    """argmax dEV subject to dEV > threshold, else DO_NOTHING (Section 44)."""
    d = dev.drop(columns=[Action.DO_NOTHING.value])
    best = d.idxmax(axis=1)
    best_val = d.max(axis=1)
    if min_margin > 0:
        second = d.apply(lambda r: r.nlargest(2).iloc[-1] if r.notna().sum() > 1 else -np.inf, axis=1)
        margin_ok = (best_val - second) >= min_margin
    else:
        margin_ok = pd.Series(True, index=base.index)
    take = (best_val > min_delta_ev) & margin_ok & best_val.notna()
    return best.where(take, Action.DO_NOTHING.value)


POLICIES = {
    "DO_NOTHING": policy_do_nothing,
    "FLAT_10_PERCENT": policy_flat_10,
    "RULES": policy_rules,
    "MODEL_CONVERSION_MAX": policy_model_conversion_max,
    "REVENUEOS": policy_revenueos,
}