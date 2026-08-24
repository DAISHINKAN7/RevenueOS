"""Synthetic-oracle evaluation (spec Sections 35-50).

THIS IS THE ONLY MODULE PERMITTED TO READ `data/generated/oracle.parquet`.

Training and feature code must never import it; `tests/test_phase3.py` asserts
that no module under `ml/features/`, `ml/models/` or `ml/calibration/`
references the oracle path.

Everything computed here is labelled **Synthetic Oracle Evaluation** and is an
upper-bound reference, not a deployable policy and not real-world causal lift.
It runs strictly after `model_freeze_manifest.json` exists.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from ml.actions import Action
from ml.financial_engine import valuate_action
from ml.evaluation.scoring import econ_for_row

ORACLE_PATH = Path("data/generated/oracle.parquet")
RESULTS = Path("evaluation/results")


def _require_freeze() -> dict:
    p = RESULTS / "model_freeze_manifest.json"
    if not p.exists():
        raise RuntimeError(
            "Model freeze manifest missing. The oracle must not be read before "
            "the model is frozen — run `python -m ml.models.train` first."
        )
    return json.loads(p.read_text())


def load_oracle(opportunity_ids) -> pd.DataFrame:
    _require_freeze()
    o = pd.read_parquet(ORACLE_PATH)
    return o[o["opportunity_id"].isin(set(opportunity_ids))].set_index("opportunity_id")


def true_probabilities(base: pd.DataFrame) -> pd.DataFrame:
    """Oracle P(recovery | action) aligned to `base` row order."""
    o = load_oracle(base["opportunity_id"]).reindex(base["opportunity_id"])
    out = {}
    for a in Action:
        col = f"p_recovery__{a.value}"
        out[a.value] = o[col].values if col in o.columns else np.full(len(base), np.nan)
    return pd.DataFrame(out, index=base.index)


def true_delta_ev(base: pd.DataFrame, true_p: pd.DataFrame,
                  elig: pd.DataFrame) -> pd.DataFrame:
    """dEV computed from oracle probabilities: the upper-bound economic view."""
    econs = [econ_for_row(r) for _, r in base.iterrows()]
    p0 = true_p[Action.DO_NOTHING.value].values
    ev0 = np.array([valuate_action(e, Action.DO_NOTHING, float(p)).expected_value
                    for e, p in zip(econs, p0)])
    out = {Action.DO_NOTHING.value: np.zeros(len(base))}
    for a in Action:
        if a is Action.DO_NOTHING:
            continue
        col = np.full(len(base), np.nan)
        m = elig[a.value].values & np.isfinite(true_p[a.value].values)
        for i in np.where(m)[0]:
            v = valuate_action(econs[i], a, float(true_p[a.value].values[i]))
            col[i] = v.expected_value - ev0[i]
        out[a.value] = col
    return pd.DataFrame(out, index=base.index)


def delta_p_analysis(probs: pd.DataFrame, true_p: pd.DataFrame,
                     elig: pd.DataFrame) -> pd.DataFrame:
    """Predicted vs oracle dP per action (Sections 37-38).

    This is the headline ML artifact: it measures whether the model learned
    *treatment response*, not merely how to rank customers.
    """
    rows = []
    pred_base = probs[Action.DO_NOTHING.value].values
    true_base = true_p[Action.DO_NOTHING.value].values
    for a in Action:
        if a is Action.DO_NOTHING:
            continue
        m = (elig[a.value].values
             & np.isfinite(probs[a.value].values)
             & np.isfinite(true_p[a.value].values))
        if m.sum() < 30:
            continue
        pd_ = probs[a.value].values[m] - pred_base[m]
        td_ = true_p[a.value].values[m] - true_base[m]
        rows.append({
            "action": a.value,
            "n": int(m.sum()),
            "pred_mean_dP": float(pd_.mean()),
            "true_mean_dP": float(td_.mean()),
            "bias": float((pd_ - td_).mean()),
            "mae": float(np.abs(pd_ - td_).mean()),
            "rmse": float(np.sqrt(((pd_ - td_) ** 2).mean())),
            "pearson": float(pearsonr(pd_, td_)[0]) if np.std(pd_) > 1e-9 else np.nan,
            "spearman": float(spearmanr(pd_, td_)[0]) if np.std(pd_) > 1e-9 else np.nan,
        })
    return pd.DataFrame(rows)


def oracle_policy_value(base: pd.DataFrame, actions: pd.Series,
                        true_p: pd.DataFrame) -> pd.DataFrame:
    """Expected business outcome of a policy under the oracle response surface.

    Returns per-opportunity economics so downstream code can bootstrap.
    """
    recs = []
    for i, (_, r) in enumerate(base.iterrows()):
        a = Action(actions.iloc[i])
        p = float(true_p[a.value].values[i])
        if not np.isfinite(p):
            a, p = Action.DO_NOTHING, float(true_p[Action.DO_NOTHING.value].values[i])
        econ = econ_for_row(r)
        v = valuate_action(econ, a, p)
        recs.append({
            "opportunity_id": r["opportunity_id"],
            "action": a.value,
            "conversion": p,
            "net_gmv": v.expected_recovered_gmv,
            "incentive_cost": p * v.incentive_cost,
            "fixed_cost": v.fixed_cost,
            "net_contribution": v.expected_value,
        })
    return pd.DataFrame(recs)