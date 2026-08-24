"""Off-policy evaluation (spec Sections 54-67).

Estimates the value of a target policy using ONLY logged actions, logged
propensities and logged outcomes. The oracle is never consulted here — that is
the entire point: this is the evidence stream that does not ask the simulator
to reveal counterfactuals.

Reward
------
Realised net contribution in rupees::

    R = base_contribution_margin - incentive_cost   (if recovered)
        - fixed_action_cost                          (always)

Not binary recovery: a binary reward would silently reintroduce the
conversion-maximising objective this project argues against.

Estimators
----------
IPS    mean( 1[A_logged = pi(X)] / p_logged * R )
SNIPS  sum(w R) / sum(w), self-normalised
DR     mean( q(X, pi(X)) + 1[A=pi(X)]/p * (R - q(X, A)) )

The DR reward model q is cross-fitted (K=5) over the evaluation rows, so the
q-value used for a row is always out-of-fold. Weight clipping is reported as a
sensitivity band, never applied silently.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold

from ml.actions import Action
from ml.financial_engine import OpportunityEconomics, incentive_cost, fixed_action_cost

CLIP_LEVELS = [None, 20.0, 10.0]
N_BOOTSTRAP = 1000
SEED = 42


def realised_reward(base: pd.DataFrame) -> np.ndarray:
    """Rupee reward actually observed for the logged action."""
    out = np.zeros(len(base))
    for i, (_, r) in enumerate(base.iterrows()):
        a = Action(r["action"])
        econ = OpportunityEconomics(
            cart_value=float(r["cart_value"]), cogs=float(r["cart_cogs"]),
            shipping_cost=float(r["shipping_cost"]),
            shipping_fee_charged=float(r["shipping_fee_charged"]),
        )
        fixed = fixed_action_cost(a)
        if r["outcome"] == 1:
            out[i] = econ.base_contribution_margin - incentive_cost(econ, a) - fixed
        else:
            out[i] = -fixed  # incentive is never paid when recovery fails
    return out


def kish_ess(w: np.ndarray) -> float:
    w = np.asarray(w, dtype=float)
    return float(w.sum() ** 2 / np.square(w).sum()) if w.sum() > 0 else 0.0


def _fit_reward_model(X: pd.DataFrame, y: np.ndarray, feature_cols) -> HistGradientBoostingRegressor:
    m = HistGradientBoostingRegressor(max_iter=200, random_state=SEED)
    m.fit(X[feature_cols], y)
    return m


def cross_fitted_q(base: pd.DataFrame, reward: np.ndarray, target_actions: pd.Series,
                   feature_cols, n_splits: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Out-of-fold q(X, A_logged) and q(X, pi(X)) for the DR estimator."""
    X = base[feature_cols].copy()
    X["__action"] = pd.Categorical(base["action"]).codes
    action_codes = dict(zip(pd.Categorical(base["action"]).categories,
                            range(len(pd.Categorical(base["action"]).categories))))
    cols = list(X.columns)

    q_logged = np.zeros(len(base))
    q_target = np.zeros(len(base))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    for tr, te in kf.split(X):
        m = _fit_reward_model(X.iloc[tr], reward[tr], cols)
        q_logged[te] = m.predict(X.iloc[te][cols])
        Xt = X.iloc[te].copy()
        Xt["__action"] = [action_codes.get(a, -1) for a in target_actions.iloc[te]]
        q_target[te] = m.predict(Xt[cols])
    return q_logged, q_target


def evaluate_policy(base: pd.DataFrame, target_actions: pd.Series, reward: np.ndarray,
                    feature_cols, clip: float | None = None,
                    q_logged: np.ndarray | None = None,
                    q_target: np.ndarray | None = None) -> dict:
    """IPS / SNIPS / DR for one deterministic target policy."""
    p_log = base["action_propensity"].values.astype(float)
    match = (base["action"].values == target_actions.values).astype(float)
    w = match / np.maximum(p_log, 1e-9)
    if clip is not None:
        w = np.minimum(w, clip)

    ips = float(np.mean(w * reward))
    snips = float(np.sum(w * reward) / np.sum(w)) if w.sum() > 0 else np.nan
    dr = float(np.mean(q_target + w * (reward - q_logged))) if q_logged is not None else np.nan

    return {
        "ips": ips, "snips": snips, "dr": dr,
        "ess": kish_ess(w[w > 0]),
        "match_rate": float(match.mean()),
        "max_weight": float(w.max()) if len(w) else 0.0,
        "n_matched": int(match.sum()),
    }


def bootstrap_dr(base: pd.DataFrame, target_actions: pd.Series, reward: np.ndarray,
                 q_logged: np.ndarray, q_target: np.ndarray,
                 clip: float | None = None, n: int = N_BOOTSTRAP) -> tuple[float, float, float]:
    rng = np.random.default_rng(SEED)
    p_log = base["action_propensity"].values.astype(float)
    match = (base["action"].values == target_actions.values).astype(float)
    w = match / np.maximum(p_log, 1e-9)
    if clip is not None:
        w = np.minimum(w, clip)
    psi = q_target + w * (reward - q_logged)
    idx = rng.integers(0, len(psi), size=(n, len(psi)))
    draws = psi[idx].mean(axis=1)
    return float(draws.mean()), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def support_check(base: pd.DataFrame, target_actions: pd.Series,
                  min_support: int = 30) -> pd.DataFrame:
    """Per-action logging support for the target policy (Section 61)."""
    rows = []
    logged_counts = base["action"].value_counts()
    for a, n_target in target_actions.value_counts().items():
        n_logged = int(logged_counts.get(a, 0))
        rows.append({
            "action": a,
            "n_selected_by_target": int(n_target),
            "n_logged": n_logged,
            "supported": n_logged >= min_support,
        })
    return pd.DataFrame(rows).sort_values("n_selected_by_target", ascending=False)


# ------------------------------------------------------------------ toy test
def toy_validation() -> dict:
    """Analytically checkable OPE case (Section 109).

    Two actions, known logging propensity, known reward means. The target policy
    always plays action 1, whose true value is 10.0. IPS/SNIPS/DR should all
    land near 10.0; if they do not, the estimator implementation is buggy and
    nothing downstream can be trusted.
    """
    rng = np.random.default_rng(0)
    n = 20_000
    p_log = 0.3
    a = (rng.random(n) < p_log).astype(int)   # action 1 with prob 0.3
    r = np.where(a == 1, rng.normal(10.0, 1.0, n), rng.normal(4.0, 1.0, n))

    prop = np.where(a == 1, p_log, 1 - p_log)
    match = (a == 1).astype(float)
    w = match / prop
    ips = float(np.mean(w * r))
    snips = float(np.sum(w * r) / np.sum(w))
    q_logged = np.where(a == 1, 10.0, 4.0)
    q_target = np.full(n, 10.0)
    dr = float(np.mean(q_target + w * (r - q_logged)))
    return {"true_value": 10.0, "ips": ips, "snips": snips, "dr": dr}


if __name__ == "__main__":
    print(json.dumps(toy_validation(), indent=2))