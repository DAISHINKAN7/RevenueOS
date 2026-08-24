"""Phase 3 & 4 test suite (spec Section 108).

Covers feature correctness, temporal leakage, oracle isolation, model output
contracts, economics, OPE estimator correctness, and policy fallbacks.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml.actions import Action
from ml.features.build import (
    ALL_FEATURES, LeakageError, _smooth, add_action_features,
    assert_no_leakage, build_base_features, context_eligible_actions,
)
from ml.financial_engine import OpportunityEconomics, valuate_action
from ml.evaluation import ope as OPE

PROC = Path("data/processed")
DATA = Path("data/generated")
pytestmark = pytest.mark.skipif(
    not (PROC / "test_features.parquet").exists(),
    reason="run `make data && python -m ml.features.build` first",
)


@pytest.fixture(scope="module")
def test_df() -> pd.DataFrame:
    return pd.read_parquet(PROC / "test_features.parquet")


# ------------------------------------------------------------------ leakage
def test_no_forbidden_columns_in_features(test_df):
    assert_no_leakage(test_df[ALL_FEATURES])


def test_leakage_detector_actually_fires():
    bad = pd.DataFrame({"hidden_price_sensitivity": [0.5], "cart_value": [100]})
    with pytest.raises(LeakageError):
        assert_no_leakage(bad)


def test_oracle_columns_rejected():
    bad = pd.DataFrame({"p_recovery__FREE_SHIPPING": [0.5]})
    with pytest.raises(LeakageError):
        assert_no_leakage(bad)


def test_outcome_not_in_feature_list():
    assert "outcome" not in ALL_FEATURES
    assert "converted_after_intervention" not in ALL_FEATURES
    assert not any(f.startswith("hidden_") for f in ALL_FEATURES)


def test_splits_are_chronologically_ordered():
    tr = pd.read_parquet(PROC / "train_features.parquet")
    va = pd.read_parquet(PROC / "validation_features.parquet")
    te = pd.read_parquet(PROC / "test_features.parquet")
    assert tr["detected_at"].max() <= va["detected_at"].min()
    assert va["detected_at"].max() <= te["detected_at"].min()


# --------------------------------------------------------- oracle isolation
def _code_lines(path: Path) -> str:
    """Source with comments and docstrings stripped, so prose about the oracle
    does not trip the isolation check — only real code does."""
    import ast
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
            if (node.body and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)):
                node.body.pop(0)
    return ast.unparse(tree)


def test_training_modules_never_read_the_oracle():
    """Only the designated evaluation module may READ oracle contents.

    Hashing the oracle file for the provenance manifest is allowed and desirable;
    parsing its rows into a training frame is not.
    """
    for d in ["ml/features", "ml/models", "ml/calibration", "ml/simulation"]:
        for py in Path(d).rglob("*.py"):
            code = _code_lines(py)
            assert "read_parquet" not in code or "oracle" not in code.split("read_parquet")[1][:60], (
                f"{py} appears to read oracle contents"
            )
            assert "oracle_eval" not in code, f"{py} imports the oracle module"


def test_oracle_eval_is_the_only_reader():
    readers = []
    for py in Path("ml").rglob("*.py"):
        code = _code_lines(py)
        for chunk in code.split("read_parquet")[1:]:
            if "oracle" in chunk[:60]:
                readers.append(py.name)
    assert set(readers) <= {"oracle_eval.py"}, readers


def test_scoring_does_not_import_oracle():
    code = _code_lines(Path("ml/evaluation/scoring.py"))
    assert "oracle" not in code.lower()


# ---------------------------------------------------------------- smoothing
def test_smoothing_pulls_sparse_rates_toward_prior():
    # 1 success out of 1 observation should not be read as 100%.
    r = _smooth([1], [1], global_rate=0.3, alpha=5.0)[0]
    assert 0.3 < r < 0.6


def test_smoothing_respects_large_samples():
    r = _smooth([80], [100], global_rate=0.3, alpha=5.0)[0]
    assert r == pytest.approx(0.79, abs=0.02)


def test_cold_start_flags_present(test_df):
    for c in ["coupon_history_missing", "shipping_history_missing", "retry_history_missing"]:
        assert c in test_df.columns
        assert set(test_df[c].unique()) <= {0, 1}


# -------------------------------------------------------------- eligibility
def test_retry_ineligible_for_abandonment():
    row = pd.Series({"opportunity_type": "CHECKOUT_ABANDONMENT", "shipping_fee_charged": 50.0})
    acts = context_eligible_actions(row)
    assert Action.IMMEDIATE_RETRY not in acts
    assert Action.DELAYED_RETRY not in acts


def test_free_shipping_ineligible_without_fee():
    row = pd.Series({"opportunity_type": "CHECKOUT_ABANDONMENT", "shipping_fee_charged": 0.0})
    assert Action.FREE_SHIPPING not in context_eligible_actions(row)


def test_switch_requires_payment_failure():
    row = pd.Series({"opportunity_type": "CHECKOUT_ABANDONMENT", "shipping_fee_charged": 50.0})
    assert Action.PAYMENT_METHOD_SWITCH not in context_eligible_actions(row)
    row2 = pd.Series({"opportunity_type": "PAYMENT_FAILURE", "shipping_fee_charged": 50.0})
    assert Action.PAYMENT_METHOD_SWITCH in context_eligible_actions(row2)


def test_do_nothing_always_eligible():
    for t in ["CHECKOUT_ABANDONMENT", "PAYMENT_FAILURE"]:
        row = pd.Series({"opportunity_type": t, "shipping_fee_charged": 0.0})
        assert Action.DO_NOTHING in context_eligible_actions(row)


# ----------------------------------------------------------- action features
def test_action_features_reflect_action_economics():
    df = pd.DataFrame({"cart_value": [5000.0], "shipping_fee_charged": [85.0],
                       "base_contribution_margin": [1285.0], "action": ["MEDIUM_DISCOUNT"]})
    out = add_action_features(df)
    assert out["action_discount_amount"].iloc[0] == pytest.approx(500.0)
    assert out["action_incentive_cost"].iloc[0] == pytest.approx(500.0)


def test_free_shipping_incentive_is_the_fee():
    df = pd.DataFrame({"cart_value": [5000.0], "shipping_fee_charged": [85.0],
                       "base_contribution_margin": [1285.0], "action": ["FREE_SHIPPING"]})
    out = add_action_features(df)
    assert out["action_incentive_cost"].iloc[0] == pytest.approx(85.0)


def test_same_action_different_cart_gives_different_cost():
    """The action label must not hide materially different monetary amounts."""
    df = pd.DataFrame({"cart_value": [1000.0, 20000.0], "shipping_fee_charged": [50.0, 50.0],
                       "base_contribution_margin": [300.0, 6000.0],
                       "action": ["MEDIUM_DISCOUNT"] * 2})
    out = add_action_features(df)
    assert out["action_incentive_cost"].iloc[0] != out["action_incentive_cost"].iloc[1]


# --------------------------------------------------------------- model I/O
@pytest.mark.skipif(not Path("ml/artifacts/recovery_model.pkl").exists(),
                    reason="train the model first")
def test_predictions_are_valid_probabilities(test_df):
    from ml.evaluation.scoring import RecoveryPredictor
    p = RecoveryPredictor().predict(test_df.head(500))
    assert np.all(np.isfinite(p))
    assert p.min() >= 0.0 and p.max() <= 1.0


@pytest.mark.skipif(not Path("ml/artifacts/recovery_model.pkl").exists(),
                    reason="train the model first")
def test_predictions_are_deterministic(test_df):
    from ml.evaluation.scoring import RecoveryPredictor
    pred = RecoveryPredictor()
    a = pred.predict(test_df.head(200))
    b = pred.predict(test_df.head(200))
    np.testing.assert_array_equal(a, b)


@pytest.mark.skipif(not Path("ml/artifacts/recovery_model.pkl").exists(),
                    reason="train the model first")
def test_scoring_respects_eligibility(test_df):
    from ml.evaluation.scoring import RecoveryPredictor, score_all_actions
    base = test_df.head(300)
    probs, elig = score_all_actions(base, RecoveryPredictor())
    abandon = base["opportunity_type"] == "CHECKOUT_ABANDONMENT"
    assert probs.loc[abandon.values, "DELAYED_RETRY"].isna().all()


# ---------------------------------------------------------------- economics
def test_delta_ev_zero_for_do_nothing():
    e = OpportunityEconomics(5000, 3700, 100, 85)
    v = valuate_action(e, Action.DO_NOTHING, 0.3,
                       baseline_probability=0.3,
                       baseline_expected_value=valuate_action(e, Action.DO_NOTHING, 0.3).expected_value)
    assert v.delta_ev == pytest.approx(0.0)


def test_policy_falls_back_to_do_nothing_when_no_positive_delta():
    from ml.evaluation.scoring import policy_revenueos
    base = pd.DataFrame({"cart_value": [5000.0]})
    probs = pd.DataFrame({"DO_NOTHING": [0.5], "SMALL_DISCOUNT": [0.5]})
    dev = pd.DataFrame({"DO_NOTHING": [0.0], "SMALL_DISCOUNT": [-10.0]})
    assert policy_revenueos(base, probs, dev).iloc[0] == "DO_NOTHING"


def test_policy_never_selects_ineligible_action(test_df):
    """RevenueOS must not pick a retry for a checkout abandonment."""
    import json
    p = Path("evaluation/results/policy_results.csv")
    if not p.exists():
        pytest.skip("run the evaluation first")
    hr = pd.read_csv("evaluation/results/high_regret_cases.csv")
    bad = hr[(hr["opportunity_type"] == "CHECKOUT_ABANDONMENT")
             & hr["revenueos_action"].isin(["IMMEDIATE_RETRY", "DELAYED_RETRY"])]
    assert len(bad) == 0


# ---------------------------------------------------------------------- OPE
def test_toy_ope_recovers_known_value():
    """Analytically checkable case: true target-policy value is 10.0."""
    r = OPE.toy_validation()
    assert abs(r["ips"] - 10.0) < 0.5
    assert abs(r["snips"] - 10.0) < 0.2
    assert abs(r["dr"] - 10.0) < 0.2


def test_kish_ess_bounds():
    assert OPE.kish_ess(np.ones(100)) == pytest.approx(100.0)
    skewed = np.array([100.0] + [0.01] * 99)
    assert OPE.kish_ess(skewed) < 5


def test_snips_is_normalised():
    """SNIPS of a constant reward must return that constant."""
    n = 1000
    base = pd.DataFrame({
        "action": ["A"] * n, "action_propensity": np.full(n, 0.25),
        "cart_value": np.full(n, 1000.0), "cart_cogs": np.full(n, 500.0),
        "shipping_cost": np.zeros(n), "shipping_fee_charged": np.zeros(n),
        "outcome": np.ones(n),
    })
    reward = np.full(n, 7.0)
    r = OPE.evaluate_policy(base, pd.Series(["A"] * n), reward, [], clip=None,
                            q_logged=np.zeros(n), q_target=np.zeros(n))
    assert r["snips"] == pytest.approx(7.0)


def test_propensities_are_valid_in_logged_data(test_df):
    p = test_df["action_propensity"]
    assert (p > 0).all() and (p <= 1).all()
    assert np.isfinite(1.0 / p).all()


def test_reward_has_no_incentive_cost_on_failure():
    """An unrecovered opportunity pays the fixed cost only."""
    base = pd.DataFrame({
        "action": ["MEDIUM_DISCOUNT"], "cart_value": [5000.0], "cart_cogs": [3700.0],
        "shipping_cost": [100.0], "shipping_fee_charged": [85.0], "outcome": [0],
    })
    r = OPE.realised_reward(base)
    assert r[0] == pytest.approx(-2.0)


def test_reward_on_success_nets_the_incentive():
    base = pd.DataFrame({
        "action": ["MEDIUM_DISCOUNT"], "cart_value": [5000.0], "cart_cogs": [3700.0],
        "shipping_cost": [100.0], "shipping_fee_charged": [85.0], "outcome": [1],
    })
    r = OPE.realised_reward(base)
    assert r[0] == pytest.approx(1285.0 - 500.0 - 2.0)