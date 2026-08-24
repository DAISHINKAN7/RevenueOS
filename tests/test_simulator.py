"""Simulator, logging-policy and leakage tests (spec Sections 79-81).

These run against a small in-memory dataset so they stay fast enough to run on
every commit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.actions import Action, ALL_ACTIONS, eligible_actions
from ml.config import FORBIDDEN_FEATURE_PREFIXES, SimulationConfig
from ml.simulation.behavior import response_surface, time_decay
from ml.simulation.customers import generate_customers
from ml.simulation.environment import build_environment
from ml.simulation.logging_policy import assign_actions
from ml.simulation.products import generate_products, shipping_cost_for


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(7)


@pytest.fixture
def cfg() -> SimulationConfig:
    return SimulationConfig(seed=7, n_customers=400, n_sessions=2_000)


def make_ctx(n: int, rng: np.random.Generator, **overrides) -> dict:
    ctx = {
        "cart_value": np.full(n, 5000.0),
        "shipping_fee_charged": np.full(n, 85.0),
        "failure_reason": np.array(["NONE"] * n),
        "opportunity_type": np.array(["CHECKOUT_ABANDONMENT"] * n),
        "minutes_since_event": np.full(n, 10.0),
        "attempt_number": np.full(n, 1.0),
        "in_bank_outage": np.zeros(n),
        "in_competitor_sale": np.zeros(n),
        "is_payday": np.zeros(n),
        "hidden_price_sensitivity": np.full(n, 0.5),
        "hidden_shipping_sensitivity": np.full(n, 0.5),
        "hidden_payment_friction": np.full(n, 0.5),
        "hidden_retry_tolerance": np.full(n, 0.5),
        "hidden_brand_loyalty": np.full(n, 0.5),
        "hidden_impulsivity": np.full(n, 0.5),
    }
    ctx.update(overrides)
    return ctx


# ------------------------------------------------------------ reproducibility
def test_same_seed_gives_identical_products():
    a = generate_products(30, np.random.default_rng(42))
    b = generate_products(30, np.random.default_rng(42))
    pd.testing.assert_frame_equal(a, b)


def test_different_seed_gives_different_products():
    a = generate_products(30, np.random.default_rng(1))
    b = generate_products(30, np.random.default_rng(2))
    assert not a["selling_price"].equals(b["selling_price"])


# ------------------------------------------------------------------ products
def test_products_have_positive_margin(rng):
    p = generate_products(60, rng)
    assert (p["selling_price"] > p["cost_of_goods"]).all()
    assert (p["gross_margin_percent"] > 0).all()


def test_shipping_scales_with_weight_and_zone(rng):
    p = generate_products(60, rng)
    heavy = p.nlargest(5, "weight_kg")["base_shipping_cost"].mean()
    light = p.nsmallest(5, "weight_kg")["base_shipping_cost"].mean()
    assert heavy > light
    assert shipping_cost_for(100, 3) > shipping_cost_for(100, 1)


def test_courier_disruption_raises_shipping_cost():
    assert shipping_cost_for(100, 1, courier_disrupted=True) > shipping_cost_for(100, 1)


# ----------------------------------------------------------------- customers
def test_hidden_traits_never_enter_observable_frame(rng):
    obs, hidden = generate_customers(300, rng, "2025-06-01")
    leaked = [c for c in obs.columns if any(c.startswith(p) for p in FORBIDDEN_FEATURE_PREFIXES)]
    assert leaked == [], f"latent traits leaked into observable customers: {leaked}"
    assert any(c.startswith("hidden_") for c in hidden.columns)


def test_observable_response_rate_is_noisy_not_a_copy_of_latent(rng):
    """Section 22: the observable rate must be a noisy finite-count estimate."""
    obs, hidden = generate_customers(3_000, rng, "2025-06-01")
    m = obs.merge(hidden, on="customer_id")
    m = m[m["coupon_offers_seen"] >= 1]
    corr = m["coupon_response_rate"].corr(m["hidden_price_sensitivity"])
    # Correlated (there is real signal) but far from a copy.
    assert 0.15 < corr < 0.85, f"correlation {corr:.2f} suggests a leak or no signal"


def test_new_customers_have_uninformative_history(rng):
    obs, _ = generate_customers(2_000, rng, "2025-06-01")
    new = obs[obs["customer_segment"] == "NEW_CUSTOMER"]
    assert new["coupon_offers_seen"].mean() < obs["coupon_offers_seen"].mean()


def test_response_rate_is_nan_when_no_offers_seen(rng):
    obs, _ = generate_customers(1_000, rng, "2025-06-01")
    zero = obs[obs["coupon_offers_seen"] == 0]
    if len(zero):
        assert zero["coupon_response_rate"].isna().all()


# ---------------------------------------------------------- response surface
def test_probabilities_are_valid(rng):
    ctx = make_ctx(200, rng)
    s = response_surface(ctx, ALL_ACTIONS, rng, 0.45)
    for a, p in s.items():
        assert np.all((p > 0) & (p < 1)), f"{a} produced out-of-range probabilities"


def test_shipping_sensitive_customers_respond_to_free_shipping(rng):
    hi = make_ctx(500, rng, hidden_shipping_sensitivity=np.full(500, 0.95))
    lo = make_ctx(500, rng, hidden_shipping_sensitivity=np.full(500, 0.05))
    s_hi = response_surface(hi, ALL_ACTIONS, rng, 0.0)
    s_lo = response_surface(lo, ALL_ACTIONS, rng, 0.0)
    uplift_hi = (s_hi[Action.FREE_SHIPPING] - s_hi[Action.DO_NOTHING]).mean()
    uplift_lo = (s_lo[Action.FREE_SHIPPING] - s_lo[Action.DO_NOTHING]).mean()
    assert uplift_hi > uplift_lo


def test_price_sensitive_customers_respond_to_discounts(rng):
    hi = make_ctx(500, rng, hidden_price_sensitivity=np.full(500, 0.95))
    lo = make_ctx(500, rng, hidden_price_sensitivity=np.full(500, 0.05))
    s_hi = response_surface(hi, ALL_ACTIONS, rng, 0.0)
    s_lo = response_surface(lo, ALL_ACTIONS, rng, 0.0)
    assert (s_hi[Action.MEDIUM_DISCOUNT] - s_hi[Action.DO_NOTHING]).mean() > \
           (s_lo[Action.MEDIUM_DISCOUNT] - s_lo[Action.DO_NOTHING]).mean()


def test_free_shipping_worthless_without_a_fee(rng):
    """No fee charged -> no shipping uplift, and contact fatigue makes it a net loss."""
    ctx = make_ctx(300, rng, shipping_fee_charged=np.zeros(300))
    s = response_surface(ctx, ALL_ACTIONS, rng, 0.0)
    assert np.all(s[Action.FREE_SHIPPING] <= s[Action.DO_NOTHING] + 1e-9)


def test_intervention_fatigue_makes_do_nothing_win_for_loyal(rng):
    """Contacting a customer who was already returning can be net negative."""
    loyal = make_ctx(400, rng, hidden_brand_loyalty=np.full(400, 0.95),
                     hidden_price_sensitivity=np.full(400, 0.15))
    s = response_surface(loyal, ALL_ACTIONS, rng, 0.0)
    assert s[Action.SMALL_DISCOUNT].mean() < s[Action.DO_NOTHING].mean()


def test_bank_timeout_favours_delayed_over_immediate_retry(rng):
    ctx = make_ctx(400, rng,
                   failure_reason=np.array(["BANK_TIMEOUT"] * 400),
                   opportunity_type=np.array(["PAYMENT_FAILURE"] * 400))
    s = response_surface(ctx, ALL_ACTIONS, rng, 0.0)
    assert s[Action.DELAYED_RETRY].mean() > s[Action.IMMEDIATE_RETRY].mean()


def test_insufficient_funds_immediate_retry_is_counterproductive(rng):
    ctx = make_ctx(400, rng,
                   failure_reason=np.array(["INSUFFICIENT_FUNDS"] * 400),
                   opportunity_type=np.array(["PAYMENT_FAILURE"] * 400))
    s = response_surface(ctx, ALL_ACTIONS, rng, 0.0)
    assert s[Action.IMMEDIATE_RETRY].mean() < s[Action.DO_NOTHING].mean()
    assert s[Action.DELAYED_RETRY].mean() > s[Action.IMMEDIATE_RETRY].mean()


def test_card_declined_favours_payment_method_switch(rng):
    ctx = make_ctx(400, rng,
                   failure_reason=np.array(["CARD_DECLINED"] * 400),
                   opportunity_type=np.array(["PAYMENT_FAILURE"] * 400))
    s = response_surface(ctx, ALL_ACTIONS, rng, 0.0)
    assert s[Action.PAYMENT_METHOD_SWITCH].mean() > s[Action.SMALL_DISCOUNT].mean()


def test_recovery_probability_decays_with_time(rng):
    fresh = make_ctx(300, rng, minutes_since_event=np.full(300, 5.0))
    stale = make_ctx(300, rng, minutes_since_event=np.full(300, 600.0))
    s_fresh = response_surface(fresh, ALL_ACTIONS, rng, 0.0)
    s_stale = response_surface(stale, ALL_ACTIONS, rng, 0.0)
    assert s_stale[Action.DO_NOTHING].mean() < s_fresh[Action.DO_NOTHING].mean()


def test_time_decay_halflife_behaviour():
    t = np.array([0.0, 90.0])
    typ = np.array(["CHECKOUT_ABANDONMENT"] * 2)
    d = time_decay(t, typ)
    assert d[0] == pytest.approx(1.0)
    assert d[1] == pytest.approx(0.5)


def test_noise_reduces_determinism(rng):
    ctx = make_ctx(500, rng)
    a = response_surface(ctx, ALL_ACTIONS, np.random.default_rng(1), 0.45)
    b = response_surface(ctx, ALL_ACTIONS, np.random.default_rng(2), 0.45)
    assert not np.allclose(a[Action.DO_NOTHING], b[Action.DO_NOTHING])


def test_discounts_damped_for_infrastructure_failures(rng):
    """A price cut does not fix a bank timeout."""
    infra = make_ctx(400, rng,
                     failure_reason=np.array(["BANK_TIMEOUT"] * 400),
                     opportunity_type=np.array(["PAYMENT_FAILURE"] * 400),
                     hidden_price_sensitivity=np.full(400, 0.9))
    intent = make_ctx(400, rng,
                      failure_reason=np.array(["NONE"] * 400),
                      hidden_price_sensitivity=np.full(400, 0.9))
    s_infra = response_surface(infra, ALL_ACTIONS, rng, 0.0)
    s_intent = response_surface(intent, ALL_ACTIONS, rng, 0.0)
    up_infra = (s_infra[Action.MEDIUM_DISCOUNT] - s_infra[Action.DO_NOTHING]).mean()
    up_intent = (s_intent[Action.MEDIUM_DISCOUNT] - s_intent[Action.DO_NOTHING]).mean()
    assert up_infra < up_intent * 0.6


def test_delayed_retry_beats_discount_on_bank_timeout(rng):
    ctx = make_ctx(500, rng,
                   failure_reason=np.array(["BANK_TIMEOUT"] * 500),
                   opportunity_type=np.array(["PAYMENT_FAILURE"] * 500))
    s = response_surface(ctx, ALL_ACTIONS, rng, 0.0)
    assert s[Action.DELAYED_RETRY].mean() > s[Action.MEDIUM_DISCOUNT].mean()


def test_switch_beats_discount_for_payment_friction(rng):
    ctx = make_ctx(500, rng,
                   failure_reason=np.array(["CARD_DECLINED"] * 500),
                   opportunity_type=np.array(["PAYMENT_FAILURE"] * 500),
                   hidden_payment_friction=np.full(500, 0.9))
    s = response_surface(ctx, ALL_ACTIONS, rng, 0.0)
    assert s[Action.PAYMENT_METHOD_SWITCH].mean() > s[Action.MEDIUM_DISCOUNT].mean()


def test_loyal_customers_have_high_unaided_baseline(rng):
    loyal = make_ctx(500, rng, hidden_brand_loyalty=np.full(500, 0.95))
    other = make_ctx(500, rng, hidden_brand_loyalty=np.full(500, 0.15))
    s_loyal = response_surface(loyal, ALL_ACTIONS, rng, 0.0)
    s_other = response_surface(other, ALL_ACTIONS, rng, 0.0)
    assert s_loyal[Action.DO_NOTHING].mean() > s_other[Action.DO_NOTHING].mean()
    # And there is less incremental room left for any intervention.
    up_loyal = (s_loyal[Action.MEDIUM_DISCOUNT] - s_loyal[Action.DO_NOTHING]).mean()
    up_other = (s_other[Action.MEDIUM_DISCOUNT] - s_other[Action.DO_NOTHING]).mean()
    assert up_loyal < up_other


# ---------------------------------------------------------- logging policy
def test_propensities_are_strictly_positive(rng):
    ctx = make_ctx(1_000, rng)
    logged = assign_actions(ctx, rng, exploration_rate=0.25, min_propensity=0.02)
    assert (logged["action_propensity"] > 0).all()
    assert (logged["action_propensity"] <= 1).all()


def test_propensity_vector_sums_to_one(rng):
    ctx = make_ctx(300, rng)
    logged = assign_actions(ctx, rng, 0.25, 0.02)
    for vec in logged["propensity_vector"]:
        assert sum(vec.values()) == pytest.approx(1.0)


def test_logged_action_has_matching_propensity(rng):
    ctx = make_ctx(400, rng)
    logged = assign_actions(ctx, rng, 0.25, 0.02)
    for _, row in logged.iterrows():
        assert row["propensity_vector"][row["action_taken"]] == pytest.approx(row["action_propensity"])


def test_exploration_rate_is_respected(rng):
    ctx = make_ctx(4_000, rng)
    logged = assign_actions(ctx, rng, exploration_rate=0.25, min_propensity=0.02)
    assert 0.21 < logged["is_exploration"].mean() < 0.29


def test_exploration_propensities_are_context_independent(rng):
    """Stratified weights are fixed, so exploration remains an exactly-known policy."""
    ctx = make_ctx(2_000, rng)
    logged = assign_actions(ctx, rng, 0.25, 0.02)
    expl = logged[logged["is_exploration"]]
    # Within one opportunity type every exploration row shares the same vector.
    vecs = {tuple(sorted(v.items())) for v in expl["propensity_vector"]}
    assert len(vecs) == 1


def test_retry_actions_oversampled_in_exploration(rng):
    """Retry actions must get materially more exploration support (rev 1.1.0)."""
    ctx = make_ctx(3_000, rng, opportunity_type=np.array(["PAYMENT_FAILURE"] * 3_000),
                   failure_reason=np.array(["BANK_TIMEOUT"] * 3_000))
    logged = assign_actions(ctx, rng, 1.0, 0.02)
    share = logged["action_taken"].value_counts(normalize=True)
    uniform = 1.0 / len(eligible_actions("PAYMENT_FAILURE"))
    assert share["DELAYED_RETRY"] > uniform * 1.5


def test_logging_policy_is_not_optimal(rng):
    """Historical behaviour must be imperfect (Section 35)."""
    ctx = make_ctx(3_000, rng)
    logged = assign_actions(ctx, rng, 0.25, 0.02)
    share = logged["action_taken"].value_counts(normalize=True)
    assert share.max() < 0.55, "logging policy is too concentrated to be realistic"
    assert share.get("DO_NOTHING", 0) > 0.15


def test_all_eligible_actions_get_support(rng):
    ctx = make_ctx(5_000, rng)
    logged = assign_actions(ctx, rng, 0.25, 0.02)
    seen = set(logged["action_taken"])
    expected = {a.value for a in eligible_actions("CHECKOUT_ABANDONMENT")}
    assert expected.issubset(seen), f"no support for {expected - seen}"


def test_retry_actions_never_logged_for_abandonment(rng):
    ctx = make_ctx(2_000, rng)
    logged = assign_actions(ctx, rng, 0.25, 0.02)
    assert not logged["action_taken"].isin(["IMMEDIATE_RETRY", "DELAYED_RETRY"]).any()


def test_importance_weights_are_bounded(rng):
    ctx = make_ctx(3_000, rng)
    logged = assign_actions(ctx, rng, 0.25, 0.02)
    assert (1.0 / logged["action_propensity"]).max() < 500


# ------------------------------------------------------------- environment
def test_hidden_environment_windows_are_sampled(cfg, rng):
    env = build_environment(cfg, rng)
    assert len(env.bank_outages) == cfg.n_bank_outages
    assert all(a < b for a, b in env.bank_outages)


def test_environment_membership_query(cfg, rng):
    env = build_environment(cfg, rng)
    start, end = env.bank_outages[0]
    inside = pd.Series([start + pd.Timedelta(minutes=1)])
    outside = pd.Series([end + pd.Timedelta(days=30)])
    assert env.in_bank_outage(inside)[0]
    assert not env.in_bank_outage(outside)[0]
