"""Unit tests for the financial engine (spec Sections 6, 7, 79).

The engine must be provably correct before any ML work begins: every downstream
metric inherits its arithmetic.
"""

from __future__ import annotations

import math

import pytest

from ml.actions import Action
from ml.financial_engine import (
    FinancialInputError,
    OpportunityEconomics,
    fixed_action_cost,
    incentive_cost,
    rank_actions,
    recovered_gmv_net_of_discount,
    select_best_action,
    valuate_action,
)


@pytest.fixture
def econ() -> OpportunityEconomics:
    # ₹5,000 cart, ₹3,700 COGS, ₹100 fulfilment, ₹85 charged to customer.
    # base CM = 5000 - 3700 + 85 - 100 = ₹1,285
    return OpportunityEconomics(
        cart_value=5000.0, cogs=3700.0, shipping_cost=100.0, shipping_fee_charged=85.0
    )


# ---------------------------------------------------------------- base maths
def test_base_contribution_margin(econ):
    assert econ.base_contribution_margin == pytest.approx(1285.0)


def test_margin_percent(econ):
    assert econ.base_margin_percent == pytest.approx(25.7)


# ------------------------------------------------------- conditional vs fixed
def test_discount_is_conditional_not_fixed(econ):
    """A discount costs nothing when the customer does not convert."""
    v = valuate_action(econ, Action.MEDIUM_DISCOUNT, recovery_probability=0.0)
    # Only the unconditional fixed cost is incurred.
    assert v.expected_value == pytest.approx(-fixed_action_cost(Action.MEDIUM_DISCOUNT))
    assert v.incentive_cost == pytest.approx(500.0)  # 10% of 5000, but unrealised


def test_fixed_cost_is_unconditional(econ):
    v = valuate_action(econ, Action.HUMAN_ESCALATION, recovery_probability=0.0)
    assert v.expected_value == pytest.approx(-fixed_action_cost(Action.HUMAN_ESCALATION))
    assert v.incentive_cost == 0.0


def test_ev_matches_canonical_formula(econ):
    p = 0.62
    v = valuate_action(econ, Action.SMALL_DISCOUNT, p)
    expected = p * (1285.0 - 250.0) - 2.0
    assert v.expected_value == pytest.approx(expected)


def test_free_shipping_incentive_is_the_waived_fee(econ):
    assert incentive_cost(econ, Action.FREE_SHIPPING) == pytest.approx(85.0)


def test_free_shipping_worthless_when_no_fee_charged():
    e = OpportunityEconomics(cart_value=6000, cogs=4000, shipping_cost=120, shipping_fee_charged=0.0)
    assert incentive_cost(e, Action.FREE_SHIPPING) == pytest.approx(0.0)


def test_do_nothing_costs_nothing(econ):
    assert incentive_cost(econ, Action.DO_NOTHING) == 0.0
    assert fixed_action_cost(Action.DO_NOTHING) == 0.0


# ------------------------------------------------------------ no double count
def test_no_double_counting_of_discount(econ):
    """The incentive is subtracted exactly once, from the margin."""
    v = valuate_action(econ, Action.MEDIUM_DISCOUNT, 1.0)
    # margin_if_recovered = base CM - discount, and nothing more.
    assert v.margin_if_recovered == pytest.approx(1285.0 - 500.0)


def test_recovered_gmv_is_net_of_discount(econ):
    assert recovered_gmv_net_of_discount(econ, Action.MEDIUM_DISCOUNT) == pytest.approx(4500.0)
    assert recovered_gmv_net_of_discount(econ, Action.FREE_SHIPPING) == pytest.approx(5000.0)


# --------------------------------------------------------------- return risk
def test_return_risk_reduces_expected_value():
    kw = dict(cart_value=5000.0, cogs=3700.0, shipping_cost=100.0, shipping_fee_charged=85.0)
    clean = valuate_action(OpportunityEconomics(**kw), Action.DO_NOTHING, 0.5)
    risky = valuate_action(
        OpportunityEconomics(**kw, return_probability=0.25), Action.DO_NOTHING, 0.5
    )
    assert risky.expected_value < clean.expected_value


def test_expected_return_loss_decomposition(econ):
    e = OpportunityEconomics(
        cart_value=5000.0, cogs=3700.0, shipping_cost=100.0,
        shipping_fee_charged=85.0, return_probability=0.2, return_handling_cost=120.0,
    )
    v = valuate_action(e, Action.DO_NOTHING, 1.0)
    assert v.expected_return_loss == pytest.approx(0.2 * (1285.0 + 120.0))
    assert v.margin_if_recovered == pytest.approx(1285.0 - v.expected_return_loss)


# ------------------------------------------------------------- delta EV logic
def test_delta_ev_is_measured_against_do_nothing(econ):
    probs = {Action.DO_NOTHING: 0.25, Action.SMALL_DISCOUNT: 0.55}
    ranked = rank_actions(econ, probs)
    nothing = next(v for v in ranked if v.action is Action.DO_NOTHING)
    disc = next(v for v in ranked if v.action is Action.SMALL_DISCOUNT)
    assert nothing.delta_ev == pytest.approx(0.0)
    assert disc.delta_ev == pytest.approx(disc.expected_value - nothing.expected_value)


def test_ranking_is_by_delta_ev_descending(econ):
    probs = {
        Action.DO_NOTHING: 0.25, Action.SMALL_DISCOUNT: 0.55,
        Action.MEDIUM_DISCOUNT: 0.74, Action.FREE_SHIPPING: 0.66,
    }
    ranked = rank_actions(econ, probs)
    assert [v.delta_ev for v in ranked] == sorted([v.delta_ev for v in ranked], reverse=True)


def test_highest_conversion_action_can_lose_on_economics(econ):
    """The central product thesis, encoded as a test.

    MEDIUM_DISCOUNT converts best but destroys contribution; FREE_SHIPPING wins.
    """
    probs = {
        Action.DO_NOTHING: 0.25,
        Action.MEDIUM_DISCOUNT: 0.74,   # highest P(recovery)
        Action.FREE_SHIPPING: 0.66,
    }
    best = select_best_action(econ, probs)
    ranked = rank_actions(econ, probs)
    top_by_probability = max(ranked, key=lambda v: v.recovery_probability)

    assert top_by_probability.action is Action.MEDIUM_DISCOUNT
    assert best.action is Action.FREE_SHIPPING


def test_do_nothing_selected_when_no_action_has_positive_delta(econ):
    probs = {Action.DO_NOTHING: 0.60, Action.MEDIUM_DISCOUNT: 0.61}
    assert select_best_action(econ, probs).action is Action.DO_NOTHING


def test_restraint_on_thin_margin_cart():
    """A 10% discount on a low-margin cart should never be worth taking."""
    thin = OpportunityEconomics(
        cart_value=2000.0, cogs=1780.0, shipping_cost=90.0, shipping_fee_charged=70.0
    )
    probs = {Action.DO_NOTHING: 0.30, Action.MEDIUM_DISCOUNT: 0.70}
    assert select_best_action(thin, probs).action is Action.DO_NOTHING


# ------------------------------------------------------------- max downside
def test_maximum_downside_is_bounded(econ):
    v = valuate_action(econ, Action.FREE_SHIPPING, 0.5)
    assert v.maximum_downside == pytest.approx(2.0 + 85.0)
    assert v.maximum_downside < econ.base_contribution_margin


# --------------------------------------------------------------- validation
@pytest.mark.parametrize("bad", [-0.1, 1.1, float("nan"), float("inf")])
def test_invalid_probability_rejected(econ, bad):
    with pytest.raises(FinancialInputError):
        valuate_action(econ, Action.DO_NOTHING, bad)


def test_negative_cart_rejected():
    with pytest.raises(FinancialInputError):
        OpportunityEconomics(cart_value=-100, cogs=10, shipping_cost=0, shipping_fee_charged=0)


def test_return_plus_cancellation_over_one_rejected():
    with pytest.raises(FinancialInputError):
        OpportunityEconomics(
            cart_value=1000, cogs=500, shipping_cost=0, shipping_fee_charged=0,
            return_probability=0.7, cancellation_probability=0.5,
        )


def test_rank_requires_do_nothing(econ):
    with pytest.raises(FinancialInputError):
        rank_actions(econ, {Action.SMALL_DISCOUNT: 0.5})


def test_retry_actions_rejected_for_abandonment(econ):
    with pytest.raises(FinancialInputError):
        rank_actions(
            econ,
            {Action.DO_NOTHING: 0.3, Action.IMMEDIATE_RETRY: 0.5},
            opportunity_type="CHECKOUT_ABANDONMENT",
        )


def test_retry_actions_allowed_for_payment_failure(econ):
    ranked = rank_actions(
        econ,
        {Action.DO_NOTHING: 0.3, Action.IMMEDIATE_RETRY: 0.5},
        opportunity_type="PAYMENT_FAILURE",
    )
    assert len(ranked) == 2


def test_all_outputs_finite(econ):
    for v in rank_actions(econ, {Action.DO_NOTHING: 0.3, Action.FREE_SHIPPING: 0.6}):
        for k, val in v.as_dict().items():
            if isinstance(val, float):
                assert math.isfinite(val), f"{v.action}.{k} is not finite"
