"""Deterministic bounded negotiation (spec §42).

The single most important property of this module: **no language model can reach
it, and no counter-price is invented.**

A counter-offer is not a haggling heuristic. It is the merchant's *reserve
price* — the lowest price at which every merchant policy still holds — computed
in closed form and then quoted. Ask the engine for a price below the reserve and
it returns exactly the reserve. Ask twice and you get the same number, because
it is a function of COGS, fulfilment, return rate and merchant policy, not of
conversation history.

This mirrors `PolicyEngine` on the recovery side, deliberately:

    evaluate() takes no text parameter.

A buyer agent's prose ("my client really needs this at 4000, please") has no
channel into the arithmetic. That is a structural defence, not a filter.

Economics
---------
For unit price ``R``, quantity ``q``, shipping fee charged ``fee``, fulfilment
cost ``F`` and return rate ``r``::

    gross_contribution = R·q + fee − C·q − F
    net_contribution   = (1−r)·gross_contribution − r·F
    revenue            = R·q + fee
    margin%            = 100 · net_contribution / revenue

A return costs the merchant the contribution it had booked *and* reverse
logistics — hence the ``− r·F`` term. This is the same shape as the recovery
engine's `expected_return_loss`.

Binding constraints
-------------------
1. ``RULE_AC_DISCOUNT_PERCENT_CEILING``  — max_autonomous_discount_percent
2. ``RULE_AC_DISCOUNT_AMOUNT_CEILING``   — max_autonomous_discount_amount
3. ``RULE_AC_MARGIN_FLOOR``              — minimum_remaining_contribution_margin_percent
4. ``RULE_AC_INVENTORY_AVAILABLE``       — inventory_level
5. ``RULE_AC_HUMAN_APPROVAL_THRESHOLD``  — human_approval_required_above_discount
6. ``RULE_AC_HIGH_VALUE_ORDER``          — high_value_order_threshold
7. ``RULE_AC_MAX_ROUNDS``                — negotiation cannot run forever

The reserve price is ``max`` of the price floors implied by 1–3, so the *binding*
constraint is always identifiable and is reported by name.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import ROUND_CEILING, ROUND_HALF_UP, Decimal
from enum import Enum

from backend.app.services.catalog import ProductEconomics

NEGOTIATION_POLICY_VERSION = "ac-policy-1.0.0"

MAX_NEGOTIATION_ROUNDS = 3

PAISE = Decimal("0.01")
RUPEE = Decimal("1")


class OfferDecision(str, Enum):
    ACCEPT = "ACCEPT"
    COUNTER = "COUNTER"
    REJECT = "REJECT"
    REQUIRE_APPROVAL = "REQUIRE_APPROVAL"


@dataclass(frozen=True)
class NegotiationRule:
    rule_id: str
    passed: bool
    reason: str
    input_value: str | None = None
    threshold: str | None = None

    def as_dict(self) -> dict:
        return {
            "rule_id": self.rule_id,
            "passed": self.passed,
            "reason": self.reason,
            "input_value": self.input_value,
            "threshold": self.threshold,
        }


@dataclass(frozen=True)
class OfferEvaluation:
    decision: OfferDecision
    reason_code: str
    product_id: str
    quantity: int
    list_price: Decimal
    requested_unit_price: Decimal
    final_unit_price: Decimal
    reserve_unit_price: Decimal
    binding_constraint: str
    shipping_fee_charged: Decimal
    fulfilment_cost: Decimal
    order_total: Decimal
    net_contribution: Decimal
    margin_percent: Decimal
    discount_percent: Decimal
    discount_amount: Decimal
    rules: list[NegotiationRule] = field(default_factory=list)
    policy_version: str = NEGOTIATION_POLICY_VERSION

    def as_dict(self) -> dict:
        f = lambda d: float(d)  # noqa: E731
        return {
            "decision": self.decision.value,
            "reason_code": self.reason_code,
            "product_id": self.product_id,
            "quantity": self.quantity,
            "list_price": f(self.list_price),
            "requested_unit_price": f(self.requested_unit_price),
            "final_unit_price": f(self.final_unit_price),
            "reserve_unit_price": f(self.reserve_unit_price),
            "binding_constraint": self.binding_constraint,
            "shipping_fee_charged": f(self.shipping_fee_charged),
            "order_total": f(self.order_total),
            "net_contribution": f(self.net_contribution),
            "margin_percent": f(self.margin_percent),
            "discount_percent": f(self.discount_percent),
            "discount_amount": f(self.discount_amount),
            "policy_version": self.policy_version,
            "rules": [r.as_dict() for r in self.rules],
            # Deliberately absent: cost_of_goods, fulfilment_cost.
        }

    def as_merchant_dict(self) -> dict:
        """Full disclosure, for the merchant dashboard and audit trail only."""
        d = self.as_dict()
        d["fulfilment_cost"] = float(self.fulfilment_cost)
        return d


def _policy():
    """Load the same MerchantPolicy the recovery engine uses.

    Field access is defensive: this module must not break the app if a policy
    field is renamed, but it must also never *invent* a permissive default.
    The fallbacks below are the spec §9 values, which are the tightest
    reasonable reading.
    """
    from backend.app.core.config import MerchantPolicy

    p = MerchantPolicy.load()
    get = lambda name, default: Decimal(str(getattr(p, name, default)))  # noqa: E731
    return {
        "max_discount_percent": get("max_autonomous_discount_percent", 7),
        "max_discount_amount": get("max_autonomous_discount_amount", 300),
        "min_margin_percent": get("minimum_remaining_contribution_margin_percent", 15),
        # NOTE: the real field is `..._amount`. Verified against
        # backend/app/core/config.py; a test asserts the attribute exists so a
        # rename cannot silently fall back to the default.
        "approval_above_discount": get("human_approval_required_above_discount_amount", 250),
        "high_value_threshold": get("high_value_order_threshold", 10000),
        "raw": p,
    }


def shipping_fee_for(econ: ProductEconomics, order_subtotal: Decimal) -> Decimal:
    """What the buyer is charged for shipping.

    Free over ₹999 — a standing merchant term, not a negotiated concession, so
    it is applied before negotiation begins and is not a lever the agent can
    pull. Below that, the buyer pays a flat ₹85.
    """
    return Decimal("0") if order_subtotal >= Decimal("999") else Decimal("85")


def _margin_floor_price(
    econ: ProductEconomics,
    quantity: int,
    fee: Decimal,
    fulfilment: Decimal,
    min_margin_pct: Decimal,
) -> Decimal | None:
    """Closed-form lowest unit price satisfying the margin floor.

    Derivation, with ``m = min_margin/100``, ``r`` the return rate::

        (1−r)(Rq + fee) − (1−r)(Cq + F) − rF  ≥  m(Rq + fee)
        (Rq + fee)(1 − r − m)                 ≥  (1−r)(Cq + F) + rF
        R                                     ≥  [ RHS/(1−r−m) − fee ] / q

    Returns ``None`` when ``1 − r − m ≤ 0``: the margin floor is unreachable at
    *any* price for this product, so no offer can be valid.
    """
    q = Decimal(quantity)
    r = econ.historical_return_rate
    m = min_margin_pct / Decimal("100")

    denom = Decimal("1") - r - m
    if denom <= 0:
        return None

    numer = (Decimal("1") - r) * (econ.cost_of_goods * q + fulfilment) + r * fulfilment
    revenue_needed = numer / denom
    unit = (revenue_needed - fee) / q
    return unit.quantize(PAISE, rounding=ROUND_CEILING)


def compute_reserve_price(
    econ: ProductEconomics,
    quantity: int = 1,
    city_tier: int = 1,
) -> tuple[Decimal | None, str, dict]:
    """The lowest unit price the merchant may **autonomously** accept.

    Two reserves are computed, and the distinction matters:

    ``reserve_hard``
        The lowest price that satisfies every hard ceiling and the margin floor.
        Below this, no human can approve it either — it breaches policy.

    ``reserve_autonomous`` (the returned ``reserve``)
        ``reserve_hard`` raised further so the discount also stays under
        ``human_approval_required_above_discount``. This is what the agent
        quotes, so a counter-offer the buyer accepts is always immediately
        executable. Quoting a price that then needs a human would make the
        counter meaningless.

    A request landing *between* the two reserves is a genuine
    ``REQUIRE_APPROVAL`` — the buyer asked for something a human could sign off
    but the agent may not grant alone.

    Returns ``(reserve_autonomous, binding_constraint, detail)``. ``None`` means
    the margin floor is unreachable for this SKU at any price.
    """
    pol = _policy()
    q = Decimal(quantity)
    subtotal_at_list = econ.selling_price * q
    fee = shipping_fee_for(econ, subtotal_at_list)
    fulfilment = econ.fulfilment_cost(city_tier)

    floors: dict[str, Decimal] = {}

    # 1. percentage ceiling
    floors["RULE_AC_DISCOUNT_PERCENT_CEILING"] = (
        econ.selling_price * (Decimal("1") - pol["max_discount_percent"] / Decimal("100"))
    ).quantize(PAISE, rounding=ROUND_CEILING)

    # 2. absolute amount ceiling (applies to the whole order)
    floors["RULE_AC_DISCOUNT_AMOUNT_CEILING"] = (
        econ.selling_price - (pol["max_discount_amount"] / q)
    ).quantize(PAISE, rounding=ROUND_CEILING)

    # 3. margin floor
    margin_floor = _margin_floor_price(econ, quantity, fee, fulfilment, pol["min_margin_percent"])
    if margin_floor is None:
        return None, "RULE_AC_MARGIN_FLOOR", {
            "fee": fee, "fulfilment": fulfilment, "floors": floors,
            "reserve_hard": None, "hard_binding": "RULE_AC_MARGIN_FLOOR",
            "unreachable": True,
        }
    floors["RULE_AC_MARGIN_FLOOR"] = margin_floor

    hard_binding = max(floors, key=lambda k: floors[k])
    reserve_hard = floors[hard_binding].quantize(RUPEE, rounding=ROUND_CEILING)

    # The autonomous zone is tighter than the hard zone whenever the approval
    # threshold is below the discount ceiling.
    approval_floor = (
        econ.selling_price - (pol["approval_above_discount"] / q)
    ).quantize(PAISE, rounding=ROUND_CEILING)
    floors_auto = dict(floors)
    floors_auto["RULE_AC_HUMAN_APPROVAL_THRESHOLD"] = approval_floor

    binding = max(floors_auto, key=lambda k: floors_auto[k])

    # Deliberately NOT capped at the list price. A reserve above list means the
    # margin floor cannot be met even at full price — a thin-margin SKU in an
    # expensive shipping zone. Capping it at list would have quietly ACCEPTED a
    # sale that breaches the floor, which is exactly the failure this engine
    # exists to prevent. The caller checks `reserve > selling_price` and rejects.
    #
    # Quote whole rupees upward — never round a counter-offer *down* into a
    # policy breach.
    reserve = floors_auto[binding].quantize(RUPEE, rounding=ROUND_CEILING)
    return reserve, binding, {
        "fee": fee,
        "fulfilment": fulfilment,
        "floors": floors_auto,
        "reserve_hard": reserve_hard,
        "hard_binding": hard_binding,
    }


def evaluate_offer(
    econ: ProductEconomics,
    requested_unit_price: Decimal,
    quantity: int = 1,
    city_tier: int = 1,
    round_number: int = 1,
    buyer_max_budget: Decimal | None = None,
) -> OfferEvaluation:
    """Evaluate a buyer's requested price. Deterministic. Takes no free text.

    The signature is asserted by `tests/test_agentic_commerce.py` so that a
    future refactor cannot quietly add a `message` parameter and open a channel
    from buyer prose into merchant pricing.
    """
    pol = _policy()
    q = Decimal(quantity)
    requested = Decimal(requested_unit_price).quantize(PAISE)
    rules: list[NegotiationRule] = []

    def add(rule_id, passed, reason, inp=None, thr=None):
        rules.append(NegotiationRule(rule_id, passed, reason,
                                     None if inp is None else str(inp),
                                     None if thr is None else str(thr)))

    reserve, binding, detail = compute_reserve_price(econ, quantity, city_tier)
    fee: Decimal = detail["fee"]
    fulfilment: Decimal = detail["fulfilment"]

    def _finance(unit: Decimal) -> tuple[Decimal, Decimal, Decimal]:
        revenue = unit * q + fee
        gross = revenue - econ.cost_of_goods * q - fulfilment
        r = econ.historical_return_rate
        net = ((Decimal("1") - r) * gross - r * fulfilment).quantize(PAISE)
        margin = (net / revenue * Decimal("100")).quantize(PAISE) if revenue > 0 else Decimal("0")
        return revenue.quantize(PAISE), net, margin

    def build(decision: OfferDecision, reason_code: str, final_unit: Decimal) -> OfferEvaluation:
        revenue, net, margin = _finance(final_unit)
        disc_amt = ((econ.selling_price - final_unit) * q).quantize(PAISE)
        disc_pct = (
            (econ.selling_price - final_unit) / econ.selling_price * Decimal("100")
        ).quantize(PAISE) if econ.selling_price > 0 else Decimal("0")
        return OfferEvaluation(
            decision=decision,
            reason_code=reason_code,
            product_id=econ.product_id,
            quantity=quantity,
            list_price=econ.selling_price,
            requested_unit_price=requested,
            final_unit_price=final_unit,
            reserve_unit_price=reserve if reserve is not None else econ.selling_price,
            binding_constraint=binding,
            shipping_fee_charged=fee,
            fulfilment_cost=fulfilment,
            order_total=revenue,
            net_contribution=net,
            margin_percent=margin,
            discount_percent=max(disc_pct, Decimal("0")),
            discount_amount=max(disc_amt, Decimal("0")),
            rules=rules,
        )

    # --- rule 0: sanity ------------------------------------------------------
    valid_request = requested > 0 and quantity >= 1
    add("RULE_AC_REQUEST_WELL_FORMED", valid_request,
        "requested price and quantity are positive" if valid_request
        else "requested price or quantity is not a positive number",
        f"{requested} x{quantity}")
    if not valid_request:
        return build(OfferDecision.REJECT, "RULE_AC_REQUEST_WELL_FORMED", econ.selling_price)

    # --- rule 1: rounds ------------------------------------------------------
    within_rounds = round_number <= MAX_NEGOTIATION_ROUNDS
    add("RULE_AC_MAX_ROUNDS", within_rounds,
        "negotiation is within the permitted number of rounds" if within_rounds
        else "negotiation round limit reached; no further counter-offers",
        round_number, MAX_NEGOTIATION_ROUNDS)
    if not within_rounds:
        return build(OfferDecision.REJECT, "RULE_AC_MAX_ROUNDS", econ.selling_price)

    # --- rule 2: inventory ---------------------------------------------------
    have_stock = econ.inventory_level >= quantity
    add("RULE_AC_INVENTORY_AVAILABLE", have_stock,
        "sufficient inventory for the requested quantity" if have_stock
        else "requested quantity exceeds available inventory",
        quantity, econ.inventory_level)
    if not have_stock:
        return build(OfferDecision.REJECT, "RULE_AC_INVENTORY_AVAILABLE", econ.selling_price)

    # --- rule 3: margin floor reachable at all -------------------------------
    if reserve is None:
        add("RULE_AC_MARGIN_FLOOR", False,
            "the margin floor is unreachable for this SKU at any price",
            None, pol["min_margin_percent"])
        return build(OfferDecision.REJECT, "RULE_AC_MARGIN_FLOOR", econ.selling_price)

    # --- rule 3b: is ANY price within policy? -------------------------------
    reserve_hard: Decimal = detail["reserve_hard"]
    if reserve_hard > econ.selling_price:
        add("RULE_AC_MARGIN_FLOOR", False,
            "no price at or below list clears the margin floor for this SKU in "
            "this shipping zone; the merchant cannot sell it within policy",
            float(reserve), float(econ.selling_price))
        return build(OfferDecision.REJECT, "RULE_AC_NO_LAWFUL_PRICE", econ.selling_price)

    # A buyer offering at or above list gets list price. The merchant does not
    # take more than it asks for; overpayment is not upside, it is a refund
    # liability.
    final_unit = min(requested, econ.selling_price)
    clears_reserve = final_unit >= reserve
    clears_hard = final_unit >= reserve_hard

    # Each rule reports its OWN condition. Reporting the combined reserve here
    # produced "margin 23.90 / floor 15 — FAIL", which is simply false and
    # would discredit the whole guardrail panel. The margin floor can pass while
    # a discount ceiling is what actually binds.
    _, _, margin_at_request = _finance(final_unit)
    margin_ok = margin_at_request >= pol["min_margin_percent"]
    add("RULE_AC_MARGIN_FLOOR", margin_ok,
        "remaining contribution margin clears the floor" if clears_reserve
        else "price would push contribution margin below the merchant floor",
        round(margin_at_request, 2), pol["min_margin_percent"])

    disc_pct_req = ((econ.selling_price - final_unit) / econ.selling_price
                    * Decimal("100")) if econ.selling_price > 0 else Decimal("0")
    disc_amt_req = (econ.selling_price - final_unit) * q

    pct_ok = disc_pct_req <= pol["max_discount_percent"]
    add("RULE_AC_DISCOUNT_PERCENT_CEILING", pct_ok,
        "discount is within the autonomous percentage ceiling" if pct_ok
        else "discount exceeds the autonomous percentage ceiling",
        round(max(disc_pct_req, Decimal("0")), 2), pol["max_discount_percent"])

    amt_ok = disc_amt_req <= pol["max_discount_amount"]
    add("RULE_AC_DISCOUNT_AMOUNT_CEILING", amt_ok,
        "discount is within the autonomous absolute ceiling" if amt_ok
        else "discount exceeds the autonomous absolute ceiling",
        round(max(disc_amt_req, Decimal("0")), 2), pol["max_discount_amount"])

    if not clears_reserve and clears_hard:
        # Between the autonomous reserve and the hard reserve: lawful, but not
        # the agent's call to make.
        add("RULE_AC_HUMAN_APPROVAL_THRESHOLD", False,
            "requested discount exceeds the agent's autonomous authority; "
            "a human operator may still approve it",
            round(max(disc_amt_req, Decimal("0")), 2), pol["approval_above_discount"])
        return build(OfferDecision.REQUIRE_APPROVAL,
                     "RULE_AC_HUMAN_APPROVAL_THRESHOLD", final_unit)

    if not clears_reserve:
        # COUNTER at the autonomous reserve — the binding constraint's price, not a guess.
        # If the buyer's stated budget cannot even reach the reserve, countering
        # is noise: reject and say why.
        if buyer_max_budget is not None and reserve * q + fee > Decimal(buyer_max_budget):
            add("RULE_AC_COUNTER_WITHIN_BUYER_BUDGET", False,
                "the lowest lawful price exceeds the buyer's stated budget",
                float(reserve * q + fee), float(buyer_max_budget))
            return build(OfferDecision.REJECT, "RULE_AC_NO_ZONE_OF_AGREEMENT", reserve)
        return build(OfferDecision.COUNTER, binding, reserve)

    # --- rule 4: approval thresholds ----------------------------------------
    order_total = final_unit * q + fee
    needs_approval_disc = disc_amt_req > pol["approval_above_discount"]
    add("RULE_AC_HUMAN_APPROVAL_THRESHOLD", not needs_approval_disc,
        "discount is below the human-approval threshold" if not needs_approval_disc
        else "discount exceeds the human-approval threshold",
        round(max(disc_amt_req, Decimal("0")), 2), pol["approval_above_discount"])

    high_value = order_total > pol["high_value_threshold"]
    add("RULE_AC_HIGH_VALUE_ORDER", not high_value,
        "order is below the high-value approval threshold" if not high_value
        else "high-value order requires human approval",
        round(order_total, 2), pol["high_value_threshold"])

    if needs_approval_disc or high_value:
        return build(OfferDecision.REQUIRE_APPROVAL,
                     "RULE_AC_HUMAN_APPROVAL_THRESHOLD" if needs_approval_disc
                     else "RULE_AC_HIGH_VALUE_ORDER",
                     final_unit)

    return build(OfferDecision.ACCEPT, "ALL_RULES_PASSED", final_unit)