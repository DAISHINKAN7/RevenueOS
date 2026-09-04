"""Agentic commerce API (spec §41–§43, §85).

Mounted from `backend/app/api.py` with two added lines. Everything here is
additive — no existing route, model or service changes behaviour.

The endpoints, in the order the demo uses them::

    GET  /api/catalog                      machine-readable catalog
    GET  /api/catalog/{product_id}
    POST /api/agent-commerce/session       NL request -> constraints + matches
    POST /api/agent-commerce/search        structured constraints -> matches
    POST /api/agent-commerce/offer         requested price -> ACCEPT/COUNTER/REJECT
    POST /api/agent-commerce/respond       buyer agent reacts to the ruling
    POST /api/agent-commerce/checkout      agreement -> Opportunity (Track 03)
    GET  /api/agent-commerce/session/{id}  full negotiation transcript
    POST /api/agent-commerce/demo/run      scripted end-to-end run for the video

The checkout endpoint is the bridge the spec's §43 describes. It does not
reimplement payments. It creates a normal `Opportunity` in `DETECTED` with a
context built from the negotiated economics, and hands back its id. From that
point the negotiated cart is indistinguishable from any other opportunity: the
same predictor scores it, the same policy engine gates it, the same Razorpay
path executes it, and a failed payment drops into the same recovery loop.
"""

from __future__ import annotations

import uuid
from decimal import Decimal

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.agents.buyer import (
    BUYER_VERSION, BuyerConstraints, BuyerParseError, decide, parse_request,
)
from backend.app.db.commerce_models import NegotiationSession, NegotiationTurn
from backend.app.services import catalog as cat
from backend.app.services.negotiation import (
    MAX_NEGOTIATION_ROUNDS, NEGOTIATION_POLICY_VERSION, OfferDecision,
    evaluate_offer, shipping_fee_for,
)

router = APIRouter(tags=["agentic-commerce"])

# A negotiated cart that reaches Razorpay and fails is a payment failure, which
# is what makes the §43 bridge work: the recovery loop sees an ordinary
# PAYMENT_FAILURE and retry actions stay eligible.
OPPORTUNITY_TYPE = "PAYMENT_FAILURE"


# --------------------------------------------------------------------------- #
# session dependency
# --------------------------------------------------------------------------- #

def get_db():
    from backend.app.db.models import get_session_factory

    s = get_session_factory()()
    try:
        yield s
    finally:
        s.close()


def _llm_enabled() -> bool:
    """Whether a real LLM is configured. `settings` carries no LLM fields —
    AGENT_LLM_* lives on `LLMConfig` in agents/llm.py."""
    from backend.app.agents.buyer import llm_active

    return llm_active()


def _catalog_or_503(fn, *a, **kw):
    try:
        return fn(*a, **kw)
    except cat.CatalogUnavailable as e:
        raise HTTPException(503, detail={"error_code": "CATALOG_UNAVAILABLE",
                                         "message": str(e)})


# --------------------------------------------------------------------------- #
# DTOs
# --------------------------------------------------------------------------- #

class SessionStart(BaseModel):
    request: str = Field(min_length=3, max_length=600)
    use_llm: bool | None = None


class SearchRequest(BaseModel):
    query: str | None = None
    category: str | None = None
    max_budget: float | None = Field(default=None, gt=0)
    min_price: float | None = Field(default=None, ge=0)
    limit: int = Field(default=5, ge=1, le=20)


class OfferRequest(BaseModel):
    session_id: str | None = None
    product_id: str
    requested_price: float = Field(gt=0, le=1_000_000)
    quantity: int = Field(default=1, ge=1, le=10)
    city_tier: int = Field(default=1, ge=1, le=3)
    max_budget: float | None = Field(default=None, gt=0)


class RespondRequest(BaseModel):
    session_id: str


class CheckoutRequest(BaseModel):
    session_id: str
    execution_mode: str = Field(default="SIMULATOR")


class DemoRequest(BaseModel):
    request: str = "I need wireless headphones under Rs 6000 with good battery life"
    aggressive_discount_percent: float = Field(default=18.0, ge=0, le=90)


# --------------------------------------------------------------------------- #
# catalog
# --------------------------------------------------------------------------- #

@router.get("/api/catalog")
def get_catalog(
    limit: int = Query(60, ge=1, le=200),
    offset: int = Query(0, ge=0),
    category: str | None = None,
    max_price: float | None = None,
):
    """Machine-readable catalog. No COGS, ever.

    An AI buyer that could read cost of goods would compute the reserve price
    exactly and the negotiation would be theatre. `ProductEconomics` is a
    separate object that never crosses this boundary.
    """
    if category or max_price is not None:
        items = _catalog_or_503(cat.search, category=category, max_price=max_price,
                                in_stock_only=False, limit=limit)
    else:
        items = _catalog_or_503(cat.list_products, limit=limit, offset=offset)
    return {
        "catalog_version": cat.CATALOG_VERSION,
        "currency": "INR",
        "count": len(items),
        "categories": _catalog_or_503(cat.categories),
        "negotiable": True,
        "offer_endpoint": "/api/agent-commerce/offer",
        "products": [p.as_dict() for p in items],
    }


@router.get("/api/catalog/{product_id}")
def get_catalog_item(product_id: str):
    p = _catalog_or_503(cat.get_product, product_id)
    if p is None:
        raise HTTPException(404, detail={"error_code": "PRODUCT_NOT_FOUND"})
    return {"catalog_version": cat.CATALOG_VERSION, "product": p.as_dict()}


@router.post("/api/agent-commerce/search")
def search_catalog(body: SearchRequest):
    items = _catalog_or_503(
        cat.search, query=body.query, category=body.category,
        max_price=body.max_budget, min_price=body.min_price, limit=body.limit)
    return {"count": len(items), "products": [p.as_dict() for p in items]}


# --------------------------------------------------------------------------- #
# negotiation
# --------------------------------------------------------------------------- #

@router.post("/api/agent-commerce/session")
def start_session(body: SessionStart, s: Session = Depends(get_db)):
    """Natural-language buyer request → parsed constraints → candidate products."""
    use_llm = _llm_enabled() if body.use_llm is None else body.use_llm
    try:
        constraints, source = parse_request(body.request, use_llm=use_llm)
    except BuyerParseError as e:
        raise HTTPException(400, detail={"error_code": "BUYER_PARSE_FAILED",
                                         "message": str(e)})

    matches = _catalog_or_503(
        cat.search, query=constraints.query, category=constraints.category,
        max_price=constraints.max_budget, limit=5)

    sess = NegotiationSession(
        id=f"neg_{uuid.uuid4().hex[:16]}",
        buyer_request_text=body.request,
        constraints=constraints.as_dict(),
        constraints_source=source,
        quantity=constraints.quantity,
        city_tier=constraints.city_tier,
        policy_version=NEGOTIATION_POLICY_VERSION,
        catalog_version=cat.CATALOG_VERSION,
    )
    s.add(sess)
    s.commit()

    return {
        "session_id": sess.id,
        "buyer_version": BUYER_VERSION,
        "constraints": constraints.as_dict(),
        "constraints_source": source,
        "max_rounds": MAX_NEGOTIATION_ROUNDS,
        "matches": [p.as_dict() for p in matches],
    }


def _record_turn(s: Session, sess: NegotiationSession, ev, buyer_price: Decimal,
                 message: str, message_source: str) -> NegotiationTurn:
    turn = NegotiationTurn(
        session_id=sess.id,
        round_number=sess.rounds,
        actor="MERCHANT",
        requested_unit_price=float(buyer_price),
        decision=ev.decision.value,
        reason_code=ev.reason_code,
        binding_constraint=ev.binding_constraint,
        offered_unit_price=float(ev.final_unit_price),
        margin_percent=float(ev.margin_percent),
        net_contribution=float(ev.net_contribution),
        rules=[r.as_dict() for r in ev.rules],
        message=message,
        message_source=message_source,
    )
    s.add(turn)
    return turn


_MERCHANT_TEMPLATES = {
    "ACCEPT": "Agreed at ₹{unit} per unit. That clears our margin floor at {margin}%.",
    "COUNTER": "₹{requested} is below what we can authorise. ₹{unit} is our floor "
               "— it is set by {binding}, not by preference.",
    "REJECT": "We cannot fill this one. {binding} does not permit any price we could offer.",
    "REQUIRE_APPROVAL": "₹{unit} is lawful but beyond what I may approve alone. "
                        "Routing to a human operator.",
}


def _merchant_message(ev) -> str:
    return _MERCHANT_TEMPLATES[ev.decision.value].format(
        unit=f"{ev.final_unit_price:,.0f}",
        requested=f"{ev.requested_unit_price:,.0f}",
        margin=f"{ev.margin_percent:.1f}",
        binding=ev.binding_constraint.replace("RULE_AC_", "").replace("_", " ").lower(),
    )


@router.post("/api/agent-commerce/offer")
def make_offer(body: OfferRequest, s: Session = Depends(get_db)):
    """The offer protocol. Deterministic pricing; the LLM never sees this path.

    `evaluate_offer` takes no text parameter, so a buyer's prose has no channel
    into the arithmetic. The counter-price returned is the merchant's reserve —
    computed from COGS, fulfilment cost, return rate and merchant policy — not a
    negotiated guess and not a model output.
    """
    econ = _catalog_or_503(cat.get_economics, body.product_id)
    if econ is None:
        raise HTTPException(404, detail={"error_code": "PRODUCT_NOT_FOUND"})

    sess: NegotiationSession | None = None
    if body.session_id:
        sess = s.get(NegotiationSession, body.session_id)
        if sess is None:
            raise HTTPException(404, detail={"error_code": "SESSION_NOT_FOUND"})
        if sess.status not in ("OPEN",):
            raise HTTPException(409, detail={
                "error_code": "SESSION_CLOSED",
                "message": f"session is {sess.status}; start a new one"})

    round_number = (sess.rounds + 1) if sess else 1

    budget = body.max_budget
    if budget is None and sess:
        budget = (sess.constraints or {}).get("max_budget")

    ev = evaluate_offer(
        econ,
        requested_unit_price=Decimal(str(body.requested_price)),
        quantity=body.quantity,
        city_tier=body.city_tier,
        round_number=round_number,
        buyer_max_budget=Decimal(str(budget)) if budget else None,
    )

    payload = ev.as_dict()
    payload["message"] = _merchant_message(ev)
    payload["round"] = round_number
    payload["rounds_remaining"] = max(0, MAX_NEGOTIATION_ROUNDS - round_number)

    if sess:
        sess.rounds = round_number
        sess.product_id = body.product_id
        sess.quantity = body.quantity
        sess.city_tier = body.city_tier
        _record_turn(s, sess, ev, Decimal(str(body.requested_price)),
                     payload["message"], "TEMPLATE")

        if ev.decision is OfferDecision.ACCEPT:
            sess.status = "AGREED"
            sess.agreed_unit_price = float(ev.final_unit_price)
            sess.agreed_order_total = float(ev.order_total)
            sess.agreed_contribution = float(ev.net_contribution)
            sess.agreed_margin_percent = float(ev.margin_percent)
        elif ev.decision is OfferDecision.REQUIRE_APPROVAL:
            sess.status = "AWAITING_APPROVAL"
        elif ev.decision is OfferDecision.REJECT:
            sess.status = "REJECTED"
        s.commit()
        payload["session_id"] = sess.id
        payload["session_status"] = sess.status

    return payload


@router.post("/api/agent-commerce/respond")
def buyer_responds(body: RespondRequest, s: Session = Depends(get_db)):
    """The buyer agent reacts to the merchant's last ruling.

    Deterministic: accept iff the order total fits the budget declared at the
    start. The model phrases the reply; it does not choose it.
    """
    sess = s.get(NegotiationSession, body.session_id)
    if sess is None:
        raise HTTPException(404, detail={"error_code": "SESSION_NOT_FOUND"})

    last = s.execute(
        select(NegotiationTurn)
        .where(NegotiationTurn.session_id == sess.id)
        .order_by(NegotiationTurn.id.desc())
    ).scalars().first()
    if last is None:
        raise HTTPException(409, detail={"error_code": "NO_OFFER_YET"})

    econ = _catalog_or_503(cat.get_economics, sess.product_id)
    if econ is None:
        raise HTTPException(409, detail={"error_code": "PRODUCT_NOT_FOUND"})

    constraints = BuyerConstraints(**{
        k: v for k, v in (sess.constraints or {}).items()
        if k in BuyerConstraints.__dataclass_fields__
    })
    unit = Decimal(str(last.offered_unit_price))
    subtotal = unit * Decimal(sess.quantity)
    total = subtotal + shipping_fee_for(econ, subtotal)

    resp = decide(
        constraints,
        decision=last.decision or "REJECT",
        offered_unit_price=unit,
        order_total=total,
        round_number=last.round_number,
        list_price=econ.selling_price,
        use_llm=_llm_enabled(),
    )

    s.add(NegotiationTurn(
        session_id=sess.id, round_number=last.round_number, actor="BUYER",
        requested_unit_price=resp.counter_unit_price,
        decision=resp.action, message=resp.message,
        message_source=resp.message_source, rules=[],
    ))

    if resp.action == "WALK_AWAY":
        sess.status = "ABANDONED"
    elif resp.action == "ACCEPT" and last.decision == "ACCEPT":
        sess.status = "AGREED"
    s.commit()

    return {
        "session_id": sess.id,
        "buyer_action": resp.action,
        "counter_unit_price": resp.counter_unit_price,
        "message": resp.message,
        "message_source": resp.message_source,
        "session_status": sess.status,
    }


@router.get("/api/agent-commerce/session/{session_id}")
def get_session(session_id: str, s: Session = Depends(get_db)):
    sess = s.get(NegotiationSession, session_id)
    if sess is None:
        raise HTTPException(404, detail={"error_code": "SESSION_NOT_FOUND"})
    turns = s.execute(
        select(NegotiationTurn).where(NegotiationTurn.session_id == sess.id)
        .order_by(NegotiationTurn.id)
    ).scalars().all()
    return {
        "session_id": sess.id,
        "status": sess.status,
        "buyer_request": sess.buyer_request_text,
        "constraints": sess.constraints,
        "constraints_source": sess.constraints_source,
        "product_id": sess.product_id,
        "quantity": sess.quantity,
        "rounds": sess.rounds,
        "policy_version": sess.policy_version,
        "catalog_version": sess.catalog_version,
        "opportunity_id": sess.opportunity_id,
        "agreed": {
            "unit_price": float(sess.agreed_unit_price) if sess.agreed_unit_price else None,
            "order_total": float(sess.agreed_order_total) if sess.agreed_order_total else None,
            "contribution": float(sess.agreed_contribution) if sess.agreed_contribution else None,
            "margin_percent": float(sess.agreed_margin_percent) if sess.agreed_margin_percent else None,
        },
        "turns": [{
            "round": t.round_number,
            "actor": t.actor,
            "requested_unit_price": float(t.requested_unit_price) if t.requested_unit_price else None,
            "decision": t.decision,
            "reason_code": t.reason_code,
            "binding_constraint": t.binding_constraint,
            "offered_unit_price": float(t.offered_unit_price) if t.offered_unit_price else None,
            "margin_percent": float(t.margin_percent) if t.margin_percent else None,
            "net_contribution": float(t.net_contribution) if t.net_contribution else None,
            "message": t.message,
            "message_source": t.message_source,
            "rules": t.rules,
            "created_at": t.created_at.isoformat() if t.created_at else None,
        } for t in turns],
    }


# --------------------------------------------------------------------------- #
# the bridge into Track 03
# --------------------------------------------------------------------------- #

def _negotiated_context(sess: NegotiationSession, econ) -> dict:
    """Build a full recovery context from the negotiated cart.

    Every field the feature builder expects is present. A partial context would
    make the predictor fail closed — correct behaviour, but it would mean an
    agentic-commerce cart could never be analysed, which defeats the bridge.
    Values are derived from the negotiation where they exist and set to
    neutral, honest defaults (a first-time buyer, no history) where they do not.
    A negotiated cart genuinely has no purchase history.
    """
    unit = Decimal(str(sess.agreed_unit_price))
    q = Decimal(sess.quantity)
    cart_value = float(sess.agreed_order_total)
    cogs = float(econ.cost_of_goods * q)
    fulfilment = float(econ.fulfilment_cost(sess.city_tier))
    fee = float(shipping_fee_for(econ, unit * q))
    margin = float(sess.agreed_contribution)

    return {
        "cart_value": cart_value, "cart_cogs": cogs,
        "shipping_cost": fulfilment, "shipping_fee_charged": fee,
        "base_contribution_margin": margin,
        "base_margin_pct": float(sess.agreed_margin_percent),
        "product_return_rate": float(econ.historical_return_rate),
        # Categorical values are drawn from the simulator's own vocabularies.
        # An unseen category here would be encoded as an unknown level and the
        # model's estimate would be quietly meaningless, so these are checked
        # against data/generated/opportunities.parquet by a test.
        "opportunity_type": OPPORTUNITY_TYPE,
        "failure_reason": "CARD_DECLINED",
        "payment_method": "CARD",
        "attempt_number": 1, "abandonment_stage": "PAYMENT_PROCESSING",
        "device_type": "DESKTOP", "traffic_source": "DIRECT",
        "network_context": "WIFI",
        "hour_of_day": 14.0, "day_of_week": 2.0,
        "coupon_attempted": 0.0, "minutes_since_event": 2.0,
        "log_minutes_since_event": 1.0986,
        "hour_sin": 0.5, "hour_cos": -0.866, "dow_sin": 0.7818, "dow_cos": 0.6235,
        "is_weekend": 0.0,
        "customer_segment": "NEW_CUSTOMER", "city_tier": float(sess.city_tier),
        "tenure_days": 0.0, "orders_lifetime": 0.0,
        "orders_last_30d": 0.0, "orders_last_90d": 0.0,
        "average_order_value": cart_value, "lifetime_value": 0.0,
        "days_since_last_purchase": 0.0,
        "previous_checkout_abandonments": 0.0, "previous_payment_failures": 0.0,
        "historical_return_rate": float(econ.historical_return_rate),
        "historical_cancellation_rate": 0.0,
        "coupon_offers_seen": 0.0, "coupon_offers_redeemed": 0.0,
        "coupon_rate_smoothed": 0.1,
        "free_shipping_offers_seen": 0.0, "free_shipping_offers_redeemed": 0.0,
        "free_shipping_rate_smoothed": 0.1,
        "retry_offers_seen": 0.0, "retry_offers_succeeded": 0.0,
        "retry_rate_smoothed": 0.1,
        "coupon_history_missing": 1.0, "shipping_history_missing": 1.0,
        "retry_history_missing": 1.0,
        "shipping_fee_to_cart_ratio": fee / cart_value if cart_value else 0.0,
        "shipping_cost_to_margin_ratio": fulfilment / margin if margin else 0.0,
        # Provenance — not a model feature, but it belongs in the record.
        "origin": "AGENTIC_COMMERCE",
        "negotiation_session_id": sess.id,
    }


@router.post("/api/agent-commerce/checkout")
def checkout(body: CheckoutRequest, s: Session = Depends(get_db)):
    """Turn an agreed negotiation into a normal recovery Opportunity.

    This is the §43 bridge and it is deliberately thin. No payment logic is
    duplicated here. The negotiated cart becomes an ordinary `Opportunity`, and
    from that point every existing endpoint applies: analyse it, gate it,
    execute it through Razorpay Test Mode, and if the payment fails the same
    recovery loop takes over with the negotiated margin as its constraint.
    """
    from backend.app.core.config import WORKFLOW_VERSION
    from backend.app.db.models import Opportunity, utcnow
    from backend.app.domain import State
    from backend.app.services.workflow import AuditRecorder, new_trace_id

    sess = s.get(NegotiationSession, body.session_id)
    if sess is None:
        raise HTTPException(404, detail={"error_code": "SESSION_NOT_FOUND"})
    if sess.status != "AGREED" or sess.agreed_unit_price is None:
        raise HTTPException(409, detail={
            "error_code": "NEGOTIATION_NOT_AGREED",
            "message": f"session status is {sess.status}; only AGREED can check out"})
    if sess.opportunity_id:
        return {"opportunity_id": sess.opportunity_id, "created": False,
                "session_id": sess.id,
                "message": "checkout already created for this negotiation"}
    if body.execution_mode not in ("SIMULATOR", "RAZORPAY_TEST"):
        raise HTTPException(400, detail={"error_code": "INVALID_EXECUTION_MODE"})

    econ = _catalog_or_503(cat.get_economics, sess.product_id)
    if econ is None:
        raise HTTPException(409, detail={"error_code": "PRODUCT_NOT_FOUND"})

    ctx = _negotiated_context(sess, econ)
    opp = Opportunity(
        id=f"OPP-AGENT-{uuid.uuid4().hex[:6]}",
        source_checkout_id=sess.id,
        customer_id=f"AGENT-{sess.id[-8:]}",
        opportunity_type=OPPORTUNITY_TYPE,
        detected_at=utcnow(),
        state=State.DETECTED.value,
        workflow_version=WORKFLOW_VERSION,
        execution_mode=body.execution_mode,
        # These four are NOT NULL on the table and are the columns the
        # dashboard aggregates over. Deriving them from the negotiation is the
        # whole point: the money at risk is the money the agent just agreed to.
        revenue_at_risk=round(float(sess.agreed_order_total), 2),
        contribution_margin_at_risk=round(float(sess.agreed_contribution), 2),
        current_attempt=1,
        trace_id=new_trace_id(),
        context=ctx,
    )
    s.add(opp)
    s.flush()

    AuditRecorder(s, opp).record(
        "OPPORTUNITY_DETECTED",
        "opportunity created from an agentic-commerce negotiation",
        {
            "origin": "AGENTIC_COMMERCE",
            "negotiation_session_id": sess.id,
            "product_id": sess.product_id,
            "quantity": sess.quantity,
            "list_price": float(econ.selling_price),
            "negotiated_unit_price": float(sess.agreed_unit_price),
            "order_total": float(sess.agreed_order_total),
            "negotiated_margin_percent": float(sess.agreed_margin_percent),
            "negotiation_rounds": sess.rounds,
            "negotiation_policy_version": sess.policy_version,
        },
    )

    sess.opportunity_id = opp.id
    sess.status = "CHECKED_OUT"
    s.commit()

    return {
        "created": True,
        "session_id": sess.id,
        "opportunity_id": opp.id,
        "state": opp.state,
        "execution_mode": opp.execution_mode,
        "order_total": float(sess.agreed_order_total),
        "negotiated_margin_percent": float(sess.agreed_margin_percent),
        "next": {
            "analyze": f"/api/opportunities/{opp.id}/analyze",
            "detail": f"/api/opportunities/{opp.id}",
        },
        "message": "negotiated cart is now an ordinary recovery opportunity",
    }


# --------------------------------------------------------------------------- #
# scripted demo
# --------------------------------------------------------------------------- #

@router.post("/api/agent-commerce/demo/run")
def demo_run(body: DemoRequest, s: Session = Depends(get_db)):
    """One call, the whole §43 flow, for the demo video.

    Every step calls the same functions the individual endpoints call. Nothing
    here is staged — the counter-price is computed, not scripted, and if the
    catalog changes the numbers change with it.
    """
    use_llm = _llm_enabled()
    constraints, source = parse_request(body.request, use_llm=use_llm)
    matches = _catalog_or_503(
        cat.search, query=constraints.query, category=constraints.category,
        max_price=constraints.max_budget, limit=5)
    if not matches:
        raise HTTPException(409, detail={
            "error_code": "NO_MATCHING_PRODUCTS",
            "message": "no catalog product matches the parsed constraints"})

    product = matches[0]
    econ = cat.get_economics(product.product_id)
    steps: list[dict] = [
        {"step": "buyer_request", "text": body.request},
        {"step": "constraints", "value": constraints.as_dict(), "source": source},
        {"step": "catalog_search", "matched": [p.product_id for p in matches],
         "selected": product.product_id, "list_price": product.list_price},
    ]

    sess = NegotiationSession(
        id=f"neg_{uuid.uuid4().hex[:16]}",
        buyer_request_text=body.request, constraints=constraints.as_dict(),
        constraints_source=source, product_id=product.product_id,
        quantity=constraints.quantity, city_tier=constraints.city_tier,
        policy_version=NEGOTIATION_POLICY_VERSION,
        catalog_version=cat.CATALOG_VERSION,
    )
    s.add(sess)
    s.flush()

    # Round 1 — the buyer opens low on purpose, so the gate has to bite.
    opening = (econ.selling_price
               * (Decimal("1") - Decimal(str(body.aggressive_discount_percent)) / 100)
               ).quantize(Decimal("1"))
    sess.rounds = 1
    ev1 = evaluate_offer(econ, opening, constraints.quantity, constraints.city_tier,
                         round_number=1)
    _record_turn(s, sess, ev1, opening, _merchant_message(ev1), "TEMPLATE")
    steps.append({"step": "offer_round_1", "requested": float(opening),
                  **ev1.as_dict(), "message": _merchant_message(ev1)})

    result = {"steps": steps, "session_id": sess.id}

    if ev1.decision is not OfferDecision.COUNTER:
        sess.status = ("AGREED" if ev1.decision is OfferDecision.ACCEPT
                       else "AWAITING_APPROVAL" if ev1.decision is OfferDecision.REQUIRE_APPROVAL
                       else "REJECTED")
        if ev1.decision is OfferDecision.ACCEPT:
            sess.agreed_unit_price = float(ev1.final_unit_price)
            sess.agreed_order_total = float(ev1.order_total)
            sess.agreed_contribution = float(ev1.net_contribution)
            sess.agreed_margin_percent = float(ev1.margin_percent)
        s.commit()
        result["outcome"] = sess.status
        return result

    # Round 2 — buyer accepts the bounded counter.
    sess.rounds = 2
    ev2 = evaluate_offer(econ, ev1.final_unit_price, constraints.quantity,
                         constraints.city_tier, round_number=2)
    _record_turn(s, sess, ev2, ev1.final_unit_price, _merchant_message(ev2), "TEMPLATE")
    steps.append({"step": "offer_round_2", "requested": float(ev1.final_unit_price),
                  **ev2.as_dict(), "message": _merchant_message(ev2)})

    if ev2.decision is OfferDecision.ACCEPT:
        sess.status = "AGREED"
        sess.agreed_unit_price = float(ev2.final_unit_price)
        sess.agreed_order_total = float(ev2.order_total)
        sess.agreed_contribution = float(ev2.net_contribution)
        sess.agreed_margin_percent = float(ev2.margin_percent)
        s.commit()
        bridge = checkout(CheckoutRequest(session_id=sess.id), s)
        steps.append({"step": "checkout_bridge", **bridge})
        result["outcome"] = "AGREED"
        result["opportunity_id"] = bridge.get("opportunity_id")
    else:
        sess.status = "AWAITING_APPROVAL" if ev2.decision is OfferDecision.REQUIRE_APPROVAL \
            else "REJECTED"
        s.commit()
        result["outcome"] = sess.status

    return result