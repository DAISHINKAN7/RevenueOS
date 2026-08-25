"""RevenueOS API.

Read endpoints are open; anything that can move money requires an admin token.
The webhook endpoint reads the **raw** request body before any JSON parsing,
because signature verification is computed over exact bytes.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from decimal import Decimal

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import func, select

from backend.app.core.config import (
    APPLICATION_VERSION, FINANCIAL_ENGINE_VERSION, MerchantPolicy,
    WORKFLOW_VERSION, settings,
)
from backend.app.db.models import (
    ActionFinancialEvaluation, AuditEvent, Opportunity, PolicyEvaluation,
    RecoveryExecution, RecoveryOutcome, SCHEMA_VERSION, get_session_factory, init_db,
)
from backend.app.domain import RevenueOSError, State
from backend.app.services.predictor import get_predictor
from backend.app.services.razorpay import (
    RazorpayRecoveryExecutor, WebhookReconciler, get_razorpay_client,
)
from backend.app.services.workflow import RecoveryWorkflow, SimulatorRecoveryExecutor

logging.basicConfig(
    level=logging.INFO,
    format='{"ts":"%(asctime)s","level":"%(levelname)s","component":"%(name)s","event":"%(message)s"}',
)
log = logging.getLogger("revenueos.api")

app = FastAPI(title="RevenueOS", version=APPLICATION_VERSION,
              description="Autonomous Revenue Recovery for Intelligent Commerce")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_url],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def db():
    s = get_session_factory()()
    try:
        yield s
    finally:
        s.close()


def require_admin(x_admin_token: str = Header(default="")) -> str:
    """Write endpoints must never be anonymous on a public deployment."""
    if x_admin_token != settings.admin_token:
        raise HTTPException(status_code=401, detail={"error_code": "UNAUTHORIZED"})
    return x_admin_token


@app.exception_handler(RevenueOSError)
async def revenueos_error_handler(request: Request, exc: RevenueOSError):
    from fastapi.responses import JSONResponse
    return JSONResponse(status_code=exc.http_status, content=exc.to_dict())


from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield

app = FastAPI(title="RevenueOS", version=APPLICATION_VERSION, lifespan=lifespan,
              description="Autonomous Revenue Recovery for Intelligent Commerce")


# ------------------------------------------------------------------- health
@app.get("/health")
def health(s=Depends(db)):
    p = get_predictor()
    try:
        s.execute(select(func.count()).select_from(Opportunity))
        db_status = "ok"
    except Exception:  # noqa: BLE001
        db_status = "error"
    h = p.health()
    # Read the policy that is actually in force, not the module default. The
    # merchant policy is a file, so reporting the constant would let /health
    # disagree with what the policy engine is enforcing.
    active_policy = MerchantPolicy.load()
    return {
        "backend_status": "ok" if (db_status == "ok" and h["model_loaded"]) else "degraded",
        "database_status": db_status,
        "autonomous_execution_enabled": h["model_loaded"],
        **h,
        "policy_version": active_policy.policy_version,
        **settings.safe_dict(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/version")
def version():
    p = get_predictor()
    active_policy = MerchantPolicy.load()
    return {
        "application_version": APPLICATION_VERSION,
        "workflow_version": WORKFLOW_VERSION,
        "model_version": p.model_version,
        "feature_pipeline_version": p.feature_pipeline_version,
        "financial_engine_version": FINANCIAL_ENGINE_VERSION,
        "policy_version": active_policy.policy_version,
        "schema_version": SCHEMA_VERSION,
        "payment_environment": "TEST",
    }


# ------------------------------------------------------------- opportunities
@app.get("/api/opportunities")
def list_opportunities(state: str | None = None, limit: int = 50, s=Depends(db)):
    q = select(Opportunity).order_by(Opportunity.detected_at.desc()).limit(limit)
    if state:
        q = q.where(Opportunity.state == state)
    rows = s.execute(q).scalars().all()
    return [{
        "opportunity_id": o.id, "opportunity_type": o.opportunity_type,
        "state": o.state, "revenue_at_risk": float(o.revenue_at_risk),
        "contribution_margin_at_risk": float(o.contribution_margin_at_risk),
        "selected_action": o.selected_action, "attempt": o.current_attempt,
        "execution_mode": o.execution_mode,
        "detected_at": o.detected_at.isoformat(),
        "customer_segment": o.context.get("customer_segment"),
    } for o in rows]


@app.get("/api/opportunities/{opportunity_id}")
def opportunity_detail(opportunity_id: str, s=Depends(db)):
    o = s.get(Opportunity, opportunity_id)
    if o is None:
        raise HTTPException(404, detail={"error_code": "NOT_FOUND"})

    evals = s.execute(
        select(ActionFinancialEvaluation)
        .where(ActionFinancialEvaluation.opportunity_id == o.id,
               ActionFinancialEvaluation.attempt_number == o.current_attempt)
        .order_by(ActionFinancialEvaluation.rank)).scalars().all()
    pol = s.execute(
        select(PolicyEvaluation).where(PolicyEvaluation.opportunity_id == o.id)
        .order_by(PolicyEvaluation.id.desc())).scalars().first()
    execs = s.execute(
        select(RecoveryExecution).where(RecoveryExecution.opportunity_id == o.id)
    ).scalars().all()
    outcome = s.get(RecoveryOutcome, o.id)
    ctx = dict(o.context)

    return {
        "opportunity": {
            "opportunity_id": o.id, "type": o.opportunity_type, "state": o.state,
            "revenue_at_risk": float(o.revenue_at_risk),
            "contribution_margin_at_risk": float(o.contribution_margin_at_risk),
            "attempt": o.current_attempt, "execution_mode": o.execution_mode,
            "trace_id": o.trace_id, "detected_at": o.detected_at.isoformat(),
        },
        "customer_summary": {
            "customer_id": o.customer_id,
            "segment": ctx.get("customer_segment"),
            "orders_lifetime": ctx.get("orders_lifetime"),
            "average_order_value": ctx.get("average_order_value"),
        },
        "checkout_summary": {
            "cart_value": ctx.get("cart_value"),
            "base_contribution_margin": ctx.get("base_contribution_margin"),
            "shipping_fee_charged": ctx.get("shipping_fee_charged"),
            "shipping_cost": ctx.get("shipping_cost"),
            "minutes_since_event": ctx.get("minutes_since_event"),
        },
        "payment_summary": {
            "failure_reason": ctx.get("failure_reason"),
            "payment_method": ctx.get("payment_method"),
        },
        "candidate_actions": [{
            "action": e.action, "rank": e.rank,
            "probability": float(e.recovery_probability),
            "incentive_cost": float(e.incentive_cost_if_recovered),
            "fixed_cost": float(e.fixed_action_cost),
            "expected_value": float(e.expected_value),
            "incremental_expected_value": float(e.incremental_expected_value),
        } for e in evals],
        "selected_action": o.selected_action,
        "policy_decision": ({
            "decision": pol.decision, "reason_code": pol.reason_code,
            "policy_version": pol.policy_version,
            "maximum_authorized_downside": float(pol.maximum_authorized_downside),
            "rules": [{"rule_id": r.rule_id, "passed": r.passed,
                       "decision": r.decision, "reason": r.reason,
                       "input": r.input_value, "threshold": r.threshold}
                      for r in pol.rules],
        } if pol else None),
        "execution": [{
            "execution_id": e.execution_id, "action": e.action, "status": e.status,
            "attempt": e.attempt_number, "provider": e.execution_provider,
            "order_id": e.external_order_id, "payment_id": e.external_payment_id,
            "amount": float(e.amount) if e.amount is not None else None,
            "error_code": e.error_code,
        } for e in execs],
        "outcome": ({
            "net_recovered_gmv": float(outcome.net_recovered_gmv),
            "realized_contribution": float(outcome.realized_contribution),
            "discount_amount": float(outcome.discount_amount),
            "recovered_at": outcome.recovery_timestamp.isoformat(),
        } if outcome else None),
        "audit_timeline": _audit(s, o.id),
    }


def _audit(s, opportunity_id: str) -> list[dict]:
    rows = s.execute(
        select(AuditEvent).where(AuditEvent.opportunity_id == opportunity_id)
        .order_by(AuditEvent.sequence_number)).scalars().all()
    return [{
        "sequence": a.sequence_number, "timestamp": a.timestamp.isoformat(),
        "event_type": a.event_type, "summary": a.summary,
        "state_before": a.workflow_state_before, "state_after": a.workflow_state_after,
        "actor": a.actor_type, "payload": a.structured_payload,
        "execution_id": a.execution_id,
    } for a in rows]


@app.get("/api/opportunities/{opportunity_id}/audit")
def audit_timeline(opportunity_id: str, s=Depends(db)):
    return {"opportunity_id": opportunity_id, "events": _audit(s, opportunity_id)}


# ------------------------------------------------------------------- actions
@app.post("/api/opportunities/{opportunity_id}/analyze")
def analyze(opportunity_id: str, s=Depends(db), _=Depends(require_admin)):
    o = s.get(Opportunity, opportunity_id)
    if o is None:
        raise HTTPException(404, detail={"error_code": "NOT_FOUND"})
    result = RecoveryWorkflow(s).analyze(o)
    s.commit()
    return result


class ApprovalBody(BaseModel):
    actor_id: str = "demo-operator"


@app.post("/api/opportunities/{opportunity_id}/approve")
def approve(opportunity_id: str, body: ApprovalBody, s=Depends(db), _=Depends(require_admin)):
    o = s.get(Opportunity, opportunity_id)
    if o is None:
        raise HTTPException(404, detail={"error_code": "NOT_FOUND"})
    r = RecoveryWorkflow(s).approve(o, body.actor_id)
    s.commit()
    return r


@app.post("/api/opportunities/{opportunity_id}/reject")
def reject(opportunity_id: str, body: ApprovalBody, s=Depends(db), _=Depends(require_admin)):
    o = s.get(Opportunity, opportunity_id)
    if o is None:
        raise HTTPException(404, detail={"error_code": "NOT_FOUND"})
    r = RecoveryWorkflow(s).reject(o, body.actor_id)
    s.commit()
    return r


@app.post("/api/opportunities/{opportunity_id}/execute")
def execute(opportunity_id: str, s=Depends(db), _=Depends(require_admin)):
    o = s.get(Opportunity, opportunity_id)
    if o is None:
        raise HTTPException(404, detail={"error_code": "NOT_FOUND"})
    executor = (RazorpayRecoveryExecutor() if o.execution_mode == "RAZORPAY_TEST"
                else SimulatorRecoveryExecutor())
    return RecoveryWorkflow(s).execute(o, executor)


# ------------------------------------------------------------------ webhooks
@app.post("/api/webhooks/razorpay")
async def razorpay_webhook(request: Request, response: Response, s=Depends(db)):
    """Verify signature over raw bytes, persist, acknowledge fast.

    Razorpay expects a 2xx within 5 seconds and retries with exponential
    backoff for 24 hours, so no slow work happens before the response.
    """
    raw = await request.body()  # raw bytes; must not be parsed first
    signature = request.headers.get("x-razorpay-signature", "")
    event_id = request.headers.get("x-razorpay-event-id", "")

    reconciler = WebhookReconciler(s)
    try:
        result = reconciler.receive(raw, signature, event_id)
    except RevenueOSError as exc:
        # Invalid signature: no state mutation, nothing trusted persisted.
        return Response(content='{"status":"rejected"}', status_code=exc.http_status,
                        media_type="application/json")

    if result["status"] == "duplicate":
        return {"status": "duplicate", "event_id": result["event_id"]}

    processed = reconciler.process(result["inbox_id"])
    return {"status": "accepted", "event_id": result["event_id"], "result": processed}


# ------------------------------------------------------------------- metrics
@app.get("/api/dashboard/metrics")
def dashboard_metrics(s=Depends(db)):
    """Operational metrics from live/seeded executions only.

    Deliberately separate from the frozen research evaluation numbers in
    `evaluation/results/` — those are a different class of evidence and mixing
    them would be misleading.
    """
    opps = s.execute(select(Opportunity)).scalars().all()
    outcomes = s.execute(select(RecoveryOutcome)).scalars().all()
    pols = s.execute(select(PolicyEvaluation)).scalars().all()

    at_risk = sum(float(o.revenue_at_risk) for o in opps)
    margin_at_risk = sum(float(o.contribution_margin_at_risk) for o in opps)
    recovered_gmv = sum(float(o.net_recovered_gmv) for o in outcomes)
    contribution = sum(float(o.realized_contribution) for o in outcomes)
    incentives = sum(float(o.discount_amount) + float(o.shipping_subsidy) for o in outcomes)
    resolved = [o for o in opps if o.state in
                (State.RECOVERED.value, State.NOT_RECOVERED.value, State.STOPPED.value)]

    return {
        "metric_class": "LIVE_OPERATIONAL",
        "note": "Seeded and Test Mode executions only. Not the frozen research evaluation.",
        "opportunities": len(opps),
        "revenue_at_risk": round(at_risk, 2),
        "contribution_margin_at_risk": round(margin_at_risk, 2),
        "recovered_gmv": round(recovered_gmv, 2),
        "net_contribution_recovered": round(contribution, 2),
        "intervention_cost": round(incentives, 2),
        "recovery_rate": round(len(outcomes) / len(resolved), 4) if resolved else 0.0,
        "number_of_policy_blocks": sum(1 for p in pols if p.decision == "REJECT"),
        "number_of_stops": sum(1 for p in pols if p.decision == "STOP"),
        "number_of_approval_cases": sum(1 for p in pols if p.decision == "REQUIRE_APPROVAL"),
        "number_of_do_nothing_decisions": sum(
            1 for o in opps if o.selected_action == "DO_NOTHING"),
    }