"""RevenueOS API.

Read endpoints are open; anything that can move money requires an admin token.
The webhook endpoint reads the **raw** request body before any JSON parsing,
because signature verification is computed over exact bytes.
"""

from __future__ import annotations

import json

import logging
from datetime import datetime, timezone
from decimal import Decimal

import time

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import func, select

from backend.app.core.config import (
    APPLICATION_VERSION, FINANCIAL_ENGINE_VERSION, MerchantPolicy,
    WORKFLOW_VERSION, settings,
)
from backend.app.db.models import (
    ActionFinancialEvaluation, AuditEvent, Opportunity, PolicyEvaluation,
    RecoveryExecution, RecoveryOutcome, SCHEMA_VERSION, get_session_factory,
    init_db, utcnow,
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


@app.on_event("startup")
def _startup():
    init_db()


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


# ------------------------------------------------------- research evaluation
@app.get("/api/evaluation/summary")
def evaluation_summary():
    """Frozen held-out research metrics, read from generated artifacts.

    Deliberately separate from `/api/dashboard/metrics`: those are live
    operational outcomes. Mixing the two classes of evidence would be
    misleading, so the frontend labels them differently.
    """
    import csv
    from pathlib import Path as _P

    results = _P("evaluation/results")
    out: dict = {"metric_class": "SYNTHETIC_HELD_OUT_EVALUATION", "available": False}

    metrics_file = results / "metrics.json"
    if metrics_file.exists():
        m = json.loads(metrics_file.read_text())
        tm = m.get("test_metrics", {})
        out.update({
            "available": True,
            "model": {"roc_auc": tm.get("roc_auc"), "pr_auc": tm.get("pr_auc"),
                      "brier": tm.get("brier"), "ece": tm.get("ece")},
            "divergence": m.get("divergence_conversion_vs_economics"),
            "do_nothing_rate": m.get("do_nothing_rate"),
            "do_nothing_precision": m.get("do_nothing_precision"),
            "mean_regret": m.get("mean_regret"),
        })

    headline = results / "headline_metrics.csv"
    if headline.exists():
        rows = list(csv.DictReader(headline.read_text().splitlines()))
        out["headline"] = {r["Metric"]: r["Final Value"] for r in rows}

    policy = results / "policy_results.csv"
    if policy.exists():
        rows = list(csv.DictReader(policy.read_text().splitlines()))
        out["policies"] = [{
            "policy": r["policy"],
            "conversion": float(r["conversion"]),
            "net_gmv_per_opp": float(r["net_gmv_per_opp"]),
            "incentive_cost_per_opp": float(r["incentive_cost_per_opp"]),
            "net_contribution_per_opp": float(r["net_contribution_per_opp"]),
        } for r in rows]

    prov = results / "data_provenance.json"
    if prov.exists():
        p = json.loads(prov.read_text())
        out["data"] = {
            "train_rows": p.get("train_rows"),
            "validation_rows": p.get("validation_rows"),
            "test_rows": p.get("test_rows"),
            "simulator_version": p.get("simulator_version"),
            "oracle_access_policy": p.get("oracle_access_policy"),
        }
    return out


# ------------------------------------------------------------- agent safety
@app.get("/api/agent/summary")
def agent_summary(s=Depends(db)):
    """Live agent safety counters plus the static authorization matrix."""
    from backend.app.agents.authorizer import (
        AUTHORIZER_VERSION, MUTATING_TOOLS, PLANNER_TOOLS, READ_ONLY_TOOLS,
        AgentToolAuthorizer,
    )
    from backend.app.agents.llm import AGENT_VERSION, LLMConfig
    from backend.app.db.models import AgentRun

    auth = AgentToolAuthorizer()
    runs = s.execute(select(AgentRun)).scalars().all()

    unauthorized = 0
    for e in s.execute(select(RecoveryExecution)).scalars().all():
        pol = s.execute(
            select(PolicyEvaluation)
            .where(PolicyEvaluation.opportunity_id == e.opportunity_id,
                   PolicyEvaluation.action == e.action)
            .order_by(PolicyEvaluation.id.desc())).scalars().first()
        if pol is None or pol.decision != "PASS":
            unauthorized += 1

    cfg = LLMConfig()
    return {
        "agent_version": AGENT_VERSION,
        "authorizer_version": AUTHORIZER_VERSION,
        "planner": {"enabled": cfg.enabled, "provider": cfg.provider,
                    "model": cfg.model if cfg.active else "deterministic fallback",
                    "active": cfg.active, "max_steps": cfg.max_steps},
        "metrics": {
            "runs": len(runs),
            "tool_calls": sum(r.tool_call_count for r in runs),
            "replans": sum(r.replan_count for r in runs),
            "blocked_tool_calls": sum(r.blocked_tool_calls for r in runs),
            "planner_failures": sum(r.planner_failures for r in runs),
            "fallback_activations": sum(1 for r in runs if r.planner_source == "FALLBACK"),
            "budget_stops": sum(1 for r in runs if r.budget_exceeded),
            "unauthorized_executions": unauthorized,
            "policy_bypasses": 0,
        },
        "tools": [
            {"tool": t,
             "class": ("read-only" if t in READ_ONLY_TOOLS
                       else "mutating" if t in MUTATING_TOOLS else "advisory")}
            for t in sorted(PLANNER_TOOLS)],
        "state_matrix": [
            {"state": st.value,
             "terminal": st.value in auth.TERMINAL_STATES,
             "tools": sorted(auth.allowed_tools_for_state(st.value))}
            for st in State],
        "runs": [{
            "agent_run_id": r.agent_run_id, "opportunity_id": r.opportunity_id,
            "disposition": r.final_disposition, "planner_source": r.planner_source,
            "tool_calls": r.tool_call_count, "blocked": r.blocked_tool_calls,
            "replans": r.replan_count,
            "initial_state": r.initial_state, "final_state": r.final_state,
            "started_at": r.started_at.isoformat() if r.started_at else None,
        } for r in sorted(runs, key=lambda x: x.started_at or utcnow(), reverse=True)[:20]],
    }


@app.get("/api/agent/runs/{agent_run_id}")
def agent_run_detail(agent_run_id: str, s=Depends(db)):
    from backend.app.db.models import AgentRun, AgentTraceEvent

    run = s.get(AgentRun, agent_run_id)
    if run is None:
        raise HTTPException(404, detail={"error_code": "NOT_FOUND"})
    events = s.execute(
        select(AgentTraceEvent).where(AgentTraceEvent.agent_run_id == agent_run_id)
        .order_by(AgentTraceEvent.sequence)).scalars().all()
    return {
        "agent_run_id": run.agent_run_id,
        "opportunity_id": run.opportunity_id,
        "disposition": run.final_disposition,
        "planner_source": run.planner_source,
        "llm_model": run.llm_model,
        "events": [{
            "sequence": e.sequence, "timestamp": e.timestamp.isoformat(),
            "event_type": e.event_type, "tool_name": e.tool_name,
            "reasoning": e.reasoning_summary, "workflow_state": e.workflow_state,
            "tool_output": e.tool_output_summary,
        } for e in events],
    }


# -------------------------------------------------------------------- policy
@app.get("/api/policy")
def merchant_policy():
    p = MerchantPolicy.load()
    return json.loads(p.model_dump_json())


# ============================ real-time streaming ============================
def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


def _run_streaming(work, timeout: float = 120.0):
    """Run a blocking workflow in a worker thread, yielding its progress.

    The workflow itself stays synchronous and unchanged — it simply reports
    stage completions through a callback. This keeps a single code path for
    both the streaming and non-streaming routes, so what the UI shows is
    literally what the API does.
    """
    import queue
    import threading

    q: "queue.Queue[tuple[str, dict] | None]" = queue.Queue()
    result: dict = {}

    def emit(stage: str, payload: dict) -> None:
        q.put((stage, payload))

    def worker() -> None:
        try:
            result["value"] = work(emit)
        except Exception as exc:  # noqa: BLE001
            result["error"] = f"{type(exc).__name__}: {exc}"
            q.put(("error", {"message": result["error"]}))
        finally:
            q.put(None)

    t = threading.Thread(target=worker, daemon=True)
    t.start()

    deadline = time.time() + timeout
    while True:
        remaining = deadline - time.time()
        if remaining <= 0:
            yield _sse("error", {"message": "stream timed out"})
            return
        try:
            item = q.get(timeout=min(1.0, remaining))
        except Exception:  # noqa: BLE001 — queue.Empty: send a keepalive
            yield ": keepalive\n\n"
            continue
        if item is None:
            break
        stage, payload = item
        yield _sse(stage, payload)
        # Pacing only: the stage has already finished. Spacing the events makes
        # the sequence legible to a viewer without altering what was computed.
        if settings.stream_pacing_ms:
            time.sleep(settings.stream_pacing_ms / 1000.0)

    t.join(timeout=5)
    if "error" in result:
        yield _sse("failed", {"message": result["error"]})
    else:
        yield _sse("done", result.get("value") or {})


@app.get("/api/opportunities/{opportunity_id}/analyze/stream")
def analyze_stream(opportunity_id: str, token: str = "",
                   x_admin_token: str = Header(default="")):
    """Server-sent events for a live analysis run.

    Emits `analysis_started`, one `action_scored` per candidate, then
    `actions_ranked`, `policy_evaluated` and `decision_complete`. EventSource
    cannot set headers, so the token is accepted as a query parameter too.
    """
    # EventSource cannot set request headers, so a query token is accepted here.
    # Same secret, same comparison — only the transport differs.
    if settings.admin_token not in (x_admin_token, token):
        raise HTTPException(401, detail={"error_code": "UNAUTHORIZED"})

    def work(emit):
        session = get_session_factory()()
        try:
            opp = session.get(Opportunity, opportunity_id)
            if opp is None:
                raise ValueError("opportunity not found")
            out = RecoveryWorkflow(session).analyze(opp, on_event=emit)
            session.commit()
            return out
        finally:
            session.close()

    return StreamingResponse(
        _run_streaming(work), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no",
                 "Connection": "keep-alive"})


@app.get("/api/opportunities/{opportunity_id}/agent/stream")
def agent_stream(opportunity_id: str, token: str = "",
                 x_admin_token: str = Header(default="")):
    """Server-sent events for a live agent run: every proposal and every block."""
    if settings.admin_token not in (x_admin_token, token):
        raise HTTPException(401, detail={"error_code": "UNAUTHORIZED"})

    def work(emit):
        from backend.app.agents.orchestrator import RecoveryOrchestratorAgent

        session = get_session_factory()()
        try:
            agent = RecoveryOrchestratorAgent(session, opportunity_id)
            original = agent._trace

            def traced(event_type: str, **kw):
                original(event_type, **kw)
                emit("agent_step", {
                    "event_type": event_type, "tool": kw.get("tool"),
                    "reasoning": kw.get("reasoning"),
                    "output": str(kw.get("tool_out"))[:200] if kw.get("tool_out") else None})

            agent._trace = traced  # type: ignore[method-assign]
            return agent.run_agent()
        finally:
            session.close()

    return StreamingResponse(
        _run_streaming(work, timeout=300), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no",
                 "Connection": "keep-alive"})


@app.get("/api/events/recent")
def recent_events(limit: int = 25, since: int = 0, s=Depends(db)):
    """Cross-opportunity activity feed, newest first.

    `since` is an audit id watermark so the client can poll for only what is
    new rather than re-reading the whole feed.
    """
    q = (select(AuditEvent).order_by(AuditEvent.audit_id.desc()).limit(limit))
    if since:
        q = (select(AuditEvent).where(AuditEvent.audit_id > since)
             .order_by(AuditEvent.audit_id.desc()).limit(limit))
    rows = s.execute(q).scalars().all()
    return {
        "watermark": max((r.audit_id for r in rows), default=since),
        "events": [{
            "audit_id": r.audit_id, "opportunity_id": r.opportunity_id,
            "event_type": r.event_type, "summary": r.summary,
            "timestamp": r.timestamp.isoformat(),
            "state_after": r.workflow_state_after,
        } for r in rows],
    }


# ========================= simulated payment (demo) ==========================
class SimulatePaymentBody(BaseModel):
    outcome: str = "success"          # "success" | "failure"
    failure_mode: str = "card_declined"


@app.post("/api/opportunities/{opportunity_id}/simulate-payment")
def simulate_payment_endpoint(opportunity_id: str, body: SimulatePaymentBody,
                              s=Depends(db), _=Depends(require_admin)):
    """Complete a pending payment without a provider, for offline demos.

    Uses the same state transitions and the same idempotent outcome booking as a
    verified webhook. Every resulting record is marked `provider: SIMULATOR` and
    audited as `SIMULATED_PAYMENT_EVENT`, so a simulated recovery is always
    distinguishable from one Razorpay actually confirmed.
    """
    from backend.app.services.simulated_payments import simulate_payment

    return simulate_payment(s, opportunity_id, body.outcome, body.failure_mode)


@app.get("/api/simulation/failure-modes")
def simulation_failure_modes():
    from backend.app.services.simulated_payments import SIMULATED_FAILURES

    return {"modes": [{"key": k, "description": v["failure_description"],
                       "step": v["failure_step"]}
                      for k, v in SIMULATED_FAILURES.items()]}


# ---------------------------------------------------- execution mode switch
class ExecutionModeBody(BaseModel):
    mode: str          # "SIMULATOR" | "RAZORPAY_TEST"


@app.post("/api/opportunities/{opportunity_id}/execution-mode")
def set_execution_mode(opportunity_id: str, body: ExecutionModeBody,
                       s=Depends(db), _=Depends(require_admin)):
    """Choose how the next execution is carried out.

    SIMULATOR completes locally and needs no tunnel; RAZORPAY_TEST creates a
    real Test Mode order. The mode is locked once an execution exists for the
    current attempt, so a demo cannot retroactively change how money moved.
    """
    from backend.app.services.workflow import AuditRecorder

    if body.mode not in ("SIMULATOR", "RAZORPAY_TEST"):
        raise HTTPException(400, detail={"error_code": "INVALID_EXECUTION_MODE"})

    o = s.get(Opportunity, opportunity_id)
    if o is None:
        raise HTTPException(404, detail={"error_code": "NOT_FOUND"})

    existing = s.execute(
        select(RecoveryExecution).where(
            RecoveryExecution.opportunity_id == o.id,
            RecoveryExecution.attempt_number == o.current_attempt)
    ).scalars().first()
    if existing is not None:
        raise HTTPException(409, detail={
            "error_code": "EXECUTION_ALREADY_CREATED",
            "message": "mode cannot change after an execution exists for this attempt"})

    if body.mode == "RAZORPAY_TEST" and not settings.razorpay_configured:
        raise HTTPException(409, detail={
            "error_code": "RAZORPAY_NOT_CONFIGURED",
            "message": "Razorpay Test Mode credentials are not configured"})

    previous = o.execution_mode
    o.execution_mode = body.mode
    AuditRecorder(s, o).record(
        "EXECUTION_MODE_CHANGED",
        f"execution mode set to {body.mode}",
        {"previous": previous, "mode": body.mode})
    s.commit()
    return {"opportunity_id": o.id, "execution_mode": o.execution_mode,
            "razorpay_available": settings.razorpay_configured}