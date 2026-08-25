"""Compare planner quality across providers on the same opportunity.

    make planner-compare

Runs the identical scenario through each configured planner and reports how
often each one proposed a tool the authorization layer had to block, how many
turns it wasted on unchanging reads, and whether it reached execution.

This isolates *planner quality* from system correctness. The safety layer is
identical in every column; only the model changes. A provider that scores badly
here is not unsafe — it is just less useful, and the deterministic fallback
covers it.

Configure a second provider in `.env` (nothing else needs to change):

    AGENT_COMPARE_BASE_URL=https://api.groq.com/openai/v1
    AGENT_COMPARE_API_KEY=gsk_...
    AGENT_COMPARE_MODEL=llama-3.3-70b-versatile
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from decimal import Decimal

from backend.app.agents.llm import (
    LLMConfig, MockLLMProvider, OpenAICompatibleProvider,
)
from backend.app.agents.orchestrator import RecoveryOrchestratorAgent
from backend.app.core.config import WORKFLOW_VERSION
from backend.app.core.dotenv import load_dotenv
from backend.app.db.models import (
    AgentTraceEvent, Opportunity, RecoveryExecution, get_session_factory, init_db, utcnow,
)
from backend.app.domain import State
from backend.app.services.workflow import new_trace_id

load_dotenv()


@dataclass
class Candidate:
    label: str
    provider: object
    model: str


def build_candidates() -> list[Candidate]:
    out: list[Candidate] = [
        Candidate("deterministic (no LLM)", MockLLMProvider(), "fallback router"),
    ]

    local = LLMConfig()
    if local.active:
        out.append(Candidate(f"local: {local.model}",
                             OpenAICompatibleProvider(local), local.model))
    else:
        print(f"note: local planner not included "
              f"(AGENT_LLM_ENABLED={local.enabled}, provider={local.provider!r}, "
              f"key_present={bool(local.api_key)}). "
              f"Set AGENT_LLM_ENABLED=true to include it.\n")

    base = os.getenv("AGENT_COMPARE_BASE_URL", "")
    key = os.getenv("AGENT_COMPARE_API_KEY", "")
    model = os.getenv("AGENT_COMPARE_MODEL", "")
    if base and model and (key or "localhost" in base):
        cfg = LLMConfig(enabled=True, provider="compare", base_url=base,
                        api_key=key, model=model,
                        timeout_seconds=float(os.getenv("AGENT_LLM_TIMEOUT", "30")))
        out.append(Candidate(f"hosted: {model}", OpenAICompatibleProvider(cfg), model))
    return out


def fresh_opportunity(session, label: str) -> Opportunity:
    import numpy as np
    import pandas as pd

    from backend.app.seed import CONTEXT_FIELDS

    t = pd.read_parquet("data/processed/test_features.parquet")
    row = t[(t.opportunity_type == "CHECKOUT_ABANDONMENT")
            & (t.shipping_fee_charged > 30)].iloc[0]
    ctx = {}
    for f in CONTEXT_FIELDS:
        if f not in row.index:
            continue
        v = row[f]
        ctx[f] = (None if pd.isna(v) else int(v) if isinstance(v, (bool, np.bool_))
                  else float(v) if isinstance(v, (int, float, np.integer, np.floating))
                  else str(v))
    o = Opportunity(
        id=f"OPP-CMP-{label}", opportunity_type=ctx["opportunity_type"],
        detected_at=utcnow(), state=State.DETECTED.value,
        workflow_version=WORKFLOW_VERSION, execution_mode="SIMULATOR",
        revenue_at_risk=Decimal(str(round(float(ctx["cart_value"]), 2))),
        contribution_margin_at_risk=Decimal(
            str(round(float(ctx["base_contribution_margin"]), 2))),
        current_attempt=1, trace_id=new_trace_id(), context=ctx)
    session.add(o)
    session.commit()
    return o


def probe(cand: Candidate) -> str | None:
    """Confirm the provider actually answers before crediting it with a result.

    Without this a provider that errors on every call still produces a clean
    row, because the deterministic fallback silently does the work. That row
    would be the fallback wearing the provider's name.
    """
    if isinstance(cand.provider, MockLLMProvider):
        return None
    try:
        out = cand.provider.decide_next_step(
            "You output JSON only. The state is terminal, so next_tool is STOP.",
            {"workflow_state": "RECOVERED", "available_tools": ["get_opportunity_state"]})
    except Exception as exc:  # noqa: BLE001
        return f"{type(exc).__name__}: {exc}"
    if not isinstance(out, dict) or not (out.get("next_tool") or out.get("next_step")):
        return f"responded but not in the required shape: {str(out)[:120]}"
    return None


def run_one(session, cand: Candidate, index: int) -> dict:
    o = fresh_opportunity(session, f"{index}")
    started = time.time()
    try:
        r = RecoveryOrchestratorAgent(session, o.id, provider=cand.provider).run_agent()
    except Exception as exc:  # noqa: BLE001
        return {"label": cand.label, "error": f"{type(exc).__name__}: {exc}"}
    elapsed = time.time() - started

    traces = (session.query(AgentTraceEvent)
              .filter_by(agent_run_id=r["agent_run_id"])
              .order_by(AgentTraceEvent.sequence).all())
    proposed = [t.tool_name for t in traces if t.event_type == "AGENT_TOOL_PROPOSED"]
    executed = session.query(RecoveryExecution).filter_by(opportunity_id=o.id).count()

    return {
        "label": cand.label,
        "model": cand.model,
        "disposition": r["disposition"],
        "final_state": r["final_state"],
        "proposals": len(proposed),
        "blocked": r["blocked_tool_calls"],
        "planner_failures": r["planner_failures"],
        "reached_execution": executed == 1,
        "executions": executed,
        "seconds": round(elapsed, 1),
        "proposed_sequence": proposed,
    }


def main() -> int:
    init_db(drop=True)
    session = get_session_factory()()
    candidates = build_candidates()

    print("Planner comparison — identical opportunity, identical safety layer.\n")
    if len(candidates) < 3:
        print("Note: configure AGENT_COMPARE_* in .env to add a hosted model.\n")

    results = []
    for i, cand in enumerate(candidates):
        print(f"running {cand.label} ...")

        failure = probe(cand)
        if failure:
            print(f"  UNREACHABLE — {failure}")
            print("  Skipping: a run here would only measure the deterministic "
                  "fallback, not this provider.\n")
            results.append({"label": cand.label, "unreachable": failure})
            continue

        res = run_one(session, cand, i)
        results.append(res)
        if "error" in res:
            print(f"  ERROR {res['error']}\n")
            continue
        if res["planner_failures"]:
            print(f"  WARNING: {res['planner_failures']} planner call(s) failed "
                  f"mid-run; the fallback covered them, so this row is not a "
                  f"clean measurement of the model.")
        print(f"  {res['disposition']:<26}proposals={res['proposals']:<3}"
              f"blocked={res['blocked']:<3}reached_execution={res['reached_execution']}"
              f"  {res['seconds']}s")
        print(f"  proposed: {' -> '.join(res['proposed_sequence'])}\n")

    print("=" * 78)
    print(f"{'planner':<32}{'disposition':<26}{'blocked':>8}{'exec':>6}")
    print("-" * 78)
    for r in results:
        if r.get("unreachable"):
            print(f"{r['label']:<32}{'UNREACHABLE':<26}{'-':>8}{'-':>6}")
            continue
        if "error" in r:
            print(f"{r['label']:<32}{'ERROR':<26}{'-':>8}{'-':>6}")
            continue
        suffix = "*" if r["planner_failures"] else ""
        print(f"{r['label']:<32}{r['disposition'] + suffix:<26}{r['blocked']:>8}"
              f"{'yes' if r['reached_execution'] else 'no':>6}")
    print("=" * 78)

    if any(r.get("planner_failures") for r in results if "error" not in r):
        print("\n* this planner failed one or more calls mid-run; the "
              "deterministic fallback covered them, so the row reflects the "
              "fallback rather than the model.")

    unreachable = [r for r in results if r.get("unreachable")]
    if unreachable:
        print("\nUnreachable providers:")
        for r in unreachable:
            print(f"  {r['label']}: {r['unreachable']}")
        print("\nCommon causes: wrong model name (check the provider's model "
              "list), invalid or missing API key, or a base URL not ending "
              "in /v1.")

    unsafe = [r for r in results
              if "error" not in r and r.get("executions", 0) > 1]
    print(f"\nRuns with more than one execution: {len(unsafe)} (must be 0)")
    print("Every planner is bounded by the same authorization layer; the "
          "difference between them is usefulness, not safety.")
    session.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())