"""Run the Recovery Orchestrator against one opportunity.

    make agent-run OPP=OPP-DEMO1-xxxx
    make agent-run                      # picks the first non-terminal opportunity

Prints the agent trace and the merchant explanation. Works with no LLM
configured (deterministic planner) and with any OpenAI-compatible free provider.
"""

from __future__ import annotations

import sys

from backend.app.core.dotenv import load_dotenv

load_dotenv()  # before any settings are read

from backend.app.agents.llm import LLMConfig
from backend.app.agents.orchestrator import (
    MerchantExplanationAgent, RecoveryOrchestratorAgent,
)
from backend.app.db.models import AgentTraceEvent, Opportunity, get_session_factory
from backend.app.domain import State

TERMINAL = {State.RECOVERED.value, State.STOPPED.value, State.EXPIRED.value}


def main() -> int:
    s = get_session_factory()()
    cfg = LLMConfig()
    print(f"planner: {'LLM ' + cfg.model if cfg.active else 'deterministic (no LLM configured)'}\n")

    oid = sys.argv[1] if len(sys.argv) > 1 else None
    if oid:
        opp = s.get(Opportunity, oid)
    else:
        opp = (s.query(Opportunity)
               .filter(~Opportunity.state.in_(list(TERMINAL))).first())
    if opp is None:
        print("No suitable opportunity. Run: python -m backend.app.seed")
        return 1

    print(f"opportunity {opp.id}  {opp.opportunity_type}  state {opp.state}\n")
    result = RecoveryOrchestratorAgent(s, opp.id).run_agent()

    for t in (s.query(AgentTraceEvent)
              .filter_by(agent_run_id=result["agent_run_id"])
              .order_by(AgentTraceEvent.sequence).all()):
        tool = f"[{t.tool_name}]" if t.tool_name else ""
        print(f"  {t.sequence:>2} {t.event_type:<24}{tool:<30}{t.reasoning_summary or ''}")

    print(f"\ndisposition {result['disposition']}  tools {result['tool_calls']}  "
          f"replans {result['replans']}  final state {result['final_state']}")

    from backend.app.api import opportunity_detail
    detail = opportunity_detail(opp.id, s)
    ex = MerchantExplanationAgent(s).explain(detail)
    print("\n--- merchant explanation ---")
    print(ex["merchant_summary"])
    if ex["why_not_alternatives"]:
        print("\nwhy not:")
        for w in ex["why_not_alternatives"]:
            print(f"  - {w}")
    print(f"\n{ex['policy_explanation']}\n{ex['risk_summary']}\n{ex['outcome_summary']}")
    s.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())