"""Drive every scenario end to end with no tunnel and no Razorpay.

    make demo-offline

Uses the real workflow, the real policy engine and the real state machine.
Payments are completed by the simulated provider, which is marked as such in
every record it writes. Nothing about the decision is faked — only the payment
confirmation is local instead of arriving from Razorpay.
"""

from __future__ import annotations

import sys
import time

from backend.app.core.dotenv import load_dotenv

load_dotenv()

from backend.app.db.models import (  # noqa: E402
    Opportunity, RecoveryExecution, RecoveryOutcome, get_session_factory,
)
from backend.app.domain import State  # noqa: E402
from backend.app.services.simulated_payments import simulate_payment  # noqa: E402
from backend.app.services.workflow import (  # noqa: E402
    RecoveryWorkflow, SimulatorRecoveryExecutor,
)

PAUSE = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0


def pause() -> None:
    if PAUSE:
        time.sleep(PAUSE)


def banner(text: str) -> None:
    print(f"\n{'─' * 72}\n{text}\n{'─' * 72}")


def show(session, oid: str) -> None:
    o = session.get(Opportunity, oid)
    outcome = session.get(RecoveryOutcome, oid)
    line = f"    state {o.state}"
    if o.selected_action:
        line += f"  ·  action {o.selected_action}"
    if outcome:
        line += (f"  ·  recovered INR {outcome.net_recovered_gmv:,.2f}"
                 f"  ·  contribution INR {outcome.realized_contribution:,.2f}")
    print(line)


def run_scenario(session, opp: Opportunity, fail_first: bool) -> None:
    wf = RecoveryWorkflow(session)
    oid = opp.id

    r = wf.analyze(opp)
    session.commit()
    print(f"    analysis → {r['selected_action']} · policy {r['policy']['decision']}")
    for line in r["explanation"][1:4]:
        print(f"      {line}")
    pause()

    opp = session.get(Opportunity, oid)
    if opp.state == State.AWAITING_APPROVAL.value:
        print("    policy requires human approval — no execution created")
        show(session, oid)
        return
    if opp.selected_action == "DO_NOTHING":
        print("    no intervention has positive incremental value — declining to spend")
        show(session, oid)
        return
    if opp.state != State.AUTHORIZED.value:
        show(session, oid)
        return

    wf.execute(opp, SimulatorRecoveryExecutor())
    pause()

    if fail_first:
        r = simulate_payment(session, oid, "failure", "card_declined")
        print(f"    simulated failure → {r['state']} "
              f"({r['normalized_reason']} / {r['category']})")
        pause()

        opp = session.get(Opportunity, oid)
        r2 = wf.analyze(opp)
        session.commit()
        print(f"    re-analysis → {r2['selected_action']} "
              f"· policy {r2['policy']['decision']}")
        pause()

        opp = session.get(Opportunity, oid)
        if opp.state == State.AUTHORIZED.value:
            wf.execute(opp, SimulatorRecoveryExecutor())
            pause()

    opp = session.get(Opportunity, oid)
    if opp.state == State.AWAITING_PAYMENT.value:
        simulate_payment(session, oid, "success")
    show(session, oid)

    execs = session.query(RecoveryExecution).filter_by(opportunity_id=oid).all()
    if len(execs) > 1:
        print("    attempts:")
        for e in execs:
            print(f"      {e.attempt_number}. {e.action:<24}{e.status:<10}"
                  f"key {e.idempotency_key[:12]}")


def main() -> int:
    session = get_session_factory()()
    opportunities = session.query(Opportunity).order_by(Opportunity.id).all()
    if not opportunities:
        print("No opportunities. Run `make demo-reset` first.")
        return 1

    print("RevenueOS — offline demo. Real decisions, simulated payment confirmation.")

    for opp in opportunities:
        label = opp.id.split("-")[1]
        banner(f"{label}  ·  {opp.opportunity_type}  ·  INR {opp.revenue_at_risk:,.0f}")
        # The payment-failure scenario exercises the adaptive retry loop.
        run_scenario(session, opp, fail_first="DEMO3" in opp.id)
        pause()

    banner("Summary")
    outcomes = session.query(RecoveryOutcome).all()
    total_gmv = sum(float(o.net_recovered_gmv) for o in outcomes)
    total_contrib = sum(float(o.realized_contribution) for o in outcomes)
    print(f"    recovered opportunities   {len(outcomes)} of {len(opportunities)}")
    print(f"    net recovered GMV         INR {total_gmv:,.2f}")
    print(f"    realized contribution     INR {total_contrib:,.2f}")
    print("\n    Payments were confirmed by the simulated provider and are marked")
    print("    `provider: SIMULATOR` in the audit trail. Decisions, economics and")
    print("    policy are identical to the live path.")
    session.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())