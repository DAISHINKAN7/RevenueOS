"""Guided failure -> retry -> recovery demo on live Razorpay Test Mode.

    make retry-demo

Walks one opportunity through the full Track 03 loop:

    detect -> decide -> execute -> FAIL -> re-decide (attempt 2) -> recover

It pauses at each payment step, writes a ready-to-open `checkout.html` with the
live order details filled in, and tells you exactly which payment option to
choose. The final audit trail is saved to `evaluation/results/live/`.

Nothing here bypasses the normal workflow: every step goes through the same
analyze / policy / execute path the API uses.
"""

from __future__ import annotations

import json
import sys
import time
import webbrowser
from pathlib import Path

from backend.app.core.config import settings
from backend.app.db.models import (
    Opportunity, RecoveryExecution, get_session_factory,
)
from backend.app.domain import State
from backend.app.services.razorpay import RazorpayRecoveryExecutor
from backend.app.services.workflow import RecoveryWorkflow

OUT = Path("evaluation/results/live")
CHECKOUT = Path("checkout.html")

CHECKOUT_HTML = """<!doctype html><html><head><meta charset="utf-8"></head>
<body style="font-family:system-ui;padding:40px;max-width:640px">
<h2>RevenueOS — Razorpay Test Mode</h2>
<p><strong>{title}</strong></p>
<p>Order <code>{order_id}</code> — ₹{amount:,.2f} ({action})</p>
<div style="background:{bg};padding:16px;border-radius:8px;margin:20px 0">
<strong>{instruction}</strong>
</div>
<button onclick="pay()" style="padding:14px 28px;font-size:16px;cursor:pointer">
Open checkout</button>
<p id="s" style="color:#555;margin-top:20px"></p>
<script src="https://checkout.razorpay.com/v1/checkout.js"></script>
<script>
function pay(){{
  var rzp = new Razorpay({{
    key: "{key_id}",
    order_id: "{order_id}",
    amount: {amount_paise},
    currency: "INR",
    name: "NovaCart",
    description: "Recovery — {action}",
    handler: function(r){{
      document.getElementById('s').innerHTML =
        "Payment submitted. payment_id <code>"+r.razorpay_payment_id+"</code>";
    }},
    modal: {{ ondismiss: function(){{
      document.getElementById('s').innerText =
        "Checkout dismissed — that also produces a failed payment.";
    }} }}
  }});
  rzp.on('payment.failed', function(r){{
    document.getElementById('s').innerHTML =
      "Payment failed: <code>"+r.error.code+"</code> — this is what we want for step 1.";
  }});
  rzp.open();
}}
</script></body></html>
"""


def write_checkout(order_id, amount, action, title, instruction, bg="#fff3cd"):
    CHECKOUT.write_text(CHECKOUT_HTML.format(
        key_id=settings.razorpay_key_id, order_id=order_id, amount=amount,
        amount_paise=int(round(amount * 100)), action=action,
        title=title, instruction=instruction, bg=bg))


def wait_for(session, opp_id: str, targets: set[str], timeout: int = 180) -> str:
    """Poll until the opportunity reaches one of `targets`."""
    start = time.time()
    last = None
    while time.time() - start < timeout:
        session.expire_all()
        state = session.get(Opportunity, opp_id).state
        if state != last:
            print(f"    [{int(time.time() - start):>3}s] {state}")
            last = state
        if state in targets:
            return state
        time.sleep(3)
    return last or "TIMEOUT"


def banner(n: int, text: str) -> None:
    print(f"\n{'=' * 66}\nSTEP {n}  {text}\n{'=' * 66}")


def main() -> int:
    if settings.razorpay_client != "test" or not settings.razorpay_configured:
        print("Razorpay Test Mode is not active.")
        print("Run:  set -a && source .env && set +a")
        return 2

    OUT.mkdir(parents=True, exist_ok=True)
    s = get_session_factory()()

    opp = (s.query(Opportunity)
           .filter(Opportunity.execution_mode == "RAZORPAY_TEST",
                   Opportunity.state == State.DETECTED.value)
           .first())
    if opp is None:
        print("No fresh RAZORPAY_TEST opportunity. Run: python -m backend.app.seed")
        return 1

    oid = opp.id
    print(f"opportunity {oid}  {opp.opportunity_type}  cart INR {opp.revenue_at_risk}")

    wf = RecoveryWorkflow(s)

    # ---------------------------------------------------------------- attempt 1
    banner(1, "Analyze and authorize (attempt 1)")
    r = wf.analyze(opp)
    s.commit()
    for line in r["explanation"]:
        print(f"  {line}")
    if opp.state != State.AUTHORIZED.value:
        print(f"\nNot authorized (state {opp.state}). Nothing to execute.")
        return 0

    banner(2, "Create Razorpay order and FAIL the payment deliberately")
    ex = wf.execute(opp, RazorpayRecoveryExecutor())
    checkout = ex["checkout"]
    amount = checkout["amount_paise"] / 100
    write_checkout(
        checkout["razorpay_order_id"], amount, opp.selected_action,
        "Step 1 of 2 — make this payment FAIL",
        "Choose Netbanking, pick any bank, then click FAILURE on the simulated "
        "bank page. Dismissing the checkout also works.")
    print(f"  order {checkout['razorpay_order_id']}  INR {amount:,.2f}")
    print(f"\n  Opening {CHECKOUT} — choose Netbanking then FAILURE.")
    webbrowser.open(f"file://{CHECKOUT.resolve()}")

    state = wait_for(s, oid, {State.PAYMENT_FAILED_RECOVERABLE.value,
                              State.RECOVERED.value})
    if state == State.RECOVERED.value:
        print("\n  That payment succeeded, so there is no failure to recover from.")
        print("  Re-run after `python -m backend.app.seed` and choose FAILURE.")
        return 0
    if state != State.PAYMENT_FAILED_RECOVERABLE.value:
        print(f"\n  Expected PAYMENT_FAILED_RECOVERABLE, got {state}.")
        print("  Check the API terminal for the incoming webhook.")
        return 1
    print("\n  Failure reconciled. Note the state is RECOVERABLE, not lost.")

    # ---------------------------------------------------------------- attempt 2
    banner(3, "Re-analyze — attempt 2, fresh policy check, new idempotency key")
    opp = s.get(Opportunity, oid)
    r2 = wf.analyze(opp)
    s.commit()
    print(f"  attempt now {opp.current_attempt}")
    for line in r2["explanation"]:
        print(f"  {line}")
    if opp.state != State.AUTHORIZED.value:
        print(f"\n  Policy did not authorize attempt 2 (state {opp.state}).")
        print("  That is a valid outcome — bounded autonomy working as designed.")
        _save(s, oid)
        return 0

    banner(4, "Execute attempt 2 and SUCCEED")
    ex2 = wf.execute(opp, RazorpayRecoveryExecutor())
    c2 = ex2["checkout"]
    amount2 = c2["amount_paise"] / 100
    write_checkout(
        c2["razorpay_order_id"], amount2, opp.selected_action,
        "Step 2 of 2 — make this payment SUCCEED",
        "Choose Netbanking, pick any bank, then click SUCCESS.", bg="#d4edda")
    print(f"  order {c2['razorpay_order_id']}  INR {amount2:,.2f}")
    print(f"\n  Opening {CHECKOUT} — this time choose SUCCESS.")
    webbrowser.open(f"file://{CHECKOUT.resolve()}")

    state = wait_for(s, oid, {State.RECOVERED.value, State.NOT_RECOVERED.value})
    banner(5, f"Final state: {state}")
    _save(s, oid)
    return 0 if state == State.RECOVERED.value else 1


def _save(session, oid: str) -> None:
    from backend.app.db.models import AuditEvent, RecoveryOutcome

    session.expire_all()
    opp = session.get(Opportunity, oid)
    outcome = session.get(RecoveryOutcome, oid)
    events = (session.query(AuditEvent).filter_by(opportunity_id=oid)
              .order_by(AuditEvent.sequence_number).all())
    execs = session.query(RecoveryExecution).filter_by(opportunity_id=oid).all()

    print(f"\n  attempts: {opp.current_attempt}   executions: {len(execs)}")
    for e in execs:
        print(f"    attempt {e.attempt_number}  {e.action:<16}{e.status:<12}"
              f"{e.external_order_id}")
    keys = {e.idempotency_key for e in execs}
    print(f"  distinct idempotency keys: {len(keys)} (one per attempt)")
    if outcome:
        print(f"  recovered GMV INR {outcome.net_recovered_gmv}   "
              f"contribution INR {outcome.realized_contribution}")

    doc = {
        "opportunity_id": oid,
        "final_state": opp.state,
        "attempts": opp.current_attempt,
        "executions": [{"attempt": e.attempt_number, "action": e.action,
                        "status": e.status, "order_id": e.external_order_id,
                        "payment_id": e.external_payment_id,
                        "idempotency_key": e.idempotency_key} for e in execs],
        "outcome": ({"net_recovered_gmv": float(outcome.net_recovered_gmv),
                     "realized_contribution": float(outcome.realized_contribution)}
                    if outcome else None),
        "audit": [{"sequence": a.sequence_number, "timestamp": a.timestamp.isoformat(),
                   "event_type": a.event_type, "summary": a.summary,
                   "state_before": a.workflow_state_before,
                   "state_after": a.workflow_state_after,
                   "payload": a.structured_payload} for a in events],
    }
    path = OUT / "failure_then_recovery.json"
    path.write_text(json.dumps(doc, indent=2, default=str))
    print(f"\n  saved {path}  ({len(events)} audit events)")


if __name__ == "__main__":
    sys.exit(main())