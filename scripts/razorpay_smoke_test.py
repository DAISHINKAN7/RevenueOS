"""Live Razorpay Test Mode smoke test.

    RAZORPAY_CLIENT=test python -m scripts.razorpay_smoke_test

Creates exactly ONE Test Mode order for one seeded opportunity, prints the
checkout payload, and waits for the webhook to reconcile. Never prints secrets.
Requires RAZORPAY_KEY_ID / RAZORPAY_KEY_SECRET (rzp_test_*) in the environment
and a publicly reachable webhook URL configured in the Razorpay dashboard.
"""

from __future__ import annotations

import json
import sys
import time

from backend.app.core.dotenv import load_dotenv

load_dotenv()  # before any settings are read

from backend.app.core.config import settings
from backend.app.db.models import Opportunity, RecoveryExecution, get_session_factory
from backend.app.domain import State
from backend.app.services.razorpay import RazorpayRecoveryExecutor
from backend.app.services.workflow import RecoveryWorkflow


def main() -> int:
    if settings.razorpay_client != "test":
        print("RAZORPAY_CLIENT must be 'test'. Refusing to run against the mock.")
        return 2
    if not settings.razorpay_configured:
        print("Razorpay Test Mode credentials are not configured.")
        return 2
    if not settings.razorpay_key_id.startswith("rzp_test"):
        print("KEY_ID is not a Test Mode key. Refusing.")
        return 2
    print(f"Test Mode confirmed. key id ends ...{settings.razorpay_key_id[-4:]}\n")

    s = get_session_factory()()
    opp = (s.query(Opportunity)
           .filter(Opportunity.execution_mode == "RAZORPAY_TEST")
           .filter(Opportunity.state.in_([State.DETECTED.value,
                                          State.AUTHORIZED.value]))
           .first())
    if opp is None:
        print("No RAZORPAY_TEST opportunity available. Run `make seed` first.")
        return 1
    print(f"opportunity {opp.id}  cart INR {opp.revenue_at_risk}  state {opp.state}")

    wf = RecoveryWorkflow(s)
    if opp.state == State.DETECTED.value:
        r = wf.analyze(opp)
        s.commit()
        print(f"selected  {r['selected_action']}  policy {r['policy']['decision']}")
        for line in r["explanation"]:
            print(f"  {line}")
    if opp.state != State.AUTHORIZED.value:
        print(f"\nNot authorized (state {opp.state}); no order will be created.")
        return 0

    result = wf.execute(opp, RazorpayRecoveryExecutor())
    print("\nOrder created:")
    print(json.dumps(result.get("checkout", {}), indent=2))
    print("\nComplete the payment with a Razorpay test card, then wait for the webhook.")

    for i in range(30):
        time.sleep(4)
        s.expire_all()
        cur = s.get(Opportunity, opp.id)
        print(f"  [{i * 4:>3}s] state = {cur.state}")
        if cur.state in (State.RECOVERED.value, State.NOT_RECOVERED.value):
            break

    cur = s.get(Opportunity, opp.id)
    print(f"\nfinal state: {cur.state}")
    ex = s.query(RecoveryExecution).filter_by(opportunity_id=opp.id).all()
    for e in ex:
        print(f"  execution {e.execution_id}  status {e.status}  order {e.external_order_id}")
    s.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())