"""Simulated payment outcomes for offline demonstration.

Why this exists
---------------
A `RAZORPAY_TEST` opportunity can only leave `AWAITING_PAYMENT` when a
signature-verified webhook arrives, which requires a public tunnel. Without one
every demo stalls. This module lets a simulated payment complete the workflow
using the *same* state transitions, the same idempotent outcome booking and the
same audit machinery as a real webhook.

What it deliberately does NOT do
--------------------------------
It never forges a Razorpay payload, never fabricates a signature, and never
claims a provider event occurred. Every event it produces is recorded with
`provider: SIMULATOR` and a distinct `SIMULATED_PAYMENT_EVENT` audit type, so a
simulated recovery can always be told apart from a verified one — in the audit
trail, in the API and in the UI.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from decimal import Decimal

from sqlalchemy import select

from backend.app.db.models import (
    Opportunity, PaymentFailureRecord, RecoveryExecution, RecoveryOutcome, utcnow,
)
from backend.app.domain import RevenueOSError, State, can_transition
from backend.app.services.failure_taxonomy import normalize_failure
from backend.app.services.workflow import AuditRecorder, RecoveryWorkflow, transition

# Failure shapes a simulated payment can produce. Chosen to exercise different
# branches of the adaptive retry layer rather than to be exhaustive.
SIMULATED_FAILURES: dict[str, dict[str, str]] = {
    "card_declined": {"failure_code": "BAD_REQUEST_ERROR",
                      "failure_step": "payment_authorization",
                      "failure_description": "Card declined by issuing bank",
                      "payment_method": "card"},
    "bank_timeout": {"failure_code": "GATEWAY_ERROR",
                     "failure_step": "payment_capture",
                     "failure_description": "Bank did not respond in time",
                     "payment_method": "netbanking"},
    "insufficient_funds": {"failure_code": "BAD_REQUEST_ERROR",
                           "failure_step": "payment_authorization",
                           "failure_description": "Payment failed due to insufficient funds",
                           "payment_method": "card"},
    "authentication": {"failure_code": "BAD_REQUEST_ERROR",
                       "failure_step": "payment_authentication",
                       "failure_description": "Customer failed authentication",
                       "payment_method": "card"},
    "user_cancelled": {"failure_code": "PAYMENT_CANCELLED",
                       "failure_step": "payment_initiation",
                       "failure_description": "Customer dismissed the checkout",
                       "payment_method": "upi"},
}


class SimulationError(RevenueOSError):
    code = "SIMULATION_NOT_APPLICABLE"
    http_status = 409


def _latest_execution(session, opportunity_id: str) -> RecoveryExecution | None:
    return session.execute(
        select(RecoveryExecution)
        .where(RecoveryExecution.opportunity_id == opportunity_id)
        .order_by(RecoveryExecution.attempt_number.desc())
    ).scalars().first()


def simulate_payment(session, opportunity_id: str, outcome: str,
                     failure_mode: str = "card_declined") -> dict:
    """Apply a simulated payment result to a pending opportunity.

    Mirrors `WebhookReconciler._apply` exactly: same target states, same
    absorbing-state protection, same single-booking guarantee. Only the
    provenance differs.
    """
    opp = session.get(Opportunity, opportunity_id)
    if opp is None:
        raise SimulationError("opportunity not found")

    state = State(opp.state)
    if state is State.RECOVERED:
        # RECOVERED is absorbing here exactly as it is for real webhooks.
        return {"status": "already_recovered", "state": opp.state, "simulated": True}
    if state not in (State.AWAITING_PAYMENT, State.EXECUTING, State.EXECUTION_PENDING):
        raise SimulationError(
            f"state {opp.state} has no payment pending; simulate a payment only "
            f"after an execution has been created")

    execution = _latest_execution(session, opportunity_id)
    if execution is None:
        raise SimulationError("no execution exists for this opportunity")

    audit = AuditRecorder(session, opp)
    payment_id = f"sim_pay_{uuid.uuid4().hex[:14]}"

    audit.record(
        "SIMULATED_PAYMENT_EVENT",
        f"simulated {outcome} payment applied (not a provider event)",
        {"provider": "SIMULATOR", "outcome": outcome, "payment_id": payment_id,
         "note": "Generated locally for demonstration. No Razorpay event occurred."},
        execution_id=execution.execution_id)

    if outcome == "success":
        return _apply_success(session, opp, execution, audit, payment_id)
    if outcome == "failure":
        return _apply_failure(session, opp, execution, audit, payment_id, failure_mode)
    raise SimulationError(f"unknown outcome {outcome!r}; expected 'success' or 'failure'")


def _apply_success(session, opp: Opportunity, execution: RecoveryExecution,
                   audit: AuditRecorder, payment_id: str) -> dict:
    if not can_transition(State(opp.state), State.RECOVERED):
        raise SimulationError(f"cannot move {opp.state} to RECOVERED")

    execution.status = "CAPTURED"
    execution.external_payment_id = payment_id
    gross = Decimal(str(execution.amount
                        or dict(opp.context).get("cart_value") or 0))

    transition(session, opp, State.RECOVERED, audit, "RECOVERY_CONFIRMED",
               "recovery confirmed by a simulated payment",
               {"provider": "SIMULATOR", "payment_id": payment_id,
                "gross": float(gross), "verified_by_provider": False})

    counted = RecoveryWorkflow(session).confirm_recovery(
        opp, execution, gross, payment_id=payment_id)
    audit.record("RECOVERY_CONFIRMED",
                 "financial outcome recorded" if counted
                 else "outcome already recorded; not double-counted",
                 {"counted": counted, "provider": "SIMULATOR"},
                 execution_id=execution.execution_id)
    session.commit()

    outcome = session.get(RecoveryOutcome, opp.id)
    return {
        "status": "recovered", "state": opp.state, "simulated": True,
        "counted": counted, "payment_id": payment_id,
        "net_recovered_gmv": float(outcome.net_recovered_gmv) if outcome else None,
        "realized_contribution": float(outcome.realized_contribution) if outcome else None,
    }


def _apply_failure(session, opp: Opportunity, execution: RecoveryExecution,
                   audit: AuditRecorder, payment_id: str, failure_mode: str) -> dict:
    shape = SIMULATED_FAILURES.get(failure_mode)
    if shape is None:
        raise SimulationError(
            f"unknown failure mode {failure_mode!r}; "
            f"expected one of {sorted(SIMULATED_FAILURES)}")

    execution.status = "FAILED"
    execution.external_payment_id = payment_id
    session.add(PaymentFailureRecord(
        opportunity_id=opp.id, execution_id=execution.execution_id,
        failure_code=shape["failure_code"], failure_step=shape["failure_step"],
        failure_description=shape["failure_description"],
        failure_source="simulator", payment_method=shape["payment_method"],
        payment_id=payment_id, provider_timestamp=datetime.now(timezone.utc)))

    # A failed payment is recoverable, never terminal — same rule as a real one.
    if not can_transition(State(opp.state), State.PAYMENT_FAILED_RECOVERABLE):
        raise SimulationError(f"cannot move {opp.state} to PAYMENT_FAILED_RECOVERABLE")

    reason, category = normalize_failure(
        shape["failure_code"], shape["failure_step"], shape["failure_description"])
    transition(session, opp, State.PAYMENT_FAILED_RECOVERABLE, audit, "PAYMENT_FAILED",
               f"simulated payment failed: {shape['failure_code']}",
               {"provider": "SIMULATOR", "error_code": shape["failure_code"],
                "error_step": shape["failure_step"],
                "normalized_reason": reason, "category": category.value,
                "verified_by_provider": False})
    session.commit()

    return {"status": "payment_failed", "state": opp.state, "simulated": True,
            "failure_code": shape["failure_code"],
            "failure_step": shape["failure_step"],
            "normalized_reason": reason, "category": category.value,
            "payment_id": payment_id}