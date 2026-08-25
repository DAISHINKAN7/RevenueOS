"""Razorpay Test Mode integration and webhook reconciliation.

Verified against Razorpay's current documentation (Aug 2026):

* Webhook signature is HMAC-SHA256 over the **raw request body**, keyed by the
  webhook secret, delivered in `X-Razorpay-Signature`. The body must not be
  parsed or re-serialised before verification.
* `x-razorpay-event-id` is unique per event and is the documented deduplication
  key.
* Endpoints must return 2xx within 5 seconds; failures are retried with
  exponential backoff for 24 hours. So the handler verifies, persists, and
  acknowledges — processing happens after the response is committed.
* Webhook ordering is explicitly not guaranteed, which is why reconciliation is
  driven by event semantics rather than arrival order.
* Order amounts are in the smallest currency unit (paise for INR).

Test Mode only. This module refuses to construct a live client.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import uuid
from datetime import datetime, timezone
from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from backend.app.core.config import settings
from backend.app.db.models import (
    Opportunity, PaymentFailureRecord, RecoveryExecution, WebhookInbox, utcnow,
)
from backend.app.domain import (
    RazorpayAPIError, State, UnmatchedWebhook, WebhookSignatureInvalid, can_transition,
)
from backend.app.services.workflow import AuditRecorder, RecoveryWorkflow, money, transition

log = logging.getLogger("revenueos.razorpay")

# Razorpay event -> internal event. Kept in one place so no string comparison
# leaks into business logic.
EVENT_MAP = {
    "payment.failed": "PAYMENT_FAILED",
    "payment.captured": "PAYMENT_CAPTURED",
    "payment.authorized": "PAYMENT_AUTHORIZED",
    "order.paid": "ORDER_PAID",
}

# Events that finalise recovery.
SUCCESS_EVENTS = {"PAYMENT_CAPTURED", "ORDER_PAID"}


def to_paise(amount: Decimal | float) -> int:
    """Razorpay amounts are in the smallest currency unit."""
    return int((Decimal(str(amount)) * 100).quantize(Decimal("1")))


def from_paise(paise: int) -> Decimal:
    return (Decimal(int(paise)) / 100).quantize(Decimal("0.01"))


def verify_webhook_signature(raw_body: bytes, signature: str, secret: str) -> bool:
    """HMAC-SHA256 over the raw bytes. Constant-time comparison."""
    if not signature or not secret:
        return False
    expected = hmac.new(secret.encode(), raw_body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature)


def verify_checkout_signature(order_id: str, payment_id: str, signature: str,
                              secret: str) -> bool:
    """Standard Checkout callback signature: HMAC over 'order_id|payment_id'."""
    if not signature or not secret:
        return False
    expected = hmac.new(secret.encode(), f"{order_id}|{payment_id}".encode(),
                        hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature)


# ------------------------------------------------------------------- clients
class MockRazorpayClient:
    """Deterministic stand-in used by CI and by every automated test."""

    mode = "mock"

    def __init__(self):
        self.orders: dict[str, dict] = {}
        self.fail_next = False

    def create_order(self, amount_paise: int, currency: str, receipt: str,
                     notes: dict | None = None) -> dict:
        if self.fail_next:
            self.fail_next = False
            raise RazorpayAPIError("simulated Razorpay outage")
        oid = f"order_mock{uuid.uuid4().hex[:12]}"
        o = {"id": oid, "amount": amount_paise, "currency": currency,
             "receipt": receipt, "status": "created", "notes": notes or {}}
        self.orders[oid] = o
        return o

    def fetch_order(self, order_id: str) -> dict:
        if order_id not in self.orders:
            raise RazorpayAPIError(f"unknown order {order_id}")
        return self.orders[order_id]

    def fetch_payment(self, payment_id: str) -> dict:
        return {"id": payment_id, "status": "captured"}


class RazorpayTestClient:
    """Thin wrapper over the official SDK, pinned to Test Mode."""

    mode = "test"

    def __init__(self):
        if not settings.razorpay_configured:
            raise RazorpayAPIError("Razorpay credentials not configured")
        if settings.razorpay_mode != "test":
            raise RazorpayAPIError("RAZORPAY_MODE must be 'test'; live mode is refused")
        if not settings.razorpay_key_id.startswith("rzp_test"):
            raise RazorpayAPIError(
                "RAZORPAY_KEY_ID does not look like a Test Mode key (expected rzp_test_*)")
        import razorpay
        self._c = razorpay.Client(
            auth=(settings.razorpay_key_id, settings.razorpay_key_secret))
        self._c.set_app_details({"title": "RevenueOS", "version": "0.5.0"})

    def create_order(self, amount_paise: int, currency: str, receipt: str,
                     notes: dict | None = None) -> dict:
        try:
            return self._c.order.create({
                "amount": amount_paise, "currency": currency,
                "receipt": receipt[:40], "notes": notes or {},
                "payment_capture": 1,
            })
        except Exception as exc:  # noqa: BLE001
            raise RazorpayAPIError(f"order create failed: {type(exc).__name__}") from exc

    def fetch_order(self, order_id: str) -> dict:
        try:
            return self._c.order.fetch(order_id)
        except Exception as exc:  # noqa: BLE001
            raise RazorpayAPIError(f"order fetch failed: {type(exc).__name__}") from exc

    def fetch_payment(self, payment_id: str) -> dict:
        try:
            return self._c.payment.fetch(payment_id)
        except Exception as exc:  # noqa: BLE001
            raise RazorpayAPIError(f"payment fetch failed: {type(exc).__name__}") from exc


_client = None


def get_razorpay_client():
    global _client
    if _client is None:
        _client = (RazorpayTestClient() if settings.razorpay_client == "test"
                   else MockRazorpayClient())
    return _client


def reset_client() -> None:
    global _client
    _client = None


# ------------------------------------------------------------------ executor
class RazorpayRecoveryExecutor:
    """Creates a real Test Mode order for one explicitly selected opportunity."""

    name = "RAZORPAY_TEST"

    def __init__(self, client=None):
        self.client = client or get_razorpay_client()

    def execute(self, opp: Opportunity, execution: RecoveryExecution, session) -> dict:
        from ml.actions import Action, spec_for

        ctx = dict(opp.context)
        spec = spec_for(execution.action)

        if execution.action == Action.DO_NOTHING.value:
            return {"status": "COMPLETED", "amount": 0,
                    "terminal_state": "NOT_RECOVERED",
                    "summary": "no payment required"}

        # The backend derives the amount. A client-supplied amount is never
        # authoritative.
        cart = Decimal(str(ctx["cart_value"]))
        discount = (cart * Decimal(str(spec.discount_percent)) / 100).quantize(Decimal("0.01"))
        fee = Decimal("0") if spec.waives_shipping_fee else Decimal(
            str(ctx.get("shipping_fee_charged", 0)))
        amount = (cart - discount + fee).quantize(Decimal("0.01"))
        if amount <= 0:
            raise RazorpayAPIError(f"computed non-positive amount {amount}")

        order = self.client.create_order(
            amount_paise=to_paise(amount), currency=settings.currency,
            receipt=f"rvos_{execution.execution_id}",
            notes={"opportunity_id": opp.id, "execution_id": execution.execution_id,
                   "action": execution.action})

        return {
            "status": "SUBMITTED", "amount": float(amount), "order_id": order["id"],
            "checkout": {
                # Only what the browser legitimately needs. No secrets.
                "razorpay_key_id": settings.razorpay_key_id,
                "razorpay_order_id": order["id"],
                "amount_paise": order["amount"],
                "currency": order["currency"],
                "display_name": settings.merchant_display_name,
                "payment_environment": "TEST",
            },
        }


# ---------------------------------------------------------------- reconciler
class WebhookReconciler:
    """Applies verified events to workflow state, order-independently."""

    def __init__(self, session):
        self.s = session

    def receive(self, raw_body: bytes, signature: str, event_id: str) -> dict:
        """Verify, persist, acknowledge. Fast path only — no external calls."""
        import json

        secret = settings.razorpay_webhook_secret
        valid = verify_webhook_signature(raw_body, signature, secret)
        if not valid:
            # Never persist as a trusted financial event.
            log.warning("webhook_signature_invalid", extra={"event_id": event_id})
            raise WebhookSignatureInvalid("signature verification failed")

        payload = json.loads(raw_body)
        event_type = payload.get("event", "unknown")
        digest = hashlib.sha256(raw_body).hexdigest()

        row = WebhookInbox(
            provider="razorpay", event_id=event_id or digest, event_type=event_type,
            payload_hash=digest, payload=payload, signature_valid=True,
            processing_status="RECEIVED",
            provider_created_at=payload.get("created_at"))
        self.s.add(row)
        try:
            self.s.commit()
        except IntegrityError:
            self.s.rollback()
            existing = self.s.execute(
                select(WebhookInbox).where(
                    WebhookInbox.provider == "razorpay",
                    WebhookInbox.event_id == (event_id or digest))
            ).scalar_one_or_none()
            log.info("webhook_duplicate", extra={"event_id": event_id})
            return {"status": "duplicate", "event_id": event_id,
                    "inbox_id": existing.id if existing else None,
                    "processing_status": "DUPLICATE"}
        return {"status": "accepted", "event_id": row.event_id, "inbox_id": row.id}

    # ------------------------------------------------------------- processing
    def process(self, inbox_id: int) -> dict:
        row = self.s.get(WebhookInbox, inbox_id)
        if row is None:
            return {"status": "missing"}
        if row.processing_status == "PROCESSED":
            return {"status": "already_processed"}

        row.processing_status = "PROCESSING"
        row.processing_attempts += 1
        self.s.flush()

        try:
            internal = EVENT_MAP.get(row.event_type)
            if internal is None:
                row.processing_status = "IGNORED"
                row.processed_at = utcnow()
                self.s.commit()
                return {"status": "ignored", "event_type": row.event_type}

            execution = self._match(row.payload)
            if execution is None:
                row.processing_status = "PROCESSED"
                row.processed_at = utcnow()
                row.last_error = "UNMATCHED_WEBHOOK"
                self.s.commit()
                return {"status": "unmatched", "event_type": row.event_type}

            opp = self.s.get(Opportunity, execution.opportunity_id)
            result = self._apply(opp, execution, internal, row)
            row.processing_status = "PROCESSED"
            row.processed_at = utcnow()
            self.s.commit()
            return result
        except Exception as exc:  # noqa: BLE001
            self.s.rollback()
            row = self.s.get(WebhookInbox, inbox_id)
            row.processing_status = "FAILED"
            row.last_error = f"{type(exc).__name__}: {exc}"[:500]
            self.s.commit()
            return {"status": "failed", "error": row.last_error}

    def _match(self, payload: dict) -> RecoveryExecution | None:
        """Correlate strictly on identifiers. Never attach heuristically."""
        entities = (payload.get("payload") or {})
        order_id = payment_id = None
        for key in ("payment", "order"):
            ent = (entities.get(key) or {}).get("entity") or {}
            order_id = order_id or ent.get("order_id") or (ent.get("id") if key == "order" else None)
            payment_id = payment_id or (ent.get("id") if key == "payment" else None)

        # Prefer the internal reference we attached at order creation.
        notes = ((entities.get("payment") or {}).get("entity") or {}).get("notes") or {}
        exec_id = notes.get("execution_id")
        if exec_id:
            ex = self.s.get(RecoveryExecution, exec_id)
            if ex:
                return ex
        if order_id:
            ex = self.s.execute(
                select(RecoveryExecution)
                .where(RecoveryExecution.external_order_id == order_id)
            ).scalars().first()
            if ex:
                if payment_id and not ex.external_payment_id:
                    ex.external_payment_id = payment_id
                return ex
        if payment_id:
            return self.s.execute(
                select(RecoveryExecution)
                .where(RecoveryExecution.external_payment_id == payment_id)
            ).scalars().first()
        return None

    def _apply(self, opp: Opportunity, execution: RecoveryExecution,
               internal_event: str, row: WebhookInbox) -> dict:
        audit = AuditRecorder(self.s, opp)
        entity = (((row.payload.get("payload") or {}).get("payment") or {}).get("entity") or {})
        current = State(opp.state)

        audit.record("WEBHOOK_RECEIVED", f"{row.event_type} received",
                     {"event_id": row.event_id, "event_type": row.event_type,
                      "payment_id": entity.get("id")},
                     execution_id=execution.execution_id)

        # RECOVERED is absorbing: a delayed failure event must not regress it.
        if current in (State.RECOVERED,):
            audit.record("AUDIT_CORRECTION",
                         f"{row.event_type} arrived after recovery; state preserved",
                         {"event_id": row.event_id, "state": current.value})
            return {"status": "ignored_terminal", "state": opp.state}

        if internal_event in SUCCESS_EVENTS:
            execution.external_payment_id = entity.get("id") or execution.external_payment_id
            execution.status = "CAPTURED"
            amount = entity.get("amount")
            gross = from_paise(amount) if amount else Decimal(
                str(dict(opp.context).get("cart_value", 0)))

            # Out-of-order tolerance: a capture may land while we are still in
            # EXECUTING, AWAITING_PAYMENT, or after a prior failure.
            if not can_transition(current, State.RECOVERED):
                audit.record("PAYMENT_STATE_CONFLICT",
                             f"cannot move {current.value} -> RECOVERED",
                             {"event_id": row.event_id})
                return {"status": "conflict", "state": opp.state}

            transition(self.s, opp, State.RECOVERED, audit,
                       "RECOVERY_CONFIRMED",
                       f"recovery confirmed by {row.event_type}",
                       {"payment_id": execution.external_payment_id,
                        "gross": float(gross)})
            wf = RecoveryWorkflow(self.s)
            counted = wf.confirm_recovery(opp, execution, gross,
                                          payment_id=execution.external_payment_id)
            audit.record("RECOVERY_CONFIRMED",
                         "financial outcome recorded" if counted
                         else "outcome already recorded; not double-counted",
                         {"counted": counted}, execution_id=execution.execution_id)
            return {"status": "recovered", "state": opp.state, "counted": counted}

        if internal_event == "PAYMENT_FAILED":
            execution.status = "FAILED"
            self.s.add(PaymentFailureRecord(
                opportunity_id=opp.id, execution_id=execution.execution_id,
                failure_code=entity.get("error_code"),
                failure_description=entity.get("error_description"),
                failure_source=entity.get("error_source"),
                failure_step=entity.get("error_step"),
                payment_method=entity.get("method"),
                payment_id=entity.get("id"),
                provider_timestamp=datetime.fromtimestamp(
                    entity.get("created_at", 0) or 0, tz=timezone.utc)
                if entity.get("created_at") else None))

            # A failed payment is NOT necessarily terminal: the same journey may
            # still be captured later. Move to a recoverable holding state.
            target = (State.PAYMENT_FAILED_RECOVERABLE
                      if can_transition(current, State.PAYMENT_FAILED_RECOVERABLE)
                      else None)
            if target is None:
                audit.record("PAYMENT_STATE_CONFLICT",
                             f"cannot move {current.value} -> PAYMENT_FAILED_RECOVERABLE",
                             {"event_id": row.event_id})
                return {"status": "conflict", "state": opp.state}

            transition(self.s, opp, target, audit, "PAYMENT_FAILED",
                       f"payment failed: {entity.get('error_code')}",
                       {"error_code": entity.get("error_code"),
                        "error_step": entity.get("error_step")})
            return {"status": "payment_failed", "state": opp.state}

        if internal_event == "PAYMENT_AUTHORIZED":
            execution.status = "AUTHORIZED"
            return {"status": "authorized", "state": opp.state}

        return {"status": "ignored", "state": opp.state}


def reprocess_failed_webhook(session, event_id: str) -> dict:
    """Idempotent manual replay hook for operations."""
    row = session.execute(
        select(WebhookInbox).where(WebhookInbox.event_id == event_id)
    ).scalar_one_or_none()
    if row is None:
        return {"status": "not_found"}
    if row.processing_status == "PROCESSED":
        return {"status": "already_processed"}
    return WebhookReconciler(session).process(row.id)