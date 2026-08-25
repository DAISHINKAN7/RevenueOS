"""Phase 6 Razorpay integration tests.

Every scenario from the spec's failure matrix, driven through the real
reconciler with the mock client. No network access, so this runs in CI.

Fixture payloads mirror the documented Razorpay webhook envelope shape. They
contain no real credentials.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
import uuid
from decimal import Decimal
from pathlib import Path

import pytest

os.environ.setdefault("RAZORPAY_CLIENT", "mock")
os.environ.setdefault("RAZORPAY_WEBHOOK_SECRET", "test_webhook_secret")

from backend.app.core.config import WORKFLOW_VERSION, settings  # noqa: E402
from backend.app.db import models as M  # noqa: E402
from backend.app.domain import (  # noqa: E402
    RazorpayAPIError, State, WebhookSignatureInvalid,
)
from backend.app.services.razorpay import (  # noqa: E402
    EVENT_MAP, MockRazorpayClient, RazorpayRecoveryExecutor, WebhookReconciler,
    from_paise, reprocess_failed_webhook, to_paise, verify_checkout_signature,
    verify_webhook_signature,
)
from backend.app.services.workflow import (  # noqa: E402
    RecoveryWorkflow, new_trace_id,
)

SECRET = "test_webhook_secret"
needs_data = pytest.mark.skipif(
    not Path("data/processed/test_features.parquet").exists(),
    reason="build features first")


@pytest.fixture(scope="module", autouse=True)
def _secret():
    settings.razorpay_webhook_secret = SECRET
    settings.razorpay_client = "mock"


@pytest.fixture(scope="module")
def dbfile(tmp_path_factory):
    p = tmp_path_factory.mktemp("rzp") / "rzp_test.db"
    os.environ["DATABASE_URL"] = f"sqlite:///{p}"
    M.reset_engine()
    M.init_db(drop=True)
    return p


@pytest.fixture
def session(dbfile):
    s = M.get_session_factory()()
    yield s
    s.close()


# --------------------------------------------------------------- fixtures
def sign(body: bytes, secret: str = SECRET) -> str:
    return hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


def payment_event(event: str, order_id: str, payment_id: str,
                  amount_paise: int = 500000, **entity) -> bytes:
    ent = {"id": payment_id, "order_id": order_id, "amount": amount_paise,
           "currency": "INR", "status": "captured" if "captured" in event else "failed",
           "method": "card", "created_at": int(time.time())}
    ent.update(entity)
    return json.dumps({
        "entity": "event", "account_id": "acc_test", "event": event,
        "contains": ["payment"], "created_at": int(time.time()),
        "payload": {"payment": {"entity": ent}},
    }).encode()


def order_paid_event(order_id: str, payment_id: str, amount_paise: int = 500000) -> bytes:
    return json.dumps({
        "entity": "event", "event": "order.paid", "contains": ["payment", "order"],
        "created_at": int(time.time()),
        "payload": {
            "payment": {"entity": {"id": payment_id, "order_id": order_id,
                                   "amount": amount_paise, "status": "captured"}},
            "order": {"entity": {"id": order_id, "amount": amount_paise,
                                 "status": "paid"}},
        },
    }).encode()


def make_execution(session, state=State.AWAITING_PAYMENT, order_id=None):
    from tests.test_backend import make_context
    ctx = make_context()
    oid = f"OPP-RZP-{uuid.uuid4().hex[:6]}"
    order_id = order_id or f"order_test{uuid.uuid4().hex[:10]}"
    o = M.Opportunity(
        id=oid, opportunity_type=ctx["opportunity_type"], detected_at=M.utcnow(),
        state=state.value, workflow_version=WORKFLOW_VERSION,
        execution_mode="RAZORPAY_TEST",
        revenue_at_risk=Decimal(str(round(float(ctx["cart_value"]), 2))),
        contribution_margin_at_risk=Decimal(
            str(round(float(ctx["base_contribution_margin"]), 2))),
        current_attempt=1, trace_id=new_trace_id(), context=ctx)
    session.add(o)
    session.flush()
    ex = M.RecoveryExecution(
        execution_id=f"exe_{uuid.uuid4().hex[:12]}", opportunity_id=oid,
        attempt_number=1, action="FREE_SHIPPING",
        idempotency_key=f"k_{uuid.uuid4().hex}", execution_provider="RAZORPAY_TEST",
        status="SUBMITTED", external_order_id=order_id,
        amount=Decimal("5000.00"))
    session.add(ex)
    session.commit()
    return o, ex


def deliver(session, body: bytes, event_id: str | None = None,
            signature: str | None = None) -> dict:
    r = WebhookReconciler(session)
    res = r.receive(body, signature or sign(body), event_id or f"evt_{uuid.uuid4().hex[:12]}")
    if res["status"] == "duplicate":
        return res
    return {**res, "processed": r.process(res["inbox_id"])}


# ------------------------------------------------------------ amount units
def test_amount_conversion_uses_paise():
    assert to_paise(Decimal("5000.00")) == 500000
    assert to_paise(Decimal("99.99")) == 9999
    assert from_paise(500000) == Decimal("5000.00")


def test_paise_roundtrip_is_lossless():
    for v in ["1.00", "0.01", "12345.67", "99999.99"]:
        assert from_paise(to_paise(Decimal(v))) == Decimal(v)


# ------------------------------------------------------- signature handling
def test_valid_signature_accepted():
    body = b'{"event":"payment.captured"}'
    assert verify_webhook_signature(body, sign(body), SECRET)


def test_invalid_signature_rejected():
    body = b'{"event":"payment.captured"}'
    assert not verify_webhook_signature(body, "deadbeef", SECRET)


def test_signature_is_computed_over_raw_bytes():
    """Re-serialising the JSON changes the bytes and must invalidate the signature."""
    body = b'{"event":"payment.captured","a":1}'
    sig = sign(body)
    reserialised = json.dumps(json.loads(body), indent=2).encode()
    assert verify_webhook_signature(body, sig, SECRET)
    assert not verify_webhook_signature(reserialised, sig, SECRET)


def test_missing_signature_rejected():
    assert not verify_webhook_signature(b"{}", "", SECRET)


def test_wrong_secret_rejected():
    body = b'{"event":"payment.captured"}'
    assert not verify_webhook_signature(body, sign(body, "other_secret"), SECRET)


def test_checkout_signature_verification():
    order_id, payment_id = "order_abc", "pay_xyz"
    sig = hmac.new(SECRET.encode(), f"{order_id}|{payment_id}".encode(),
                   hashlib.sha256).hexdigest()
    assert verify_checkout_signature(order_id, payment_id, sig, SECRET)
    assert not verify_checkout_signature(order_id, payment_id, "bad", SECRET)


# ---------------------------------------------------- Scenario F: bad signature
def test_invalid_signature_causes_no_state_mutation(session):
    o, ex = make_execution(session)
    before = o.state
    body = payment_event("payment.captured", ex.external_order_id, "pay_1")
    with pytest.raises(WebhookSignatureInvalid):
        WebhookReconciler(session).receive(body, "invalid_signature", "evt_bad")
    session.expire_all()
    assert session.get(M.Opportunity, o.id).state == before
    assert session.query(M.WebhookInbox).filter_by(event_id="evt_bad").count() == 0


# ------------------------------------------------ Scenario A: successful payment
def test_payment_captured_marks_recovered(session):
    o, ex = make_execution(session)
    body = payment_event("payment.captured", ex.external_order_id, "pay_ok")
    res = deliver(session, body)
    assert res["processed"]["status"] == "recovered"
    session.expire_all()
    assert session.get(M.Opportunity, o.id).state == State.RECOVERED.value
    assert session.get(M.RecoveryOutcome, o.id) is not None


def test_order_paid_also_finalizes(session):
    o, ex = make_execution(session)
    res = deliver(session, order_paid_event(ex.external_order_id, "pay_op"))
    assert res["processed"]["status"] == "recovered"
    session.expire_all()
    assert session.get(M.Opportunity, o.id).state == State.RECOVERED.value


# ---------------------------------------------- Scenario B: failed payment
def test_payment_failed_is_not_terminal(session):
    """A failure moves to a recoverable holding state, never straight to lost."""
    o, ex = make_execution(session)
    body = payment_event("payment.failed", ex.external_order_id, "pay_f",
                         error_code="BAD_REQUEST_ERROR", error_step="payment_authentication")
    res = deliver(session, body)
    assert res["processed"]["status"] == "payment_failed"
    session.expire_all()
    assert session.get(M.Opportunity, o.id).state == State.PAYMENT_FAILED_RECOVERABLE.value


def test_payment_failure_details_persisted(session):
    o, ex = make_execution(session)
    deliver(session, payment_event(
        "payment.failed", ex.external_order_id, "pay_f2",
        error_code="GATEWAY_ERROR", error_description="bank down",
        error_source="bank", error_step="payment_authorization"))
    rec = session.query(M.PaymentFailureRecord).filter_by(opportunity_id=o.id).first()
    assert rec and rec.failure_code == "GATEWAY_ERROR" and rec.failure_source == "bank"


# ------------------------------------- Scenario C: failure then success
def test_failure_then_capture_ends_recovered(session):
    o, ex = make_execution(session)
    deliver(session, payment_event("payment.failed", ex.external_order_id, "pay_a"))
    session.expire_all()
    assert session.get(M.Opportunity, o.id).state == State.PAYMENT_FAILED_RECOVERABLE.value

    deliver(session, payment_event("payment.captured", ex.external_order_id, "pay_b"))
    session.expire_all()
    assert session.get(M.Opportunity, o.id).state == State.RECOVERED.value
    # Both events must remain visible in the audit trail.
    types = {e.event_type for e in session.query(M.AuditEvent)
             .filter_by(opportunity_id=o.id).all()}
    assert "PAYMENT_FAILED" in types and "RECOVERY_CONFIRMED" in types


# ----------------------------------------- Scenario D: duplicate webhook
def test_duplicate_event_id_is_not_reprocessed(session):
    o, ex = make_execution(session)
    body = payment_event("payment.captured", ex.external_order_id, "pay_d")
    eid = "evt_duplicate_1"
    first = deliver(session, body, event_id=eid)
    second = deliver(session, body, event_id=eid)
    assert first["processed"]["status"] == "recovered"
    assert second["status"] == "duplicate"
    assert session.query(M.WebhookInbox).filter_by(event_id=eid).count() == 1


def test_duplicate_capture_does_not_double_count_revenue(session):
    o, ex = make_execution(session)
    body = payment_event("payment.captured", ex.external_order_id, "pay_dd")
    deliver(session, body, event_id="evt_dd_1")
    # A different event id carrying the same capture must still not double count.
    deliver(session, payment_event("payment.captured", ex.external_order_id, "pay_dd"),
            event_id="evt_dd_2")
    assert session.query(M.RecoveryOutcome).filter_by(opportunity_id=o.id).count() == 1


# --------------------------------------- Scenario E: out-of-order events
def test_late_failure_cannot_regress_recovered(session):
    o, ex = make_execution(session)
    deliver(session, payment_event("payment.captured", ex.external_order_id, "pay_x"))
    session.expire_all()
    assert session.get(M.Opportunity, o.id).state == State.RECOVERED.value

    res = deliver(session, payment_event("payment.failed", ex.external_order_id, "pay_x"))
    assert res["processed"]["status"] == "ignored_terminal"
    session.expire_all()
    assert session.get(M.Opportunity, o.id).state == State.RECOVERED.value


def test_order_paid_then_capture_stays_recovered(session):
    o, ex = make_execution(session)
    deliver(session, order_paid_event(ex.external_order_id, "pay_oo"))
    deliver(session, payment_event("payment.captured", ex.external_order_id, "pay_oo"))
    session.expire_all()
    assert session.get(M.Opportunity, o.id).state == State.RECOVERED.value
    assert session.query(M.RecoveryOutcome).filter_by(opportunity_id=o.id).count() == 1


def test_regression_attempt_is_audited_not_silent(session):
    o, ex = make_execution(session)
    deliver(session, payment_event("payment.captured", ex.external_order_id, "pay_au"))
    deliver(session, payment_event("payment.failed", ex.external_order_id, "pay_au"))
    types = [e.event_type for e in session.query(M.AuditEvent)
             .filter_by(opportunity_id=o.id).all()]
    assert "AUDIT_CORRECTION" in types


# ---------------------------------------------------------- event mapping
def test_event_map_covers_required_events():
    for e in ("payment.failed", "payment.captured", "order.paid"):
        assert e in EVENT_MAP


def test_unknown_event_is_ignored_not_failed(session):
    body = json.dumps({"event": "payment.dispute.created",
                       "payload": {}}).encode()
    res = deliver(session, body)
    assert res["processed"]["status"] == "ignored"


def test_unmatched_webhook_does_not_attach_randomly(session):
    body = payment_event("payment.captured", "order_nonexistent", "pay_none")
    res = deliver(session, body)
    assert res["processed"]["status"] == "unmatched"


# ----------------------------------------------------------- inbox hygiene
def test_payload_hash_recorded(session):
    o, ex = make_execution(session)
    body = payment_event("payment.captured", ex.external_order_id, "pay_h")
    res = deliver(session, body)
    row = session.get(M.WebhookInbox, res["inbox_id"])
    assert row.payload_hash == hashlib.sha256(body).hexdigest()
    assert row.signature_valid is True


def test_reprocess_is_idempotent(session):
    o, ex = make_execution(session)
    body = payment_event("payment.captured", ex.external_order_id, "pay_r")
    res = deliver(session, body, event_id="evt_reproc")
    again = reprocess_failed_webhook(session, "evt_reproc")
    assert again["status"] == "already_processed"
    assert session.query(M.RecoveryOutcome).filter_by(opportunity_id=o.id).count() == 1


# ------------------------------------------------------------ order creation
def test_mock_order_creation_returns_safe_checkout_payload(session):
    o, _ = make_execution(session, state=State.AUTHORIZED)
    o.selected_action = "FREE_SHIPPING"
    session.commit()
    ex = M.RecoveryExecution(
        execution_id="exe_order1", opportunity_id=o.id, attempt_number=1,
        action="FREE_SHIPPING", idempotency_key="k_order1",
        execution_provider="RAZORPAY_TEST", status="PENDING")
    session.add(ex)
    session.commit()

    result = RazorpayRecoveryExecutor(MockRazorpayClient()).execute(o, ex, session)
    checkout = result["checkout"]
    assert checkout["razorpay_order_id"].startswith("order_")
    assert checkout["payment_environment"] == "TEST"
    # Secrets must never reach the browser payload.
    blob = json.dumps(checkout).lower()
    assert "secret" not in blob


def test_backend_derives_amount_not_the_client(session):
    o, _ = make_execution(session, state=State.AUTHORIZED)
    ex = M.RecoveryExecution(
        execution_id="exe_amt", opportunity_id=o.id, attempt_number=1,
        action="MEDIUM_DISCOUNT", idempotency_key="k_amt",
        execution_provider="RAZORPAY_TEST", status="PENDING")
    session.add(ex)
    session.commit()
    result = RazorpayRecoveryExecutor(MockRazorpayClient()).execute(o, ex, session)
    cart = float(o.context["cart_value"])
    assert result["amount"] < cart  # 10% discount actually applied


def test_do_nothing_creates_no_order(session):
    o, _ = make_execution(session, state=State.AUTHORIZED)
    ex = M.RecoveryExecution(
        execution_id="exe_dn", opportunity_id=o.id, attempt_number=1,
        action="DO_NOTHING", idempotency_key="k_dn",
        execution_provider="RAZORPAY_TEST", status="PENDING")
    session.add(ex)
    session.commit()
    result = RazorpayRecoveryExecutor(MockRazorpayClient()).execute(o, ex, session)
    assert result.get("order_id") is None
    assert result["terminal_state"] == "NOT_RECOVERED"


# ------------------------------------------------------- failure injection
def test_razorpay_outage_does_not_invent_success(session):
    o, _ = make_execution(session, state=State.AUTHORIZED)
    o.selected_action = "FREE_SHIPPING"
    session.commit()
    client = MockRazorpayClient()
    client.fail_next = True
    ex = M.RecoveryExecution(
        execution_id="exe_fail", opportunity_id=o.id, attempt_number=1,
        action="FREE_SHIPPING", idempotency_key="k_fail",
        execution_provider="RAZORPAY_TEST", status="PENDING")
    session.add(ex)
    session.commit()
    with pytest.raises(RazorpayAPIError):
        RazorpayRecoveryExecutor(client).execute(o, ex, session)
    session.expire_all()
    assert session.get(M.RecoveryOutcome, o.id) is None


def test_workflow_marks_execution_failed_on_outage(session):
    from tests.test_backend import make_opportunity
    o = make_opportunity(session)
    o.execution_mode = "RAZORPAY_TEST"
    session.commit()
    wf = RecoveryWorkflow(session)
    wf.analyze(o)
    session.commit()
    if o.state != State.AUTHORIZED.value:
        pytest.skip("scenario did not authorize")
    client = MockRazorpayClient()
    client.fail_next = True
    res = wf.execute(o, RazorpayRecoveryExecutor(client))
    assert res["status"] == "FAILED"
    session.expire_all()
    assert session.get(M.Opportunity, o.id).state == State.EXECUTION_FAILED.value
    assert session.get(M.RecoveryOutcome, o.id) is None


def test_live_client_refuses_non_test_key(monkeypatch):
    from backend.app.services.razorpay import RazorpayTestClient
    monkeypatch.setattr(settings, "razorpay_key_id", "rzp_live_abc123")
    monkeypatch.setattr(settings, "razorpay_key_secret", "secret")
    monkeypatch.setattr(settings, "razorpay_mode", "test")
    with pytest.raises(RazorpayAPIError, match="Test Mode"):
        RazorpayTestClient()


def test_live_mode_refused(monkeypatch):
    from backend.app.services.razorpay import RazorpayTestClient
    monkeypatch.setattr(settings, "razorpay_key_id", "rzp_test_abc")
    monkeypatch.setattr(settings, "razorpay_key_secret", "secret")
    monkeypatch.setattr(settings, "razorpay_mode", "live")
    with pytest.raises(RazorpayAPIError, match="live mode is refused"):
        RazorpayTestClient()


def test_malformed_json_after_valid_signature_is_handled(session):
    body = b"{not valid json"
    r = WebhookReconciler(session)
    with pytest.raises(Exception):
        r.receive(body, sign(body), "evt_malformed")