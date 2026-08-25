"""Workflow state machine and error taxonomy.

The state machine is the spine of the system: every financial side effect is
gated on a legal transition, so a bug that tries to shortcut from DETECTED to
RECOVERED fails loudly instead of silently booking revenue.
"""

from __future__ import annotations

from enum import Enum


class State(str, Enum):
    DETECTED = "DETECTED"
    ANALYZING = "ANALYZING"
    CANDIDATES_SCORED = "CANDIDATES_SCORED"
    ECONOMICALLY_RANKED = "ECONOMICALLY_RANKED"
    POLICY_CHECKED = "POLICY_CHECKED"
    AWAITING_APPROVAL = "AWAITING_APPROVAL"
    AUTHORIZED = "AUTHORIZED"
    EXECUTION_PENDING = "EXECUTION_PENDING"
    EXECUTING = "EXECUTING"
    AWAITING_PAYMENT = "AWAITING_PAYMENT"
    PAYMENT_FAILED_RECOVERABLE = "PAYMENT_FAILED_RECOVERABLE"
    RECOVERED = "RECOVERED"
    NOT_RECOVERED = "NOT_RECOVERED"
    STOPPED = "STOPPED"
    ESCALATED = "ESCALATED"
    EXPIRED = "EXPIRED"
    EXECUTION_FAILED = "EXECUTION_FAILED"


TERMINAL_STATES = frozenset({
    State.RECOVERED, State.NOT_RECOVERED, State.STOPPED,
    State.EXPIRED, State.ESCALATED,
})

# RECOVERED is absorbing: a stale payment.failed arriving after a capture must
# never downgrade it (Razorpay does not guarantee webhook ordering).
ABSORBING_STATES = frozenset({State.RECOVERED})

ALLOWED_TRANSITIONS: dict[State, frozenset[State]] = {
    State.DETECTED: frozenset({State.ANALYZING, State.EXPIRED, State.STOPPED}),
    State.ANALYZING: frozenset({State.CANDIDATES_SCORED, State.STOPPED,
                                State.ESCALATED, State.EXECUTION_FAILED}),
    State.CANDIDATES_SCORED: frozenset({State.ECONOMICALLY_RANKED, State.STOPPED}),
    State.ECONOMICALLY_RANKED: frozenset({State.POLICY_CHECKED, State.STOPPED}),
    State.POLICY_CHECKED: frozenset({State.AUTHORIZED, State.AWAITING_APPROVAL,
                                     State.STOPPED, State.ESCALATED,
                                     State.NOT_RECOVERED}),
    State.AWAITING_APPROVAL: frozenset({State.AUTHORIZED, State.STOPPED, State.EXPIRED}),
    State.AUTHORIZED: frozenset({State.EXECUTION_PENDING, State.STOPPED}),
    State.EXECUTION_PENDING: frozenset({State.EXECUTING, State.EXECUTION_FAILED, State.STOPPED}),
    State.EXECUTING: frozenset({State.AWAITING_PAYMENT, State.RECOVERED,
                                State.NOT_RECOVERED, State.EXECUTION_FAILED}),
    State.AWAITING_PAYMENT: frozenset({State.RECOVERED, State.NOT_RECOVERED,
                                       State.PAYMENT_FAILED_RECOVERABLE, State.EXPIRED}),
    State.PAYMENT_FAILED_RECOVERABLE: frozenset({State.ANALYZING, State.STOPPED,
                                                 State.RECOVERED, State.NOT_RECOVERED,
                                                 State.EXPIRED}),
    State.EXECUTION_FAILED: frozenset({State.ANALYZING, State.STOPPED, State.ESCALATED}),
    # Terminal
    State.RECOVERED: frozenset(),
    State.NOT_RECOVERED: frozenset({State.ANALYZING}),
    State.STOPPED: frozenset(),
    State.ESCALATED: frozenset({State.AUTHORIZED, State.STOPPED}),
    State.EXPIRED: frozenset(),
}


def can_transition(src: State, dst: State) -> bool:
    if src in ABSORBING_STATES:
        return False
    return dst in ALLOWED_TRANSITIONS.get(src, frozenset())


# --------------------------------------------------------------- error taxonomy
class RevenueOSError(Exception):
    code = "REVENUEOS_ERROR"
    http_status = 400

    def __init__(self, message: str = "", **context):
        super().__init__(message or self.code)
        self.message = message or self.code
        self.context = context

    def to_dict(self) -> dict:
        return {"error_code": self.code, "message": self.message, "context": self.context}


class ModelVersionMismatch(RevenueOSError):
    code = "MODEL_VERSION_MISMATCH"
    http_status = 503


class FeatureSchemaMismatch(RevenueOSError):
    code = "FEATURE_SCHEMA_MISMATCH"
    http_status = 503


class InvalidModelOutput(RevenueOSError):
    code = "INVALID_MODEL_OUTPUT"
    http_status = 422


class PolicyRejected(RevenueOSError):
    code = "POLICY_REJECTED"
    http_status = 403


class ApprovalRequired(RevenueOSError):
    code = "APPROVAL_REQUIRED"
    http_status = 409


class InvalidStateTransition(RevenueOSError):
    code = "INVALID_STATE_TRANSITION"
    http_status = 409


class DuplicateExecution(RevenueOSError):
    code = "DUPLICATE_EXECUTION"
    http_status = 409


class RazorpayAPIError(RevenueOSError):
    code = "RAZORPAY_API_ERROR"
    http_status = 502


class WebhookSignatureInvalid(RevenueOSError):
    code = "WEBHOOK_SIGNATURE_INVALID"
    http_status = 400


class WebhookDuplicate(RevenueOSError):
    code = "WEBHOOK_DUPLICATE"
    http_status = 200


class PaymentStateConflict(RevenueOSError):
    code = "PAYMENT_STATE_CONFLICT"
    http_status = 409


class OpportunityExpired(RevenueOSError):
    code = "OPPORTUNITY_EXPIRED"
    http_status = 410


class UnmatchedWebhook(RevenueOSError):
    code = "UNMATCHED_WEBHOOK"
    http_status = 200