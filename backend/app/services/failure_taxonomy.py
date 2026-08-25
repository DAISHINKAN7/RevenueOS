"""Normalized payment-failure taxonomy (spec Part A §5).

Provider error codes are vendor-specific and change over time. The recovery
logic reasons over a small, stable set of *internal* categories, so a new
Razorpay error code cannot silently alter recovery behaviour — it maps to
UNKNOWN and the system degrades safely instead of guessing.

Two distinct vocabularies exist and must not be confused:

* `FailureCategory` — coarse internal semantics used by the adaptive layer.
* the synthetic `failure_reason` values the frozen model was trained on
  (BANK_TIMEOUT, CARD_DECLINED, ...). The model only ever sees these.
"""

from __future__ import annotations

from enum import Enum

TAXONOMY_VERSION = "failure-taxonomy-1.0.0"


class FailureCategory(str, Enum):
    PAYMENT_INFRASTRUCTURE_FAILURE = "PAYMENT_INFRASTRUCTURE_FAILURE"
    PAYMENT_METHOD_FAILURE = "PAYMENT_METHOD_FAILURE"
    AUTHENTICATION_FAILURE = "AUTHENTICATION_FAILURE"
    CUSTOMER_FUNDS_FAILURE = "CUSTOMER_FUNDS_FAILURE"
    CUSTOMER_ABORT = "CUSTOMER_ABORT"
    UNKNOWN_PAYMENT_FAILURE = "UNKNOWN_PAYMENT_FAILURE"


# Synthetic (model-vocabulary) reason -> internal category.
REASON_TO_CATEGORY: dict[str, FailureCategory] = {
    "BANK_TIMEOUT": FailureCategory.PAYMENT_INFRASTRUCTURE_FAILURE,
    "NETWORK_ERROR": FailureCategory.PAYMENT_INFRASTRUCTURE_FAILURE,
    "UPI_TIMEOUT": FailureCategory.PAYMENT_INFRASTRUCTURE_FAILURE,
    "CARD_DECLINED": FailureCategory.PAYMENT_METHOD_FAILURE,
    "AUTHENTICATION_FAILURE": FailureCategory.AUTHENTICATION_FAILURE,
    "INSUFFICIENT_FUNDS": FailureCategory.CUSTOMER_FUNDS_FAILURE,
    "USER_CANCELLED": FailureCategory.CUSTOMER_ABORT,
    "UNKNOWN": FailureCategory.UNKNOWN_PAYMENT_FAILURE,
}

# Razorpay `error_step` is the most informative field available, so it is
# consulted first; `error_code` is the fallback. Both map into the model's
# vocabulary, which then maps into the internal category above.
STEP_TO_REASON: dict[str, str] = {
    "payment_authentication": "AUTHENTICATION_FAILURE",
    "payment_authorization": "CARD_DECLINED",
    "payment_initiation": "NETWORK_ERROR",
    "payment_capture": "BANK_TIMEOUT",
}

CODE_TO_REASON: dict[str, str] = {
    "GATEWAY_ERROR": "BANK_TIMEOUT",
    "NETWORK_ERROR": "NETWORK_ERROR",
    "SERVER_ERROR": "BANK_TIMEOUT",
    "BAD_REQUEST_ERROR": "UNKNOWN",
}

# Razorpay reports customer dismissal through these; treat as an abort signal.
ABORT_CODES = {"PAYMENT_CANCELLED", "USER_CANCELLED"}


def normalize_failure(error_code: str | None, error_step: str | None,
                      description: str | None = None) -> tuple[str, FailureCategory]:
    """Map provider fields to (model_reason, internal_category).

    Unrecognised input deliberately returns UNKNOWN rather than a nearest
    guess: a wrong category would steer the adaptive layer toward the wrong
    family of recovery actions.
    """
    code = (error_code or "").upper()
    step = (error_step or "").lower()

    if code in ABORT_CODES:
        return "USER_CANCELLED", FailureCategory.CUSTOMER_ABORT

    reason = STEP_TO_REASON.get(step) or CODE_TO_REASON.get(code) or "UNKNOWN"

    # Razorpay frequently returns BAD_REQUEST_ERROR with the real cause only in
    # the description. Read it, but only for unambiguous phrases.
    if reason == "UNKNOWN" and description:
        d = description.lower()
        if "insufficient" in d:
            reason = "INSUFFICIENT_FUNDS"
        elif "cancel" in d or "abort" in d:
            reason = "USER_CANCELLED"
        elif "declin" in d:
            reason = "CARD_DECLINED"
        elif "timeout" in d or "timed out" in d:
            reason = "BANK_TIMEOUT"

    return reason, REASON_TO_CATEGORY.get(reason, FailureCategory.UNKNOWN_PAYMENT_FAILURE)