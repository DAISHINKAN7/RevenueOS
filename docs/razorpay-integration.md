# Razorpay Test Mode Integration

Verified against current Razorpay documentation (August 2026). Test Mode only —
the client refuses to construct with a non-`rzp_test_*` key or `RAZORPAY_MODE`
other than `test`.

## Setup

Set in `.env` (never committed):

```
RAZORPAY_KEY_ID=rzp_test_xxxxxxxx
RAZORPAY_KEY_SECRET=
RAZORPAY_WEBHOOK_SECRET=
RAZORPAY_MODE=test
RAZORPAY_CLIENT=test        # 'mock' for CI
```

Dashboard → Developers → Webhooks → add `https://<host>/api/webhooks/razorpay`,
subscribe to `payment.captured`, `payment.failed`, `order.paid`. Local testing
needs a tunnel (ngrok or equivalent) because the endpoint must be publicly
reachable.

## Order creation

Amounts are in the smallest currency unit (paise), so ₹5,000 is `500000`. The
backend derives the amount from cart minus approved discount; a client-supplied
amount is never authoritative. `notes` carries `opportunity_id` and
`execution_id` for webhook correlation — no customer data.

## Signature verification

HMAC-SHA256 over the **raw request body**, keyed by the webhook secret,
delivered in `X-Razorpay-Signature`. The body must not be parsed or
re-serialised first; `test_signature_is_computed_over_raw_bytes` asserts that
re-serialising invalidates the signature. Comparison is constant-time. Invalid
signature returns 400 with no inbox row and no state change.

The Standard Checkout callback signature is a different computation —
HMAC over `order_id|payment_id` — and is implemented separately. Browser success
UI alone never marks recovery; only a verified server-side event does.

## Deduplication

`x-razorpay-event-id` is unique per event and is the deduplication key, enforced
by `UNIQUE(provider, event_id)`. A duplicate returns 200 without reprocessing.

## Acknowledgement

Razorpay requires a 2xx within 5 seconds and retries with exponential backoff
for 24 hours. The handler verifies, persists, commits, and returns; processing
happens after that commit so slow work never blocks the acknowledgement.

## Ordering and reconciliation

Razorpay does not guarantee ordering. Reconciliation is therefore semantic:

- `RECOVERED` is absorbing — a late `payment.failed` is recorded as
  `AUDIT_CORRECTION` and ignored for state.
- `payment.failed` moves to `PAYMENT_FAILED_RECOVERABLE`, never directly to
  `NOT_RECOVERED`, because the same journey may still be captured.
- Correlation order: `notes.execution_id`, then `order_id`, then `payment_id`.
  No match produces `UNMATCHED_WEBHOOK` rather than a guess.

## Replay guard

Deliberately event-id based rather than timestamp-window based. Razorpay retries
legitimately for up to 24 hours, and a strict time window would reject valid
deliveries.

## Failure handling

API timeout, 5xx or auth error → `EXECUTION_FAILED`, never an invented success.
`create_order` is not blindly retried; the execution record is reconciled first
so a retry cannot produce a second order.