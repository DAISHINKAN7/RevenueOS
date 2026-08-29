# RevenueOS — Razorpay Gate Report (Phase 6)

Generated 2026-08-27T13:54:06.190219+00:00

## Test Mode status
- mode: `test`
- client: `mock`
- credentials configured: **False**
- live smoke available: **False**
- the client refuses to construct with a non-`rzp_test_*` key or with `RAZORPAY_MODE != test`

## Integration method
Official `razorpay` Python SDK, wrapped behind an internal `RazorpayClient` interface with a `MockRazorpayClient` used by CI. Verified against current Razorpay documentation (Aug 2026):

- webhook signature: HMAC-SHA256 over the **raw request body**, keyed by the webhook secret, delivered in `X-Razorpay-Signature`
- deduplication: `x-razorpay-event-id`, unique per event
- acknowledgement: 2xx within 5s; failures retried with exponential backoff for 24 hours
- ordering: explicitly not guaranteed, so reconciliation is semantic
- amounts: smallest currency unit (paise)

## Event mapping
| Razorpay event | internal |
|---|---|
| `payment.failed` | `PAYMENT_FAILED` |
| `payment.captured` | `PAYMENT_CAPTURED` |
| `payment.authorized` | `PAYMENT_AUTHORIZED` |
| `order.paid` | `ORDER_PAID` |

## Scenario results
| scenario | expected | result |
|---|---|---|
| A — payment.captured | RECOVERED | PASS |
| A — order.paid | RECOVERED | PASS |
| B — payment.failed | PAYMENT_FAILED_RECOVERABLE (not terminal) | PASS |
| C — failed then captured | RECOVERED, both audited | PASS |
| D — duplicate event id | single state effect | PASS |
| E — late failure after capture | state preserved | PASS |
| E — order.paid then capture | one outcome row | PASS |
| F — invalid signature | no state mutation, nothing persisted | PASS |
| G — concurrent execute | exactly one order | PASS |
| Razorpay outage | EXECUTION_FAILED, no invented success | PASS |
| unknown event | stored, IGNORED, 200 | PASS |
| unmatched order id | UNMATCHED, no random attachment | PASS |

## Reconciliation rules
- `RECOVERED` is absorbing. A `payment.failed` arriving after a capture is recorded as `AUDIT_CORRECTION` and ignored for state.
- `payment.failed` moves to `PAYMENT_FAILED_RECOVERABLE`, never straight to `NOT_RECOVERED`, because the same journey may still be captured.
- Correlation is strict: `notes.execution_id` first, then `order_id`, then `payment_id`. No heuristic attachment.
- `RecoveryOutcome` has the opportunity id as its primary key, so recovery can only be counted once regardless of how many events arrive.

## Security
- signature verified before any parsing or persistence
- invalid signature: 400, no inbox row, no state change
- write endpoints require `X-Admin-Token`
- checkout payload contains key id, order id, amount, currency and display name only — never a secret
- amount is derived server-side from cart and approved discount; a client-supplied amount is never authoritative

## Tests

- `tests/test_razorpay.py`: **32 passed, 0 failed**

## Live smoke result
- not run: credentials absent or `RAZORPAY_CLIENT=mock`. Run `python -m scripts.razorpay_smoke_test` with Test Mode keys.

## Known limitations
- Payment Links are not implemented; the Orders + Standard Checkout flow was prioritised as the spec directs.
- Replay protection relies on event-id deduplication rather than a timestamp window, deliberately: Razorpay retries legitimately for up to 24 hours and a strict window would reject valid deliveries.
- Provider fetch reconciliation exists in the client but is not yet wired into an automatic conflict-resolution path.
