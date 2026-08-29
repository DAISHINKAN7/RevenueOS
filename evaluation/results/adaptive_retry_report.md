# RevenueOS — Adaptive Retry Gate Report (Part A)

Generated 2026-08-27T13:56:07.806177+00:00

Rules `adaptive-recovery-rules-1.0.0` · taxonomy `failure-taxonomy-1.0.0`

## Diagnosis of the original defect

Attempt 2 previously reproduced attempt 1 exactly. The cause was not the retry
mechanics — those were correct — but the decision context. `opp.context` was a
snapshot written once at detection and never refreshed, while
`attempt_number`, `minutes_since_event`, `failure_reason`, `payment_method` and
`opportunity_type` are all live model features. The model therefore re-scored an
unchanged input and unsurprisingly returned unchanged probabilities.

Two fixes, kept deliberately separate:

1. **Context refresh** — recompute the live features from current state before
   re-analysis. This uses the frozen model exactly as trained; nothing is
   retrained or mutated.
2. **Adaptive Recovery Adjustment Layer** — a deterministic, versioned,
   bounded post-model layer that encodes retry semantics the model has no
   vocabulary for ("this action already failed", "the blocker is now payment").
   It is reported separately from the model probability and never disguised as
   a learned effect.

## Attempt 1 (state DETECTED)

Selected **FREE_SHIPPING**

| action | P | base P | adj | incentive | dEV |
|---|---:|---:|---:|---:|---:|
| FREE_SHIPPING | 0.5187 | 0.5187 | +0.000 | 44.18 | 62.20 |
| DO_NOTHING | 0.3256 | 0.3256 | +0.000 | 0.00 | 0.00 |
| PAYMENT_LINK | 0.3256 | 0.3256 | +0.000 | 0.00 | -3.00 |
| SMALL_DISCOUNT | 0.3978 | 0.3978 | +0.000 | 84.95 | -3.23 |
| MEDIUM_DISCOUNT | 0.4199 | 0.4199 | +0.000 | 169.90 | -30.80 |
| HUMAN_ESCALATION | 0.3256 | 0.3256 | +0.000 | 0.00 | -130.00 |

## New provider evidence

| field | value |
|---|---|
| error code | `BAD_REQUEST_ERROR` |
| error step | `payment_authorization` |
| normalized reason | `CARD_DECLINED` |
| internal category | `PAYMENT_METHOD_FAILURE` |
| previous action | `FREE_SHIPPING` |
| active incentives | `['FREE_SHIPPING']` |
| cumulative recovery cost | INR 2.0 |

## Attempt 2 (attempt 2)

Selected **PAYMENT_METHOD_SWITCH** — changed from attempt 1

| action | P | base P | adj | incentive | dEV |
|---|---:|---:|---:|---:|---:|
| PAYMENT_METHOD_SWITCH | 0.4475 | 0.3275 | +0.120 | 0.00 | 49.60 |
| FREE_SHIPPING | 0.4263 | 0.4863 | -0.060 | 0.00 | 40.03 |
| PAYMENT_LINK | 0.3931 | 0.3331 | +0.060 | 0.00 | 24.06 |
| DELAYED_RETRY | 0.3451 | 0.3951 | -0.050 | 0.00 | 4.44 |
| DO_NOTHING | 0.3331 | 0.3331 | +0.000 | 0.00 | 0.00 |
| SMALL_DISCOUNT | 0.3338 | 0.3738 | -0.040 | 84.95 | -30.04 |

## Why the decision moved

The observed blocker was reclassified as `PAYMENT_METHOD_FAILURE`. The adaptive layer raises the relative priority of actions that address that blocker and applies a repeat penalty to the action that already failed. The financial engine then re-ranks on incremental expected value exactly as before — the adjustment changes inputs, never the ranking rule.

## Acceptance

| check | result |
|---|---|
| attempt-2 context differs from attempt-1 | PASS |
| difference visible in structured audit | PASS (`ADAPTIVE_ADJUSTMENT_APPLIED`) |
| provider evidence affects reasoning | PASS |
| second action may differ for justified reasons | PASS (changed) |
| no hard-coded demo outcome | PASS (rules are category-driven) |
| active incentive not double-counted | PASS |
| cumulative downside tracked | PASS |
| policy re-runs on every attempt | PASS |
| idempotency key changes per attempt | PASS |
| tests | 65 passed, 0 failed |

## Known limitations

- The adjustment magnitudes are expert priors, not measured effects. They are bounded at ±0.12 and reported separately so they cannot be mistaken for model output.
- A future model version could learn these directly from retry data; that would require regenerating the simulator with multi-attempt episodes, which would invalidate the current frozen evaluation.
