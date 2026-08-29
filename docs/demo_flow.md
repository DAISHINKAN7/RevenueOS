# Demo Flow

Two terminals, then follow the sequence. Total run time about four minutes.

```bash
# terminal 1 — backend
make demo-reset && make api

# terminal 2 — frontend
cd frontend && npm run dev
```

Open `http://localhost:3000`.

## 1 · Overview (20s)

Headline metrics are live operational values. The hero insight states the
product thesis: a percentage of decisions change when optimizing for
contribution instead of conversion.

## 2 · Opportunity detail — economics beats conversion (60s)

Open the free-shipping demo. Point at **Conversion versus economics**: the
highest-converting action and the best economic action are different, side by
side. Then the candidate table — `MEDIUM_DISCOUNT` has the highest recovery
probability *and* a negative ΔEV, because its incentive costs more than the
uplift is worth.

Expand **Why not the alternatives** for the per-action reasoning.

## 3 · Policy guardrails (30s)

Right column. Every rule with its input and threshold; expand one to show the
comparison. Maximum authorized downside is the fintech framing: the most this
action can cost.

## 4 · Execute through Razorpay (60s)

On an authorized Test Mode opportunity, press **Execute recovery**. Confirmation
shows action, amount, downside and environment. Checkout opens. After paying,
the panel reads "payment submitted — awaiting verified webhook" and only flips
to recovered once the backend confirms. Say this out loud: the browser callback
is not trusted.

## 5 · Adaptive retry (30s)

Open the payment-failure demo after a failed attempt. **Strategy changed after
new evidence** shows attempt 1 and attempt 2 side by side with the provider
failure between them, and two distinct idempotency keys.

## 6 · Human approval (20s)

The high-value opportunity sits in `AWAITING_APPROVAL` with zero executions.
Positive economics are not sufficient — policy requires a human above the
threshold.

## 7 · Agent safety (40s)

Unauthorized executions and policy bypasses at zero, alongside a non-zero
blocked-call count — the gate is load-bearing, not decorative. The state-to-tool
matrix shows `request_execution` reachable from exactly one state.

## 8 · Evaluation (30s)

Held-out research results, clearly labelled as synthetic and separate from the
live numbers. Net contribution per opportunity across policies; flat discount
converts better and earns less.