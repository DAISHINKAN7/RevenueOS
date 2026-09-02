# RevenueOS — Complete Demo Script

Everything to say, show and run. Two modes: **offline** (no tunnel, no keys,
fully reproducible) and **live Razorpay Test Mode**. Offline is the default and
is what you should record.

---

## Part 0 · Setup

### Offline — the default

```bash
# terminal 1
cd ~/Desktop/revenueos && source venv/bin/activate
make demo-reset
make api-demo                    # paced streaming, watchable

# terminal 2
cd frontend && npm run dev
```

All five scenarios seed in `SIMULATOR` mode. No tunnel, no Razorpay key, no
network. Everything completes.

### Live Razorpay Test Mode — optional

```bash
# terminal 3
cloudflared tunnel --url http://localhost:8000
```

The quick-tunnel hostname changes on every restart, so re-register it each time:
Razorpay Dashboard → Developers → Webhooks → edit the URL to
`https://<new-host>/api/webhooks/razorpay`, same secret as
`RAZORPAY_WEBHOOK_SECRET`, events `payment.captured`, `payment.failed`,
`order.paid`.

`.env` needs `RAZORPAY_CLIENT=test` and Test Mode keys. Then in the UI, on any
authorized opportunity, switch **Execution mode** to *Razorpay Test Mode* before
pressing Execute.

### Fallback if anything breaks mid-demo

```bash
make demo-reset          # clean state, 3 seconds
make demo-offline        # every scenario in the terminal, no UI
```

---

## Part 1 · The problem  (45s)

> "When a customer abandons a cart or a payment fails, most merchants respond
> with a discount. It works — conversions go up. But a discount is a transfer of
> margin, and if the incentive costs more than the uplift is worth, you have
> recovered revenue and lost money."

> "So the question isn't *can we recover this transaction*. It's *should we, with
> what, and does it actually make the merchant better off*."

> "RevenueOS answers that. It detects revenue at risk, predicts how each
> intervention would perform, ranks them on incremental contribution rather than
> conversion, enforces the merchant's financial policy, executes through
> Razorpay, and verifies the money actually arrived."

Show the architecture line: **ML predicts · economics ranks · policy authorizes ·
Razorpay executes · webhooks verify · audit records.**

---

## Part 2 · Why the data is honest  (90s)

This is the part most submissions skip. It is also the part that separates a
demo from a system.

### The problem with public data

> "Recovery is a counterfactual problem. To know whether free shipping was the
> right call, you need to know what would have happened with a discount, or with
> nothing. No public dataset contains that — they record what customers did, never
> what they would have done."

> "So I built a documented synthetic behavioural environment. And the risk with
> synthetic data is that you accidentally write the answer key into it."

### Four things done to prevent that

Open `docs/data-card.md` and `evaluation/results/data_validation_report.md`.

**1. Latent traits are hidden from the model.**
> "Each customer has hidden traits — price sensitivity, shipping sensitivity,
> payment friction. Those drive the true response surface, and the model never
> sees them. What it sees are finite event counts: three coupons offered, one
> redeemed. That carries real sampling noise. Measured correlation with the
> latent trait is about 0.4 — real signal, nothing like a copy."

**2. The logging policy is deliberately bad, and its propensities are recorded.**
> "Historical actions were assigned stochastically by an imperfect merchant
> heuristic — heavy on doing nothing and blanket discounting. Every row records
> the exact probability with which its action was chosen, and 25% of
> opportunities got a randomised action. Without that, 'high-LTV customers got
> free shipping and converted' proves nothing — it is confounded. With it, I can
> run doubly robust off-policy evaluation."

**3. Hidden mechanisms cap achievable accuracy.**
> "Bank outages, competitor sales, courier disruptions, payday effects. The model
> sees the consequences but never the cause. Plus shared logit noise. If a model
> came back at 0.95 AUC on this data, that would be a bug in my simulator, not a
> result."

**4. Leakage is enforced by tests, not by intention.**
Point at section 10 of the validation report: six checks, all PASS. Chronological
split, quarantined oracle, no hidden columns in the feature matrix.

### The number that proves it isn't rigged

> "Test ROC-AUC is 0.60. That's modest, and I'm reporting it rather than hiding
> it. Decisions here are expected-value comparisons, so what matters is whether
> the probabilities are calibrated and whether relative treatment response is
> right — not whether I can rank customers."

> "And here's the finding I didn't expect: a smoothed lookup table plus the
> financial engine captures about 70% of available value; the full gradient
> boosted model captures 63%. The ML is not what makes this work. The economics
> layer is. I'd rather say that than overclaim."

---

## Part 3 · The live decision  (90s) — the centrepiece

`make demo-reset` first. Open the UI.

**Overview** → four live metric cards, labelled *Live demo · Razorpay Test Mode*.

> "Everything here is computed from actual executions. Research metrics are on a
> separate page and never mixed in."

Point at the hero insight.

> "About 40% of decisions change when you optimize for contribution instead of
> conversion. That single number is the product."

**Opportunities** → all five in *Detected*.

> "Nothing has been decided yet. You're about to watch the decision happen."

**Open DEMO1 (₹4,869) → Run live analysis.**

Events stream in one at a time. Narrate as they land:

> "Each action is scored independently by the frozen model… now the financial
> engine ranks them on incremental value against doing nothing… and now a
> deterministic policy engine decides whether the winner is even allowed."

---

## Part 4 · Conversion is not contribution  (60s) — the argument

Scroll to **Conversion versus economics**.

> "Medium discount has the highest predicted recovery probability — 36.2%. Free
> shipping is lower. But look at the economics."

Then the candidate table.

> "Medium discount costs ₹486.90 as an incentive. That produces a *negative*
> incremental expected value — minus ₹48. It converts better and loses money.
> Free shipping costs ₹42 and creates ₹105 of value. That's what got selected."

> "A conversion-optimizing system picks the discount every time. That's the
> mistake this product exists to prevent."

Expand **Why not the alternatives** for per-action reasoning.

---

## Part 5 · Bounded authority  (45s)

**Policy guardrails**, right column.

> "Twelve named rules, each with its input and its threshold, all recorded."

Expand one: *Free shipping limit — ₹42.05 ≤ ₹150 — PASS*.

> "Maximum authorized downside: ₹44.05. That's the most this action can cost the
> merchant, and it's stored on the record."

> "The important line is at the bottom: the policy engine takes no model input
> and no language input. The model proposes. Only these rules authorize."

---

## Part 6 · Execute and recover  (60s)

Press **Execute recovery**. Show the confirmation — action, amount, downside,
environment.

### Offline

> "Execution mode is set to offline, so no Razorpay order is created. The
> simulated payment panel drives the same state machine, the same idempotent
> outcome booking, the same audit trail. Only the confirmation is local — and it's
> marked `provider: SIMULATOR` in the record, so a simulated recovery can never
> be mistaken for a verified one."

Press **Succeed** → **Recovered**, ₹4,869 GMV, ₹1,963.76 contribution, with a
*Simulated* badge.

### Live Razorpay

Switch **Execution mode** to *Razorpay Test Mode* before executing.

> "This creates a real Test Mode order. Checkout opens."

Pay with **Netbanking → any bank → Success** (cards trip the international-card
rejection).

> "Notice it says *payment submitted, awaiting verified webhook* — not recovered.
> The browser callback is never trusted. Only a signature-verified webhook moves
> this to recovered."

Show terminal 1 as the webhook lands, then the UI flipping.

---

## Part 7 · Adaptive retry  (75s) — the strongest technical moment

Open **DEMO3** (payment failure). Analyse → execute → choose *Card declined* →
**Fail**.

> "The payment failed. Notice the state: payment failed, *recoverable* — not
> lost. A failed payment isn't terminal, because the same order may still be
> captured."

Press **Run live analysis** again.

> "The system now has evidence it didn't have before. The provider reported an
> authorization failure, which normalizes to a payment-method problem. The
> diagnosis changes from shipping friction to payment friction, retry actions
> become eligible, and the decision moves."

```
attempt 1   DELAYED_RETRY           FAILED
attempt 2   PAYMENT_METHOD_SWITCH   CAPTURED
```

> "Two distinct idempotency keys. A repeated request cannot create a second
> order — that's enforced by a database constraint, not application logic."

Press **Succeed**.

---

## Part 8 · Restraint and approval  (45s)

**DEMO2 (₹469)** → analysis selects **Do nothing**.

> "No intervention has positive incremental value, so the system declines to
> spend. Most recovery systems cannot express this. About a third of decisions
> here are do-nothing."

**DEMO4 (₹55,116)** → **Require approval**.

> "Positive economics — ΔEV +₹235 — and policy still stops it, because the order
> exceeds the high-value threshold. Executions created: zero. Good economics are
> not sufficient authority."

---

## Part 9 · Agent safety  (60s)

`/agent`.

> "There's an LLM here, and it does exactly one thing: choose which permitted
> workflow tool runs next. It cannot set an amount, a discount, a probability, a
> policy outcome or a payment state."

Point at the metrics:

> "Zero unauthorized executions. Zero policy bypasses. And a *non-zero* blocked
> count — the gate is load-bearing, not decorative."

The state-to-tool matrix:

> "`request_execution` is reachable from exactly one state, and only the policy
> engine can produce that state. Terminal states expose read-only tools."

Prompt injection card:

> "A customer note demanding a 50% discount. It changes nothing — no 50% discount
> exists in the action space, and the policy engine accepts no text at all."

If you have it, show `make planner-compare`:

> "Three planners — a 3B local model, a hosted 20B, and no model at all. The 3B
> is genuinely bad at this and stalls. All three produce zero unauthorized
> executions. Safety is a property of the architecture, not of model quality."

---

## Part 10 · Evidence  (60s)

`/evaluation`, labelled *Synthetic held-out evaluation*.

> "Separate page, separate label. Research evidence and live operational numbers
> are never mixed."

Point at the policy comparison:

> "Flat 10% discount converts better than RevenueOS and earns less per
> opportunity. That contrast is the whole thesis, measured on held-out data."

> "The doubly robust estimate uses only logged actions, outcomes and recorded
> propensities — it never asks the simulator for a counterfactual. It agrees with
> the oracle within about 3%. Two independent routes to the same answer."

> "And I report the model metrics honestly. AUC 0.60. Calibration matters more
> than ranking here, and the environment has deliberate noise in it."

---

## Part 11 · Audit  (30s)

`/audit`.

> "Every material transition, append-only, with the model version and policy
> version in force at the time."

Expand an event.

> "For any rupee this system moved, I can tell you which model produced the
> probability, what the alternatives were, what each was worth, which rule
> authorized it, what the maximum downside was, what was sent to Razorpay, and
> which webhook confirmed it."

---

## Part 12 · Close  (30s)

> "RevenueOS doesn't ask whether a transaction *can* be recovered. It asks what
> caused the loss, which intervention creates incremental economic value, whether
> the agent is authorized to take it, and whether the money actually arrived."

> "The model can be wrong. The planner can hallucinate. A webhook can arrive
> twice or out of order. The system stays bounded through all of it — and every
> decision can be reconstructed from the audit trail."

---

## Answers to likely questions

**"Isn't the synthetic data circular?"**
> "Partly, and that's why there are three independent evidence streams. The
> doubly robust estimate uses only logged data and never queries the
> counterfactual. It agrees with the oracle within 3%. I report both."

**"Why is AUC so low?"**
> "Deliberate. Hidden mechanisms and shared noise cap it. If it were 0.95 I'd be
> looking for leakage. And decisions are expected-value comparisons — calibration
> matters more than ranking."

**"What does the LLM actually do?"**
> "Chooses the next tool. Nothing else. There's a property test asserting no
> planner-facing tool accepts an argument resembling money or authorization."

**"Is any of this real?"**
> "The Razorpay integration is real Test Mode — order creation, HMAC signature
> verification over the raw body, event-id deduplication, out-of-order
> reconciliation. There's a captured live recovery in `evaluation/results/live/`.
> The offline mode exists so the demo doesn't depend on a tunnel."

**"What would you fix first?"**
> "Payment-method-switch response isn't learned — near-zero correlation with the
> oracle, so the policy under-selects it and that costs contribution in the tail.
> It's the first thing I'd investigate."

---

## Timing

| Part | Time |
|---|---|
| Problem | 0:45 |
| Data honesty | 1:30 |
| Live decision | 1:30 |
| Conversion vs contribution | 1:00 |
| Bounded authority | 0:45 |
| Execute and recover | 1:00 |
| Adaptive retry | 1:15 |
| Restraint and approval | 0:45 |
| Agent safety | 1:00 |
| Evidence | 1:00 |
| Audit | 0:30 |
| Close | 0:30 |
| **Total** | **~11:30** |

**For a 5-minute cut:** problem (0:30), data honesty (0:45), live decision
(1:00), conversion vs contribution (0:45), adaptive retry (1:00), agent safety
(0:30), evidence (0:30). Drop restraint, approval and audit — mention them.