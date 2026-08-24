# RevenueOS — Data Card

Status: Phase 2 complete. Regenerate metrics with `make data && make validate`.

---

## 1. Dataset purpose

RevenueOS requires customer, checkout, payment and intervention-response data to
estimate which recovery intervention produces the greatest incremental economic
value.

Public transaction datasets do not contain the counterfactual recovery outcomes
required for recovery-policy learning: they record what customers did, never
what they *would have done* under a different intervention. The project
therefore combines three data layers with clearly separated roles.

---

## 2. Data sources

| Layer | Source | Role | NOT used for |
|---|---|---|---|
| A | UCI Online Retail II (public) | Distribution calibration only | Any intervention response |
| B | RevenueOS behavioural simulator | ML training, validation, held-out test, oracle analysis | Claims about real merchants |
| C | Razorpay Test Mode | Integration proof | Model training |

### Layer A — scope of the calibration claim

Only two moments are borrowed from public retail data:

1. the **shape** of the order-value distribution (log-normal with a long right tail)
2. the **concentration curve** of customer activity (a minority of customers
   generate the majority of sessions)

Category mix, absolute price points, margin structure, shipping economics and
all payment behaviour are **constructed for an Indian consumer-electronics
merchant**, not borrowed. UCI Online Retail is a UK gift wholesaler and does not
transfer beyond those two shape parameters. This narrow claim is deliberate.

### Layer C — scope

Razorpay Test Mode supplies real order IDs, payment IDs, statuses, webhook
events and signatures for a small number of live demo transactions. Test Mode
event frequencies do not represent production payment behaviour and are never
aggregated into evaluation metrics.

---

## 3. Generated tables

| File | Rows (seed 42) | Notes |
|---|---:|---|
| `products.parquet` | 60 | Includes weight and zone-varying shipping cost |
| `customers.parquet` | 8,000 | **Observable only** — safe for the model |
| `customers_hidden.parquet` | 8,000 | Latent traits — **simulator only** |
| `sessions.parquet` | 120,000 | |
| `checkouts.parquet` | 70,087 | |
| `payment_attempts.parquet` | 44,066 | |
| `opportunities.parquet` | 36,265 | Chronologically sorted + split label |
| `interventions.parquet` | 36,265 | Logged action, propensity, outcome |
| `oracle.parquet` | 36,265 | Counterfactual P(recovery) for every action |

`oracle.parquet` is quarantined by filename and by automated leakage tests. It
exists solely for Evaluation Stream C and must never enter a feature matrix.

---

## 4. Hidden variables (never exposed to the model)

`hidden_price_sensitivity`, `hidden_shipping_sensitivity`,
`hidden_payment_friction`, `hidden_retry_tolerance`, `hidden_brand_loyalty`,
`hidden_impulsivity`, `hidden_return_propensity`,
`hidden_cancellation_propensity`.

Enforced by `tests/test_simulator.py::test_hidden_traits_never_enter_observable_frame`.

### Finite-history response features

Observable response features are **not** copies of latent traits. They are
finite event counts:

```
coupon_offers_seen      = 3
coupon_offers_redeemed  = 1
coupon_response_rate    = 0.333
```

Redemptions are Binomial draws against the latent trait, so the observable rate
is a noisy estimate carrying genuine sampling error. Measured correlation with
the latent trait is ~0.4 — real signal, far from a copy. New customers have
`NaN` rates and near-zero observation counts, so their priors are genuinely
uninformative.

---

## 5. Hidden environmental mechanisms

Windows that shift behaviour with **no corresponding feature**:

- 6 bank outage windows (9h each) — elevate `BANK_TIMEOUT`, suppress immediate retry
- 4 competitor flash sales (3d each) — depress conversion and unaided recovery
- 3 courier disruptions (4d each) — raise fulfilment cost by 35%
- payday effects on days 1–3 and 28–30

Plus shared logit noise (`response_logit_noise_sd = 0.45`) on the response
surface. Together these cap achievable predictive accuracy. **If a trained model
reaches test ROC-AUC above ~0.85, treat it as a simulator defect, not a win.**

---

## 6. Historical logging policy and exploration

The historical policy is deliberately **imperfect** — heavy on doing nothing and
on blanket discounting, mimicking a real merchant's heuristics (discount big
carts, retry on timeouts, escalate rarely).

Two properties make off-policy evaluation valid:

1. **Stored propensities.** Every logged row records the exact
   `P(action | context)` with which its action was chosen, plus the full
   propensity vector over all eligible actions.
2. **A ~25% randomised exploration cohort** where the action is drawn from a
   fixed, context-independent weighting over the eligible action set
   (`EXPLORATION_WEIGHTS`). Retry actions are over-sampled because they are
   eligible only for payment failures and would otherwise carry too little
   held-out support. They are never assigned to checkout abandonment.

Approximately **25%** of historical decisions belong to a stochastic exploration
cohort, rather than the 15–20% originally sketched, because
the headline doubly robust estimate is computed on the held-out 15% of
opportunities. At 3,000 total opportunities and 15% exploration, the held-out
cohort would have carried ~10 samples per action — far too thin to bootstrap.

Every action retains at least `min_propensity = 0.02`, so importance weights
stay finite (observed max weight 51).

### Why exploration is necessary

If high-LTV customers historically always received free shipping and converted,
observing (high LTV + free shipping + conversion) does **not** establish that
free shipping caused anything. The logged policy is confounded with context.
The randomised cohort guarantees support for action/context combinations the
greedy historical policy would never produce.

---

## 7. Splits

Chronological (`train_frac=0.70`, `validation_frac=0.15`, remainder test).
Verified by `_split_ordered` in the validation report: every TRAIN timestamp
precedes every VALIDATION timestamp, which precedes every TEST timestamp.

The test set must not be used for feature selection, hyperparameter tuning,
threshold tuning, policy selection or calibration fitting.

---

## 7b. Funnel interpretation

Session-to-checkout (~58%) and attempt-to-failure (~23%) rates are **not**
generic website conversion rates. The simulator represents commerce sessions
that have already demonstrated substantial purchase intent — a high-intent
cohort, not all anonymous merchant traffic. This oversampling is deliberate: it
ensures adequate recovery-opportunity volume for policy learning and off-policy
evaluation. Do not read `checkouts started / sessions` as a site-wide
conversion rate.

## 8. Known limitations

- Synthetic behaviour may not reflect production merchants.
- Response surfaces are modelled assumptions, not measured effects.
- Public retail data calibrates only order-value shape and activity concentration.
- Test Mode events do not represent production payment frequencies.
- Off-policy estimates still depend on overlap and modelling assumptions; the
  validation report prints Kish ESS and max importance weight for this reason.
- Results are **not** real-world causal proof.
- All actions now carry >70 held-out exploration observations after the
  revision 1.1.0 rescale; earlier revisions had thin `DELAYED_RETRY` support.
- All customer identities are synthetic. No real personal or financial data is
  used anywhere in this project.
