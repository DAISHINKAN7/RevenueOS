# RevenueOS — Behavioural Simulator

Implementation: `ml/simulation/`. Every constant lives in `ml/config.py`.

---

## Generation pipeline

```
build_environment()      hidden windows: outages, sales, courier, payday
        ↓
generate_products()      60 SKUs, weight-based shipping cost
        ↓
generate_customers()     observable frame + hidden latent frame
        ↓
generate_sessions()      sessions → checkouts → payment attempts
        ↓
build_opportunities()    abandoned checkouts + failed payments
        ↓
response_surface()       P(recovery | context, action) for ALL actions
        ↓
assign_actions()         stochastic logged action + stored propensity
        ↓
outcomes                 Bernoulli draw against the taken action's true P
```

Everything is seeded through a single `numpy.random.Generator`. Same seed →
byte-identical parquet output.

---

## The response surface

Recovery probability is built in logit space, then decayed in probability space:

```
z = BASE_LOGIT
  + 1.30 * brand_loyalty
  + 0.55 * impulsivity
  - 0.85 * payment_friction
  - 0.28 * log1p(cart_value / 3000)
  - 2.20 * shipping_sensitivity * clip(fee_ratio, 0, 0.25)
  - 0.60 * in_competitor_sale
  + 0.35 * is_payday
  - 0.30 * in_bank_outage
  - 0.40 * max(0, attempt_number - 1)
  + shared_noise ~ N(0, 0.45)

P(recovery | action) = sigmoid(z + uplift(action)) * decay(minutes)
```

The noise draw is **shared across actions within a row**, so it represents
unobserved context rather than independent per-action measurement error. Relative
action ordering stays meaningful while the absolute level stays unpredictable.

### Action uplifts

| Action | Uplift logic |
|---|---|
| `DO_NOTHING` | 0 by definition — the incremental baseline |
| `FREE_SHIPPING` | `0.45 + 3.10*shipping_sens + 6.0*fee_ratio`, **zero if no fee charged** |
| `SMALL_DISCOUNT` | `0.30 + 2.05*price_sens` |
| `MEDIUM_DISCOUNT` | `1.75 ×` the small-discount uplift (diminishing returns) |
| `PAYMENT_METHOD_SWITCH` | Keyed on failure reason; strongest on `CARD_DECLINED` |
| `IMMEDIATE_RETRY` | Keyed on failure reason; **negative** on `INSUFFICIENT_FUNDS`, heavily penalised during a bank outage |
| `DELAYED_RETRY` | Strongest on `BANK_TIMEOUT`; *boosted* during an outage |
| `PAYMENT_LINK` | `0.55 + 0.70*impulsivity - 0.30*payment_friction` |
| `HUMAN_ESCALATION` | Flat 1.45 — effective everywhere, but ₹45 fixed cost makes it uneconomic on small carts |

### Time decay

Multiplicative, with cause-specific half-lives:

- `CHECKOUT_ABANDONMENT` → 90 minutes (intent evaporates fast)
- `PAYMENT_FAILURE` → 240 minutes (a bank-side problem stays recoverable longer)

---

## Verified structure (seed 42)

True ΔP(recovery) vs `DO_NOTHING`, by segment:

| segment | FREE_SHIPPING | SMALL_DISCOUNT | PAYMENT_METHOD_SWITCH |
|---|---:|---:|---:|
| CONVENIENCE_SENSITIVE | **0.270** | 0.205 | 0.082 |
| DEAL_SEEKER | 0.241 | **0.399** | 0.067 |
| PRICE_SENSITIVE | 0.212 | **0.373** | 0.068 |
| PAYMENT_FRICTION | 0.201 | 0.231 | **0.130** |
| LOYAL | 0.159 | 0.167 | 0.077 |
| HIGH_LTV | 0.171 | 0.149 | 0.084 |

Each segment responds most to the intervention matching its latent trait, and
loyal/high-LTV customers show the lowest uplift everywhere — they largely
recover unaided. This is the learnable signal.

---

## The core thesis, visible in the generated data

Comparing the **conversion-maximising** action against the **ΔEV-maximising**
action across 1,500 sampled opportunities:

| Action | Conversion-max share | Economics-max share |
|---|---:|---:|
| MEDIUM_DISCOUNT | **65.7%** | 24.1% |
| FREE_SHIPPING | 21.9% | 33.1% |
| HUMAN_ESCALATION | 9.9% | 29.4% |
| DELAYED_RETRY | 1.6% | 4.7% |
| SMALL_DISCOUNT | 0.0% | 4.3% |

**The two objectives disagree on 48.4% of opportunities.** A system optimising
conversion would reach for a 10% discount two-thirds of the time; optimising
incremental contribution picks it less than a quarter of the time.

---

## Known tuning issue

`HUMAN_ESCALATION` is selected as economics-max in 29% of cases, which is higher
than a real merchant would tolerate. Its ₹45 fixed cost is too cheap relative to
its flat 1.45 logit uplift. Recommended fix before Phase 3: raise
`HUMAN_ESCALATION.fixed_cost` toward ₹120–150 and/or make its uplift depend on
cart value. Recorded here rather than silently patched, per the scientific
integrity rule.

---

## Anti-leakage rules

1. No `hidden_*` column may appear in any model feature matrix.
2. `oracle.parquet` is quarantined and used only for Evaluation Stream C.
3. Historical response features use only events before the decision timestamp.
4. Splits are chronological; the test fold is touched only for final evaluation.

Enforced in `tests/test_simulator.py` and section 10 of the validation report.
