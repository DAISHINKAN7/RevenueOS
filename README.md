# RevenueOS

**Autonomous Revenue Recovery for Intelligent Commerce**

RevenueOS detects where ecommerce revenue is slipping away, diagnoses why,
estimates which recovery intervention creates the greatest **incremental
contribution margin**, enforces merchant financial policy, executes through
Razorpay-compatible workflows, and verifies whether the money was recovered.

> Most recovery systems ask whether a transaction *can* be recovered.
> RevenueOS asks what caused the loss, which intervention creates incremental
> economic value, whether the agent is authorized to take it, and whether the
> money was actually recovered.

**Status: Phases 1–2 complete** — financial engine and synthetic behavioural
environment, both tested and reproducible.

---

## Quick start

```bash
pip install -r requirements.txt
make gate
```

`make gate` generates the dataset, runs the validation report, and runs the test
suite. Takes about 15 seconds end to end.

| Command | What it does |
|---|---|
| `make data` | Generate the dataset (`SEED=42` by default) |
| `make data-small` | Fast 4k-session dataset for iteration |
| `make validate` | Data validation report → `evaluation/results/` |
| `make test` | Full test suite (58 tests) |
| `make gate` | All three — run after any simulator change |
| `make api` | FastAPI dev server on :8000 |
| `make data SEED=7` | Regenerate under a different seed |

---

## The key insight

Recovered revenue is not recovered profit. A blanket discount reliably increases
conversions **and** can destroy contribution margin.

That is not a hypothetical here — it is measurable in the generated data.
Comparing the conversion-maximising action against the ΔEV-maximising action
across 1,500 sampled opportunities:

| Action | Conversion-max | Economics-max |
|---|---:|---:|
| MEDIUM_DISCOUNT (10%) | **65.7%** | 24.1% |
| FREE_SHIPPING | 21.9% | 33.1% |
| HUMAN_ESCALATION | 9.9% | 29.4% |
| SMALL_DISCOUNT | 0.0% | 4.3% |

**The two objectives disagree on 48.4% of opportunities.**

---

## Financial objective

```
EV(a) = P(recovery | context, a)
        * ( base_contribution_margin
            - incentive_cost_if_recovered(a)
            - expected_return_loss(a)
            - expected_cancellation_loss(a) )
        - fixed_action_cost(a)

ΔEV(a) = EV(a) - EV(DO_NOTHING)
```

Three properties, each enforced by unit tests:

1. **Incentive costs are conditional** — a discount costs nothing if the
   customer does not convert. Fixed costs (messaging, API, agent time) are not.
2. **No double counting** — an incentive is subtracted exactly once, and
   reported GMV is net of discount.
3. **Ranking is incremental** — by ΔEV against `DO_NOTHING`, never raw EV. If no
   action has ΔEV > 0, the system does nothing. Restraint is the default.

---

## Data strategy

Public transaction datasets do not contain counterfactual recovery outcomes —
they record what customers did, never what they would have done under a
different intervention. Three layers with separated roles:

| Layer | Source | Role |
|---|---|---|
| A | UCI Online Retail II | Order-value shape + activity concentration **only** |
| B | RevenueOS simulator | Training, validation, held-out test, oracle |
| C | Razorpay Test Mode | Integration proof only |

Full detail and limitations: [`docs/data-card.md`](docs/data-card.md).

### Propensity logging and exploration

Every logged intervention stores the exact `P(action | context)` under which it
was chosen, and 25% of opportunities are assigned a **randomised** action over
the eligible set. This is what makes off-policy evaluation valid: without it,
"high-LTV customers got free shipping and converted" is confounded, not causal.

Held-out fold: 2,905 opportunities, 708 exploration rows, Kish ESS **1,145**,
max importance weight **51**.

### Not too easy on purpose

The response surface carries shared logit noise plus hidden environmental
mechanisms with no corresponding features — bank outages, competitor flash
sales, courier disruptions, payday effects. If a trained model later reaches
test ROC-AUC above ~0.85, that is a simulator defect, not a success.

---

## Evaluation

Three independent streams, because relying on any one would mislead:

- **A — Observed** held-out outcomes under the logged policy. Factual floor.
- **B — Off-policy** (headline): IPS / SNIPS / **Doubly Robust**, computed from
  logged outcomes and recorded propensities *without querying the simulator's
  counterfactuals*. Reward is net contribution in rupees, not binary recovery.
- **C — Synthetic oracle**: exact counterfactual regret, always labelled as
  such, never presented as production causal lift.

Calibration — Brier, ECE, reliability diagram, and predicted-vs-true ΔP per
action — is the primary ML metric. AUC is reported but is not the success
criterion, because decisions depend on probability magnitude, not ranking.

Methodology: [`docs/evaluation.md`](docs/evaluation.md).

---

## Repository

```
ml/
  actions.py            closed action space + cost semantics
  financial_engine.py   canonical EV / ΔEV        <- 28 unit tests
  config.py             every tunable constant + seed
  simulation/
    environment.py      hidden windows (outages, sales, courier, payday)
    products.py         60 SKUs, weight/zone shipping
    customers.py        observable frame vs latent frame
    sessions.py         sessions -> checkouts -> payments
    behavior.py         P(recovery | context, action) for ALL actions
    logging_policy.py   stochastic policy + stored propensities
    generate.py         `python -m ml.simulation.generate`
  validation/report.py  the Phase 2 review gate
  features/ models/ calibration/ off_policy/ evaluation/   <- Phase 3+
backend/app/            FastAPI (health live; routers Phase 6)
frontend/               Phase 7
docs/                   product-spec, data-card, simulator, evaluation
tests/                  58 tests
```

---

## Reproducibility

One seed controls everything. `manifest.json` persists the seed, simulator
version, logging-policy version, config, record counts, hidden environment
windows and dataset period. Same seed → identical parquet output.

---

## Limitations

- Synthetic behaviour may not reflect production merchants; response surfaces
  are modelled assumptions, not measured effects.
- Public retail data calibrates only order-value shape and activity
  concentration — nothing about intervention response.
- Off-policy estimates depend on overlap; ESS and max weight are reported
  alongside every estimate.
- Held-out support for `DELAYED_RETRY` is thin (~19 rows); its per-action
  estimate is unreliable and flagged as such.
- `HUMAN_ESCALATION` is currently selected more often than a real merchant would
  tolerate — its fixed cost needs raising. Recorded in `docs/simulator.md`
  rather than silently patched.
- No real personal or financial data is used anywhere.

---

## Next: Phase 3

Feature engineering with time-aware historical aggregates, then
`P(recovery | context, action)` via XGBoost, isotonic calibration on the
validation fold only, and the predicted-vs-true ΔP figure.
