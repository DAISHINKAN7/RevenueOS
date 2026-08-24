# RevenueOS — Evaluation Methodology

Three independent evidence streams. Each answers a different question, and each
has different weaknesses. Reporting only one would be misleading.

---

## Why three streams

The simulator generates the data, the model trains on that data, and the oracle
comes from the same simulator. A result of the form "RevenueOS beats flat 10% by
+18%" measured only against the oracle is partly a measurement of how well the
model recovered structure we injected ourselves.

Stream B exists specifically to produce a number that **does not query the
simulator's counterfactuals** — it uses only logged outcomes and recorded
propensities. Where A, B and C disagree, the disagreement is itself reported.

---

## Stream A — Observed held-out outcomes

On the newest 15% of opportunities, report what actually happened under the
logged historical policy: action taken, observed recovery, observed
contribution. No counterfactual claims. This is the factual floor.

## Stream B — Off-policy evaluation (headline)

Because every logged row stores `P(action | context)`, the value of the
RevenueOS policy can be estimated from logged data alone.

Estimators: **IPS**, **SNIPS**, and **Doubly Robust** (headline).

**Reward definition — important.** The DR reward is *net contribution in rupees*:

```
r = recovered_contribution - incentive_cost_realised - fixed_action_cost
```

Not binary recovery. A binary reward would silently reintroduce the
conversion-maximising objective the entire project argues against, and would not
be comparable to the business baseline table.

**Weight clipping must never be silent.** Every OPE table reports a sensitivity
row set: `unclipped`, `clip@20`, `clip@10`. If the estimate moves materially
across those, the overlap is inadequate and the result is reported as such
rather than quoting the most flattering variant.

**Mandatory diagnostics** reported alongside every estimate:

- Kish effective sample size (currently **2,127** on the held-out fold)
- minimum logged propensity (currently **0.0196**)
- maximum importance weight (currently **51**)
- clipped-weight fraction
- propensity overlap histogram
- per-action held-out support counts

An OPE point estimate without these diagnostics is not interpretable. Held-out
support for `DELAYED_RETRY` is thin (~19 exploration rows) and its per-action
estimate is reported with that caveat rather than suppressed.

## Stream C — Synthetic oracle

The simulator holds `P(recovery | context, action)` for every action, enabling
exact counterfactual computation of:

- policy regret vs the oracle-optimal policy
- true incremental contribution
- action-selection accuracy
- true ΔP error per action

Labelled **Synthetic Oracle Evaluation** everywhere. Never presented as
production causal lift.

---

## Baselines

| Strategy | Definition |
|---|---|
| A | `DO_NOTHING` everywhere |
| B | Flat 10% discount everywhere |
| C | Simple rules: abandonment → 5% discount; bank timeout → immediate retry |
| RevenueOS | ML response prediction + ΔEV optimisation + policy gate |

Reported for each: net recovered GMV (net of discount), incentive cost, net
contribution, recovery rate, DR policy value, policy violations.

**Baseline B is expected to win on recovery rate and lose on net contribution.**
That contrast is the project's central demonstration, not an inconvenience.

---

## ML metrics

Calibration is the primary requirement, because decisions depend on probability
*magnitude*, not just ranking:

- Brier score, log loss, expected calibration error
- reliability diagram
- **predicted vs true ΔP(recovery) per action** — the headline ML figure

ROC-AUC and PR-AUC are reported but are explicitly not the success criterion.

Calibration (isotonic or Platt) is fitted on the **validation fold only**.

### Difficulty guard

Test ROC-AUC above ~0.85 or near-perfect calibration indicates the simulator is
too easy — raise `response_logit_noise_sd` or add a hidden mechanism. This is a
defect condition, not a success condition.

---

## Confidence intervals

1,000 bootstrap resamples on all headline business metrics. Reported as:

```
RevenueOS incremental contribution uplift: +X%
95% bootstrap CI: [lo, hi]
```

---

## Scientific integrity rule

If RevenueOS fails to beat a baseline under any stream, **report it and explain
the mechanism**. Do not adjust the seed, split, threshold or simulator
assumptions until the result improves. A submission that says "we lose on
recovery rate but win on net contribution, and here is why" is more convincing
than one that wins everywhere.
