# RevenueOS — Final Model Audit

Pipeline `1.1.0` · model `dcbdeaa11e1f49df` · calibration `none` · simulator `1.1.0` (frozen)

## Headline metrics

| Metric                                 | Final Value   |
|:---------------------------------------|:--------------|
| TEST ROC-AUC                           | 0.5965        |
| TEST PR-AUC                            | 0.4703        |
| TEST Brier                             | 0.2278        |
| TEST ECE                               | 0.0125        |
| RevenueOS Oracle Value / Opportunity   | INR 853.15    |
| RevenueOS DR Value / Opportunity       | INR 827.94    |
| DR vs Oracle Relative Error            | 2.96%         |
| RevenueOS vs DO_NOTHING                | INR +91.90    |
| RevenueOS vs Flat 10%                  | INR +280.09   |
| Oracle Incremental Value Captured      | 60.6%         |
| Conversion/Economics Action Divergence | 40.5%         |
| DO_NOTHING Selection Rate              | 34.5%         |
| Mean Regret                            | INR 59.71     |
| P95 Regret                             | INR 296.53    |

## 1. Corrected calibration protocol

> The initial model gate exposed an optimistic calibration-selection procedure because isotonic calibration was evaluated on its fitting partition. The pipeline was corrected by separating calibration fitting from model selection before finalizing the model.


**What went wrong.** The first gate fitted Platt and isotonic on VALIDATION and
then *selected between them on the same VALIDATION rows*. Isotonic scored
ECE = **0.00000** —
which is not a measurement of calibration quality at all. Isotonic regression is
a step function fitted to those exact points; reproducing them is guaranteed.
The procedure structurally favoured the most flexible calibrator.

**The correction.** VALIDATION is now split chronologically:

| partition | period | rows | role |
|---|---|---:|---|
| TRAIN | 2025-06-01 → 2025-10-04 | 25,385 | fits base models |
| CALIBRATION | 2025-10-04 → 2025-10-17 | 2,720 | fits Platt / isotonic |
| MODEL_SELECTION | 2025-10-17 → 2025-10-31 | 2,720 | chooses among candidates |
| TEST | after | 5,440 | reporting only |

No candidate is ever scored on rows that fitted it.

### MODEL_SELECTION metrics (honest)
| model               |   roc_auc |   pr_auc |   brier |   log_loss |    ece |
|:--------------------|----------:|---------:|--------:|-----------:|-------:|
| global_mean         |    0.5000 |   0.3783 |  0.2352 |     0.6633 | 0.0073 |
| segment_lookup      |    0.5734 |   0.4357 |  0.2319 |     0.6562 | 0.0258 |
| logistic_regression |    0.6099 |   0.4908 |  0.2259 |     0.6433 | 0.0151 |
| xgboost_raw         |    0.6292 |   0.5117 |  0.2231 |     0.6373 | 0.0254 |
| xgboost_platt       |    0.6292 |   0.5117 |  0.2248 |     0.6410 | 0.0373 |
| xgboost_isotonic    |    0.6271 |   0.4930 |  0.2237 |     0.6382 | 0.0283 |


Measured honestly, isotonic ECE is **0.0283**,
not 0.0000 — and raw XGBoost now has both the best Brier
(0.22306) and the lowest ECE
(0.0254) of the three.

**Selected: `none` calibration.** This choice came
exclusively from MODEL_SELECTION. TEST was not consulted, even though the first
gate's TEST numbers happened to point the same way — using them would have been
selection-on-test regardless of the outcome.

## 2. Final TEST metrics (reporting only)

| model   |    ece |   log_loss |   brier |   pr_auc |   roc_auc |
|:--------|-------:|-----------:|--------:|---------:|----------:|
| final   | 0.0125 |     0.6478 |  0.2278 |   0.4703 |    0.5965 |

## 3. Economic value by model family (ML contribution decomposition)

| policy           |   conversion |   net_contribution_per_opp |   incremental_vs_do_nothing |   value_capture_pct |   mean_regret |   do_nothing_rate |
|:-----------------|-------------:|---------------------------:|----------------------------:|--------------------:|--------------:|------------------:|
| A_segment_lookup |       0.4352 |                   866.8531 |                    105.5989 |             69.6539 |       46.0062 |            0.3866 |
| B_logistic       |       0.4353 |                   860.7813 |                     99.5271 |             65.6489 |       52.0780 |            0.3607 |
| C_xgboost_raw    |       0.4354 |                   853.1508 |                     91.8966 |             60.6158 |       59.7085 |            0.3454 |
| D_final_selected |       0.4354 |                   853.1508 |                     91.8966 |             60.6158 |       59.7085 |            0.3454 |
| RULES            |       0.4153 |                   819.4090 |                     58.1548 |             38.3594 |       93.4503 |            0.2936 |
| FLAT_10_PERCENT  |       0.3821 |                   573.0587 |                   -188.1955 |           -124.1353 |      339.8006 |            0.0000 |
| DO_NOTHING       |       0.3450 |                   761.2542 |                      0.0000 |              0.0000 |      151.6051 |            1.0000 |
| ORACLE_ECONOMIC  |       0.4634 |                   912.8593 |                    151.6051 |            100.0000 |        0.0000 |            0.2551 |


**This is the most important table in the audit.** All four rows A-D use the
identical financial engine and policy rule; only the response model changes, so
the differences isolate the marginal economic value of the modelling itself.

- The financial engine driven by a crude segment lookup table reaches
  INR 866.85/opp, capturing
  69.7% of available headroom.
- Moving to the final XGBoost **subtracts** **INR -13.70/opp**
  (60.6% capture).
- Flat 10% discounting *destroys*
  INR 188.20/opp
  relative to doing nothing, and the rule baseline captures only
  38.4%.

**The honest reading, stated plainly: gradient boosting does not beat a smoothed
contingency table on economic value here — it is
INR 13.70/opp worse.** Essentially all of the
economic win comes from the financial optimisation layer correctly refusing to
discount, not from model sophistication.

Why this happens is visible in the response metrics: at ROC-AUC ~0.60 the model
carries little individual-level signal, and what the policy actually needs is a
coarse, well-behaved estimate of *which action class* helps *which kind of
opportunity*. A smoothed lookup over (opportunity_type x failure_reason x action)
supplies exactly that, with less variance. XGBoost's extra flexibility mostly
adds noise to dEV comparisons that are frequently close calls.

This does not invalidate the system — the policy still beats every naive
baseline by a wide margin, and off-policy evidence confirms it independently. It
does mean the defensible claim is about *incremental economics plus bounded
authority*, not about model sophistication. Reporting it this way is more
credible than an ML-centric story the numbers do not support.

## 4. Oracle incremental value capture


```
capture = (policy_value - DO_NOTHING_value) / (ORACLE_value - DO_NOTHING_value)
```
Headroom available to the oracle: INR 151.61/opp
(DO_NOTHING 761.25 → ORACLE 912.86).

**RevenueOS captures 60.6%**
of the incremental economic value available to a perfectly-informed oracle.

## 5. Regret distribution
| metric                  |    mean |    p50 |     p75 |      p90 |      p95 |      p99 |       max |
|:------------------------|--------:|-------:|--------:|---------:|---------:|---------:|----------:|
| absolute_regret_inr     | 59.7085 | 0.0000 | 34.3174 | 150.4503 | 296.5294 | 934.3614 | 4317.1497 |
| regret_over_cart_value  |  0.0075 | 0.0000 |  0.0071 |   0.0253 |   0.0433 |   0.0769 |    0.1439 |
| regret_over_base_margin |  0.0227 | 0.0000 |  0.0231 |   0.0766 |   0.1323 |   0.2293 |    0.3641 |


The worst 1% of opportunities account for **27.6%** of total regret.

## 6. Tail-risk analysis (worst 1%)

55 opportunities with regret >= INR 934.

| attribute                            |       tail |   overall |
|:-------------------------------------|-----------:|----------:|
| share PAYMENT_FAILURE                |     0.8545 |    0.2824 |
| mean cart_value                      | 33999.9636 | 7677.5632 |
| mean base_margin                     | 10605.5945 | 2395.4322 |
| share cold-start (no coupon history) |     0.3455 |    0.3015 |
| share DO_NOTHING selected            |     0.2182 |    0.3454 |


### Tail composition by selected vs oracle action

| selected        | oracle_action         |   n |
|:----------------|:----------------------|----:|
| IMMEDIATE_RETRY | DELAYED_RETRY         |  23 |
| DELAYED_RETRY   | PAYMENT_METHOD_SWITCH |   9 |
| IMMEDIATE_RETRY | PAYMENT_METHOD_SWITCH |   8 |
| DO_NOTHING      | PAYMENT_METHOD_SWITCH |   6 |
| DO_NOTHING      | SMALL_DISCOUNT        |   4 |
| DO_NOTHING      | MEDIUM_DISCOUNT       |   2 |
| SMALL_DISCOUNT  | DO_NOTHING            |   2 |
| DELAYED_RETRY   | SMALL_DISCOUNT        |   1 |


### Tail by failure reason

| failure_reason         |   n |
|:-----------------------|----:|
| CARD_DECLINED          |  15 |
| BANK_TIMEOUT           |  11 |
| NO_PAYMENT_FAILURE     |   8 |
| AUTHENTICATION_FAILURE |   8 |
| UPI_TIMEOUT            |   6 |
| NETWORK_ERROR          |   4 |
| USER_CANCELLED         |   2 |
| UNKNOWN                |   1 |


Tail cases carry **4.4x the mean cart value** of the overall test set.
Regret scales with the money at stake, so the natural mitigation is a
cart-value-linked human-approval threshold rather than a better model — which is
exactly what the Phase 5 policy engine should encode.

## 7. High-value cohort

Threshold: VALIDATION p90 cart value = INR 17,817 (derived without touching TEST). Cohort n = 537.


| cohort     |    n |   mean_regret |   p95_regret |   do_nothing_rate |   revenueos_value |   oracle_value |
|:-----------|-----:|--------------:|-------------:|------------------:|------------------:|---------------:|
| high_value |  537 |      277.5937 |    1313.8991 |            0.6778 |         2983.6809 |      3261.2746 |
| rest       | 4903 |       35.8446 |     191.5139 |            0.3090 |          619.8050 |       655.6496 |


### High-value action distribution

| selected        |   share |
|:----------------|--------:|
| DO_NOTHING      |  0.6778 |
| DELAYED_RETRY   |  0.1508 |
| IMMEDIATE_RETRY |  0.0931 |
| SMALL_DISCOUNT  |  0.0782 |

## 8. Does calibration change money decisions?


| comparison | value |
|---|---:|
| Action divergence (raw vs final) | 0.00% |
| Net contribution difference | INR +0.00/opp |
| DO_NOTHING rate difference | +0.00% |

Since the corrected protocol selected `none`, the final
model *is* raw XGBoost and this comparison is definitionally zero. That is itself
the finding: the honest calibration procedure concluded no post-hoc calibrator
improved on the base model's probabilities, so no calibration step is applied.

## 9. Final off-policy cross-check
| clip_level   |      ips |    snips |       dr |      ess |   match_rate |   max_weight |   n_matched |
|:-------------|---------:|---------:|---------:|---------:|-------------:|-------------:|------------:|
| none         | 871.5217 | 880.2636 | 827.9388 | 764.6830 |       0.2191 |      27.1370 |        1192 |
| clip20       | 864.0771 | 886.3125 | 829.4134 | 822.6198 |       0.2191 |      20.0000 |        1192 |
| clip10       | 805.9168 | 874.1007 | 834.2628 | 976.5091 |       0.2191 |      10.0000 |        1192 |


DR bootstrap (1000 resamples): **INR 827.18/opp**, 95% CI [719.13, 940.65].

## 10. DR vs oracle agreement


| quantity | value |
|---|---:|
| DR estimate (logged data only) | INR 827.94/opp |
| Oracle value (counterfactual) | INR 853.15/opp |
| Absolute error | INR 25.21 |
| Relative error | 2.96% |

Two independent routes — one using only logged actions and propensities, the
other using the simulator's counterfactuals — agree to within
3.0%. This is the core
credibility metric: it says the off-policy machinery recovers the truth on data
where the truth happens to be knowable.

## 11. Limitations

1. **ML contribution is not positive.** A segment lookup table plus the financial
   engine captures
   69.7% of headroom vs
   60.6% for the final model —
   gradient boosting is INR 13.70/opp *worse* on economics. The
   value comes from the financial layer, not the model.
2. **No calibrator helped.** Under the corrected protocol neither Platt nor
   isotonic beat raw XGBoost, so the "calibration matters" story is weaker than
   the first gate implied — though the *measurement* of that is now sound.
3. **PAYMENT_METHOD_SWITCH response is not learned** (near-zero oracle dP
   correlation), so the policy never selects it despite genuine effectiveness on
   CARD_DECLINED.
4. **Regret is concentrated in high-value carts** (4.4x mean cart value in
   the tail), which is precisely where errors are most expensive.
5. Synthetic environment throughout: this is policy evaluation under a documented
   behavioural model, not real-world causal uplift.
6. Off-policy estimates depend on overlap; match rate is
   21.9% with ESS 765.

## 12. Final recommendation

**SELECTED_PRODUCTION_MODEL: XGBoost (no post-hoc calibration)** — retained for
the response-estimation interface, but with an explicit caveat: on this dataset
the segment lookup baseline achieves higher economic value
(INR 866.85 vs 853.15/opp). XGBoost is kept because it produces
per-action probabilities across the full feature space that the policy and future
UI depend on, and because the lookup table cannot extrapolate to unseen
context combinations. If the gap persists after Phase 5, the lookup table should
replace it. **This should be stated in the README rather than omitted.**

Rejected alternatives, with reasons:

- *Isotonic* — appeared best only because it was scored on its own fitting
  partition. Under the corrected protocol its ECE is
  0.0283 vs raw
  0.0254.
- *Platt* — worse Brier and ECE than raw on MODEL_SELECTION.
- *Logistic regression* — lower value capture, no interpretability advantage that
  matters here since the financial engine already supplies the explanation layer.
- *Segment lookup* — **economically superior here** and must be reported as such.
  Rejected as the production interface only because it cannot score arbitrary
  context/action combinations, not because it performed worse.

**Gate: PASS**, with the ML-contribution caveat stated explicitly rather than
buried. The defensible claim is: *a modest response model combined with a correct
incremental-economics layer and a deterministic policy gate beats both naive
discounting and conversion-maximisation, and off-policy evidence independently
confirms it to within 3%.*
