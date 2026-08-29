# RevenueOS — Model Report (Phase 3 & 4)

Model `dcbdeaa11e1f49df` · calibration `none` · simulator `1.1.0` · seed `42`

## 1. Executive summary


| item | value |
|---|---|
| Selected model | XGBoost + none calibration |
| TEST ROC-AUC | 0.5965 |
| TEST PR-AUC | 0.4703 |
| TEST Brier | 0.2278 |
| TEST ECE | 0.0125 |
| DR estimated RevenueOS policy value | INR 827.94/opp (95% CI [719.13, 940.65]) |
| Oracle RevenueOS policy value | INR 853.15/opp |
| RevenueOS vs flat 10% contribution | +280.09/opp |
| RevenueOS vs do-nothing contribution | +91.90/opp |
| Conversion-max vs economics divergence | 40.5% |
| DO_NOTHING selection rate | 34.5% |
| Mean policy regret vs oracle | INR 59.71/opp |

## 2. Data integrity


| check | status |
|---|---|
| Simulator version | `1.1.0` (frozen) |
| Split boundaries | TRAIN ≤ 2025-10-04 < VAL ≤ 2025-10-31 < TEST |
| Train/val/test rows | 25,385 / 5,440 / 5,440 |
| Train hash | `9fbda2210f1b432e` |
| Test hash | `603a48f175ace88c` |
| Oracle hash | `c81fa26e0f9c92a1` |
| Oracle access policy | `evaluation_only` |
| Model frozen at | 2026-08-27T13:46:59.814386+00:00 |
| Leakage audit | PASS (no forbidden columns in feature matrix) |

The model was frozen before any TEST or oracle read. Calibration was fitted on
VALIDATION only.

## 3. Baseline comparison (VALIDATION)

| model               |   roc_auc |   pr_auc |   brier |   log_loss |    ece |
|:--------------------|----------:|---------:|--------:|-----------:|-------:|
| global_mean         |    0.5000 |   0.3783 |  0.2352 |     0.6633 | 0.0073 |
| segment_lookup      |    0.5734 |   0.4357 |  0.2319 |     0.6562 | 0.0258 |
| logistic_regression |    0.6099 |   0.4908 |  0.2259 |     0.6433 | 0.0151 |
| xgboost_raw         |    0.6292 |   0.5117 |  0.2231 |     0.6373 | 0.0254 |
| xgboost_platt       |    0.6292 |   0.5117 |  0.2248 |     0.6410 | 0.0373 |
| xgboost_isotonic    |    0.6271 |   0.4930 |  0.2237 |     0.6382 | 0.0283 |


Held-out TEST, selected model: Brier **0.2278** vs global-mean floor
**0.2352** — an improvement of
3.2%.
Uncalibrated TEST Brier was 0.2278.

ROC-AUC of 0.596 is modest by classification standards and that is
expected: the simulator injects shared logit noise plus hidden environment
windows that cap achievable accuracy. What matters for this system is whether
probabilities are *calibrated* and whether *relative* action response is
recovered, both reported below.

## 4. Calibration detail

### By action
| action                |    n |   actual_rate |   predicted_mean |     gap |   brier |
|:----------------------|-----:|--------------:|-----------------:|--------:|--------:|
| DELAYED_RETRY         |  145 |        0.4966 |           0.4730 | -0.0236 |  0.2374 |
| DO_NOTHING            | 1825 |        0.3627 |           0.3471 | -0.0157 |  0.2252 |
| FREE_SHIPPING         |  674 |        0.4837 |           0.4730 | -0.0107 |  0.2387 |
| HUMAN_ESCALATION      |  269 |        0.3346 |           0.3321 | -0.0024 |  0.2237 |
| IMMEDIATE_RETRY       |  230 |        0.3826 |           0.4193 |  0.0366 |  0.2293 |
| MEDIUM_DISCOUNT       |  519 |        0.3410 |           0.3648 |  0.0237 |  0.2236 |
| PAYMENT_LINK          |  248 |        0.3347 |           0.3464 |  0.0117 |  0.2174 |
| PAYMENT_METHOD_SWITCH |  352 |        0.3352 |           0.3384 |  0.0032 |  0.2212 |
| SMALL_DISCOUNT        | 1178 |        0.3633 |           0.3522 | -0.0112 |  0.2311 |


> Aggregate calibration can hide action-specific failure, which is why this table exists. A large positive gap means the model over-promises recovery for that action and would over-spend on it.

## 5. Oracle response evaluation (post-freeze)

**Synthetic Oracle Evaluation** — an upper-bound reference, not causal lift.


| action                |    n |   pred_mean_dP |   true_mean_dP |    bias |    mae |   rmse |   pearson |   spearman |
|:----------------------|-----:|---------------:|---------------:|--------:|-------:|-------:|----------:|-----------:|
| FREE_SHIPPING         | 2916 |         0.1413 |         0.1598 | -0.0185 | 0.0734 | 0.0900 |    0.3866 |     0.3463 |
| SMALL_DISCOUNT        | 5440 |         0.0226 |         0.0084 |  0.0142 | 0.0533 | 0.0654 |    0.5655 |     0.5448 |
| MEDIUM_DISCOUNT       | 5440 |         0.0337 |         0.0371 | -0.0034 | 0.0643 | 0.0780 |    0.5302 |     0.5064 |
| PAYMENT_METHOD_SWITCH | 1536 |        -0.0011 |         0.0447 | -0.0458 | 0.0843 | 0.1108 |    0.1306 |     0.1360 |
| IMMEDIATE_RETRY       | 1536 |         0.0250 |         0.0147 |  0.0103 | 0.0389 | 0.0486 |    0.2110 |     0.1535 |
| DELAYED_RETRY         | 1536 |         0.0805 |         0.1197 | -0.0392 | 0.0773 | 0.0970 |    0.2029 |     0.2031 |
| PAYMENT_LINK          | 5440 |        -0.0006 |        -0.0213 |  0.0207 | 0.0391 | 0.0513 |    0.2155 |     0.1952 |
| HUMAN_ESCALATION      | 5440 |        -0.0006 |         0.0056 | -0.0062 | 0.0351 | 0.0411 |    0.2567 |     0.2232 |


`bias` is mean(predicted dP - true dP): positive means the model over-estimates
that action's uplift and will over-select it. This table is the headline ML
artifact, because it measures learned *treatment response* rather than customer
ranking.

## 6. Policy comparison (oracle evaluation)

| policy                 |   conversion |   net_gmv_per_opp |   incentive_cost_per_opp |   fixed_cost_per_opp |   net_contribution_per_opp |
|:-----------------------|-------------:|------------------:|-------------------------:|---------------------:|---------------------------:|
| LOGGED_HISTORICAL      |       0.3695 |         2450.6019 |                  70.4328 |               7.6351 |                   712.1322 |
| DO_NOTHING             |       0.3450 |         2423.8744 |                   0.0000 |               0.0000 |                   761.2542 |
| FLAT_10_PERCENT        |       0.3821 |         2421.3947 |                 269.0439 |               2.0000 |                   573.0587 |
| RULES                  |       0.4153 |         2744.5555 |                  56.4260 |               1.2950 |                   819.4090 |
| MODEL_CONVERSION_MAX   |       0.4635 |         2895.8916 |                 152.1749 |               1.7200 |                   796.2557 |
| REVENUEOS              |       0.4354 |         2767.0180 |                  22.1444 |               1.0553 |                   853.1508 |
| REVENUEOS_CONSERVATIVE |       0.4173 |         2734.8859 |                  15.1671 |               0.8018 |                   849.2704 |
| ORACLE_ECONOMIC        |       0.4634 |         3015.0236 |                  44.8166 |               3.7893 |                   912.8593 |


### The central result

| | conversion | incentive cost | net contribution |
|---|---:|---:|---:|
| DO_NOTHING | 0.3450 | 0.00 | 761.25 |
| FLAT 10% | 0.3821 | 269.04 | 573.06 |
| REVENUEOS | 0.4354 | 22.14 | 853.15 |

Flat 10% converts better than doing
nothing and earns less.
RevenueOS spends INR 246.90/opp
less on incentives than flat discounting.

## 7. Conversion vs contribution


- Conversion-max and economics-max select **different actions in 40.5%** of TEST opportunities.
- DO_NOTHING selection rate: **34.5%**
- DO_NOTHING precision (selected & oracle-optimal): **52.7%**
- DO_NOTHING recall (of oracle DO_NOTHING cases): **71.3%**

Action distribution vs the logged historical policy:


| action                |   revenueos |   logged |
|:----------------------|------------:|---------:|
| DELAYED_RETRY         |      0.2349 |   0.0267 |
| DO_NOTHING            |      0.3454 |   0.3355 |
| FREE_SHIPPING         |      0.3631 |   0.1239 |
| HUMAN_ESCALATION      |      0.0000 |   0.0494 |
| IMMEDIATE_RETRY       |      0.0189 |   0.0423 |
| MEDIUM_DISCOUNT       |      0.0000 |   0.0954 |
| PAYMENT_LINK          |      0.0000 |   0.0456 |
| PAYMENT_METHOD_SWITCH |      0.0000 |   0.0647 |
| SMALL_DISCOUNT        |      0.0377 |   0.2165 |

## 8. Off-policy evaluation

Reward is realised **net contribution in rupees**, not binary recovery.


| policy               | clip_level   |      ips |    snips |       dr |       ess |   match_rate |   max_weight |   n_matched |
|:---------------------|:-------------|---------:|---------:|---------:|----------:|-------------:|-------------:|------------:|
| DO_NOTHING           | none         | 746.6108 | 741.0362 | 733.5915 | 1385.5565 |       0.3355 |      11.8000 |        1825 |
| DO_NOTHING           | clip20       | 746.6108 | 741.0362 | 733.5915 | 1385.5565 |       0.3355 |      11.8000 |        1825 |
| DO_NOTHING           | clip10       | 737.8706 | 738.9132 | 734.9663 | 1431.0215 |       0.3355 |      10.0000 |        1825 |
| FLAT_10_PERCENT      | none         | 487.9493 | 519.7573 | 508.7048 |  413.8902 |       0.0954 |      33.7405 |         519 |
| FLAT_10_PERCENT      | clip20       | 482.7307 | 533.4314 | 510.7437 |  451.1588 |       0.0954 |      20.0000 |         519 |
| FLAT_10_PERCENT      | clip10       | 440.0491 | 573.8202 | 525.2675 |  501.3801 |       0.0954 |      10.0000 |         519 |
| RULES                | none         | 774.2195 | 784.1960 | 798.0027 |  852.2418 |       0.2410 |      31.6596 |        1311 |
| RULES                | clip20       | 760.5221 | 782.0388 | 790.8048 |  938.6667 |       0.2410 |      20.0000 |        1311 |
| RULES                | clip10       | 736.3055 | 778.3565 | 778.8213 | 1020.8483 |       0.2410 |      10.0000 |        1311 |
| MODEL_CONVERSION_MAX | none         | 788.6696 | 814.7711 | 771.5832 |  734.2792 |       0.1763 |      27.1370 |         959 |
| MODEL_CONVERSION_MAX | clip20       | 781.8294 | 814.3313 | 771.5415 |  761.6680 |       0.1763 |      20.0000 |         959 |
| MODEL_CONVERSION_MAX | clip10       | 721.7152 | 784.3686 | 770.3194 |  846.0035 |       0.1763 |      10.0000 |         959 |
| REVENUEOS            | none         | 871.5217 | 880.2636 | 827.9388 |  764.6830 |       0.2191 |      27.1370 |        1192 |
| REVENUEOS            | clip20       | 864.0771 | 886.3125 | 829.4134 |  822.6198 |       0.2191 |      20.0000 |        1192 |
| REVENUEOS            | clip10       | 805.9168 | 874.1007 | 834.2628 |  976.5091 |       0.2191 |      10.0000 |        1192 |


### Bootstrap CIs (DR, unclipped, 1000 resamples)

| policy               |   dr_mean |   ci_low |   ci_high |
|:---------------------|----------:|---------:|----------:|
| DO_NOTHING           |  732.3466 | 638.1220 |  825.9264 |
| FLAT_10_PERCENT      |  508.6547 | 401.7862 |  604.5805 |
| RULES                |  797.0580 | 704.6886 |  889.2217 |
| MODEL_CONVERSION_MAX |  772.5266 | 663.8401 |  892.4610 |
| REVENUEOS            |  827.1849 | 719.1329 |  940.6509 |


`ess` is Kish effective sample size on matched rows; `match_rate` is the share
of TEST rows where the logged action equals the target policy's action. A low
match rate means the estimate rests on few rows regardless of nominal n.

## 9. Does OPE recover simulator truth?

| policy               |   oracle_value |   dr_estimate |   abs_diff |   rel_diff |
|:---------------------|---------------:|--------------:|-----------:|-----------:|
| DO_NOTHING           |       761.2542 |      733.5915 |    27.6627 |     0.0363 |
| FLAT_10_PERCENT      |       573.0587 |      508.7048 |    64.3539 |     0.1123 |
| RULES                |       819.4090 |      798.0027 |    21.4063 |     0.0261 |
| MODEL_CONVERSION_MAX |       796.2557 |      771.5832 |    24.6725 |     0.0310 |
| REVENUEOS            |       853.1508 |      827.9388 |    25.2120 |     0.0296 |


This is the most methodologically important table in the report. Stream B (DR,
logged data only) and Stream C (oracle counterfactuals) are computed by
completely different routes. Where they agree, the off-policy machinery is
working; where they disagree, the discrepancy is reported rather than hidden.

## 10. Ablations

| ablation            |   n_features |   brier |   log_loss |   pr_auc |   oracle_dP_mae |   brier_delta_vs_full |   dP_mae_delta_vs_full |
|:--------------------|-------------:|--------:|-----------:|---------:|----------------:|----------------------:|-----------------------:|
| full                |           58 |  0.2277 |     0.6476 |   0.4711 |          0.0668 |                0.0000 |                 0.0000 |
| no_action_history   |           46 |  0.2283 |     0.6488 |   0.4676 |          0.0714 |                0.0006 |                 0.0046 |
| no_customer_history |           46 |  0.2281 |     0.6485 |   0.4686 |          0.0657 |                0.0004 |                -0.0011 |
| no_economic_context |           53 |  0.2276 |     0.6473 |   0.4727 |          0.0652 |               -0.0001 |                -0.0016 |
| no_time_context     |           56 |  0.2275 |     0.6472 |   0.4716 |          0.0659 |               -0.0002 |                -0.0009 |

## 11. Robustness

### Policy stability under probability perturbation

|   perturbation |   action_change_rate |
|---------------:|---------------------:|
|         0.0200 |               0.2638 |
|         0.0500 |               0.4384 |


### Calibration impact on economics
Using raw (uncalibrated) probabilities instead of calibrated ones changes the
selected action on **0.0%** of opportunities. This is the practical
argument for calibration: miscalibrated probabilities feed directly into dEV and
shift real spending decisions.

### Regret distribution
mean INR 59.71 · median INR 0.00 · p90 INR 150.45

## 12. Error analysis


The 20 highest-regret TEST cases are in `high_regret_cases.csv`. Regret is
concentrated: the top decile accounts for
77%
of total regret, so failures are localised rather than systemic.

## 13. Warnings

No automated warnings raised.

## 14. Model gate assessment

- PASS — Integrity — no leakage, oracle isolated, TEST untouched until freeze
- PASS — Calibration — beats global-mean Brier floor
- PASS — Response learning — positive predicted/true dP correlation
- PASS — Economics — beats flat discount on oracle contribution
- PASS — OPE — DR ranks RevenueOS above flat discount
- PASS — Robustness — policy not catastrophically unstable

**Gate: PASS**
