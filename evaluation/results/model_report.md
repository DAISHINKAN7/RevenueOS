# RevenueOS — Model Report (Phase 3 & 4)

Model `dcbdeaa11e1f49df` · calibration `isotonic` · simulator `1.1.0` · seed `42`

## 1. Executive summary


| item | value |
|---|---|
| Selected model | XGBoost + isotonic calibration |
| TEST ROC-AUC | 0.5949 |
| TEST PR-AUC | 0.4600 |
| TEST Brier | 0.2285 |
| TEST ECE | 0.0221 |
| DR estimated RevenueOS policy value | INR 807.19/opp (95% CI [708.45, 912.58]) |
| Oracle RevenueOS policy value | INR 836.55/opp |
| RevenueOS vs flat 10% contribution | +263.49/opp |
| RevenueOS vs do-nothing contribution | +75.30/opp |
| Conversion-max vs economics divergence | 31.6% |
| DO_NOTHING selection rate | 28.2% |
| Mean policy regret vs oracle | INR 76.31/opp |

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
| Model frozen at | 2026-08-24T13:48:06.054546+00:00 |
| Leakage audit | PASS (no forbidden columns in feature matrix) |

The model was frozen before any TEST or oracle read. Calibration was fitted on
VALIDATION only.

## 3. Baseline comparison (VALIDATION)

| model               |   roc_auc |   pr_auc |   brier |   log_loss |    ece |
|:--------------------|----------:|---------:|--------:|-----------:|-------:|
| global_mean         |    0.5000 |   0.3645 |  0.2317 |     0.6561 | 0.0065 |
| segment_lookup      |    0.5733 |   0.4173 |  0.2289 |     0.6500 | 0.0260 |
| logistic_regression |    0.6012 |   0.4639 |  0.2242 |     0.6398 | 0.0102 |
| xgboost_raw         |    0.6160 |   0.4799 |  0.2220 |     0.6351 | 0.0134 |
| xgboost_platt       |    0.6160 |   0.4799 |  0.2220 |     0.6351 | 0.0118 |
| xgboost_isotonic    |    0.6209 |   0.4727 |  0.2204 |     0.6313 | 0.0000 |


Held-out TEST, selected model: Brier **0.2285** vs global-mean floor
**0.2317** — an improvement of
1.4%.
Uncalibrated TEST Brier was 0.2278.

ROC-AUC of 0.595 is modest by classification standards and that is
expected: the simulator injects shared logit noise plus hidden environment
windows that cap achievable accuracy. What matters for this system is whether
probabilities are *calibrated* and whether *relative* action response is
recovered, both reported below.

## 4. Calibration detail

### By action
| action                |    n |   actual_rate |   predicted_mean |     gap |   brier |
|:----------------------|-----:|--------------:|-----------------:|--------:|--------:|
| DELAYED_RETRY         |  145 |        0.4966 |           0.4774 | -0.0192 |  0.2336 |
| DO_NOTHING            | 1825 |        0.3627 |           0.3388 | -0.0239 |  0.2259 |
| FREE_SHIPPING         |  674 |        0.4837 |           0.4828 | -0.0009 |  0.2369 |
| HUMAN_ESCALATION      |  269 |        0.3346 |           0.3180 | -0.0165 |  0.2293 |
| IMMEDIATE_RETRY       |  230 |        0.3826 |           0.4162 |  0.0336 |  0.2296 |
| MEDIUM_DISCOUNT       |  519 |        0.3410 |           0.3581 |  0.0170 |  0.2249 |
| PAYMENT_LINK          |  248 |        0.3347 |           0.3393 |  0.0046 |  0.2171 |
| PAYMENT_METHOD_SWITCH |  352 |        0.3352 |           0.3279 | -0.0073 |  0.2232 |
| SMALL_DISCOUNT        | 1178 |        0.3633 |           0.3432 | -0.0201 |  0.2322 |


> Aggregate calibration can hide action-specific failure, which is why this table exists. A large positive gap means the model over-promises recovery for that action and would over-spend on it.

## 5. Oracle response evaluation (post-freeze)

**Synthetic Oracle Evaluation** — an upper-bound reference, not causal lift.


| action                |    n |   pred_mean_dP |   true_mean_dP |    bias |    mae |   rmse |   pearson |   spearman |
|:----------------------|-----:|---------------:|---------------:|--------:|-------:|-------:|----------:|-----------:|
| FREE_SHIPPING         | 2916 |         0.1613 |         0.1598 |  0.0015 | 0.0755 | 0.0937 |    0.3543 |     0.3228 |
| SMALL_DISCOUNT        | 5440 |         0.0241 |         0.0084 |  0.0158 | 0.0559 | 0.0695 |    0.4718 |     0.4820 |
| MEDIUM_DISCOUNT       | 5440 |         0.0357 |         0.0371 | -0.0014 | 0.0663 | 0.0815 |    0.4482 |     0.4626 |
| PAYMENT_METHOD_SWITCH | 1536 |        -0.0013 |         0.0447 | -0.0460 | 0.0846 | 0.1111 |    0.0320 |    -0.0014 |
| IMMEDIATE_RETRY       | 1536 |         0.0265 |         0.0147 |  0.0118 | 0.0460 | 0.0574 |    0.1113 |     0.1180 |
| DELAYED_RETRY         | 1536 |         0.0881 |         0.1197 | -0.0316 | 0.0812 | 0.1022 |    0.1644 |     0.1741 |
| PAYMENT_LINK          | 5440 |        -0.0006 |        -0.0213 |  0.0207 | 0.0393 | 0.0517 |    0.0510 |     0.0271 |
| HUMAN_ESCALATION      | 5440 |        -0.0006 |         0.0056 | -0.0062 | 0.0353 | 0.0415 |    0.0643 |     0.0337 |


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
| MODEL_CONVERSION_MAX   |       0.4548 |         2825.6154 |                 106.9180 |               1.6708 |                   806.2747 |
| REVENUEOS              |       0.4377 |         2764.5294 |                  44.5558 |               1.2119 |                   836.5513 |
| REVENUEOS_CONSERVATIVE |       0.4201 |         2723.9394 |                  36.1195 |               0.9434 |                   830.6837 |
| ORACLE_ECONOMIC        |       0.4634 |         3015.0236 |                  44.8166 |               3.7893 |                   912.8593 |


### The central result

| | conversion | incentive cost | net contribution |
|---|---:|---:|---:|
| DO_NOTHING | 0.3450 | 0.00 | 761.25 |
| FLAT 10% | 0.3821 | 269.04 | 573.06 |
| REVENUEOS | 0.4377 | 44.56 | 836.55 |

Flat 10% converts better than doing
nothing and earns less.
RevenueOS spends INR 224.49/opp
less on incentives than flat discounting.

## 7. Conversion vs contribution


- Conversion-max and economics-max select **different actions in 31.6%** of TEST opportunities.
- DO_NOTHING selection rate: **28.2%**
- DO_NOTHING precision (selected & oracle-optimal): **54.8%**
- DO_NOTHING recall (of oracle DO_NOTHING cases): **60.5%**

Action distribution vs the logged historical policy:


| action                |   revenueos |   logged |
|:----------------------|------------:|---------:|
| DELAYED_RETRY         |      0.1866 |   0.0267 |
| DO_NOTHING            |      0.2816 |   0.3355 |
| FREE_SHIPPING         |      0.3796 |   0.1239 |
| HUMAN_ESCALATION      |      0.0000 |   0.0494 |
| IMMEDIATE_RETRY       |      0.0382 |   0.0423 |
| MEDIUM_DISCOUNT       |      0.0039 |   0.0954 |
| PAYMENT_LINK          |      0.0000 |   0.0456 |
| PAYMENT_METHOD_SWITCH |      0.0000 |   0.0647 |
| SMALL_DISCOUNT        |      0.1101 |   0.2165 |

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
| MODEL_CONVERSION_MAX | none         | 812.9079 | 814.9102 | 798.5287 |  845.4664 |       0.2143 |      27.1370 |        1166 |
| MODEL_CONVERSION_MAX | clip20       | 806.9830 | 814.8528 | 798.8558 |  878.3950 |       0.2143 |      20.0000 |        1166 |
| MODEL_CONVERSION_MAX | clip10       | 758.8061 | 796.0562 | 797.7947 |  982.1821 |       0.2143 |      10.0000 |        1166 |
| REVENUEOS            | none         | 811.5140 | 809.0328 | 807.1868 |  808.1619 |       0.2276 |      27.1370 |        1238 |
| REVENUEOS            | clip20       | 803.4953 | 811.6947 | 806.9021 |  863.1462 |       0.2276 |      20.0000 |        1238 |
| REVENUEOS            | clip10       | 755.7193 | 802.6338 | 811.5659 | 1009.8221 |       0.2276 |      10.0000 |        1238 |


### Bootstrap CIs (DR, unclipped, 1000 resamples)

| policy               |   dr_mean |   ci_low |   ci_high |
|:---------------------|----------:|---------:|----------:|
| DO_NOTHING           |  732.3466 | 638.1220 |  825.9264 |
| FLAT_10_PERCENT      |  508.6547 | 401.7862 |  604.5805 |
| RULES                |  797.0580 | 704.6886 |  889.2217 |
| MODEL_CONVERSION_MAX |  796.7505 | 696.3704 |  909.7340 |
| REVENUEOS            |  805.8845 | 708.4495 |  912.5754 |


`ess` is Kish effective sample size on matched rows; `match_rate` is the share
of TEST rows where the logged action equals the target policy's action. A low
match rate means the estimate rests on few rows regardless of nominal n.

## 9. Does OPE recover simulator truth?

| policy               |   oracle_value |   dr_estimate |   abs_diff |   rel_diff |
|:---------------------|---------------:|--------------:|-----------:|-----------:|
| DO_NOTHING           |       761.2542 |      733.5915 |    27.6627 |     0.0363 |
| FLAT_10_PERCENT      |       573.0587 |      508.7048 |    64.3539 |     0.1123 |
| RULES                |       819.4090 |      798.0027 |    21.4063 |     0.0261 |
| MODEL_CONVERSION_MAX |       806.2747 |      798.5287 |     7.7460 |     0.0096 |
| REVENUEOS            |       836.5513 |      807.1868 |    29.3646 |     0.0351 |


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
|         0.0200 |               0.2550 |
|         0.0500 |               0.4033 |


### Calibration impact on economics
Using raw (uncalibrated) probabilities instead of calibrated ones changes the
selected action on **16.6%** of opportunities. This is the practical
argument for calibration: miscalibrated probabilities feed directly into dEV and
shift real spending decisions.

### Regret distribution
mean INR 76.31 · median INR 0.00 · p90 INR 178.52

## 12. Error analysis


The 20 highest-regret TEST cases are in `high_regret_cases.csv`. Regret is
concentrated: the top decile accounts for
79%
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
