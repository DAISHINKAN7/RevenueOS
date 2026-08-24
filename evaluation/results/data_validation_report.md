# RevenueOS — Data Validation Report

Seed `42` · simulator `1.1.0` · logging policy `1.1.0`

Period: 2025-06-01 00:21:00 → 2025-11-28 00:21:00

## 1. Record counts

| table            |   rows |
|:-----------------|-------:|
| products         |     60 |
| customers        |   8000 |
| customers_hidden |   8000 |
| sessions         | 120000 |
| checkouts        |  70087 |
| payment_attempts |  44066 |
| opportunities    |  36265 |
| interventions    |  36265 |
| oracle           |  36265 |

## 2. Funnel
| stage             |      n |   rate_of_prior |
|:------------------|-------:|----------------:|
| sessions          | 120000 |        nan      |
| checkouts started |  70087 |          0.5841 |
| abandoned         |  26021 |          0.3713 |
| payment attempts  |  44066 |          0.6287 |
| payment failures  |  10244 |          0.2325 |
| opportunities     |  36265 |        nan      |


> **Interpretation.** These are not generic website conversion rates. The simulator represents commerce sessions that have already demonstrated substantial purchase intent — a high-intent cohort, not all anonymous merchant traffic. This oversampling is deliberate: it ensures adequate recovery-opportunity volume for policy learning and off-policy evaluation. Do not read `checkouts started / sessions` as a site-wide conversion rate.

## 3. Chronological split and off-policy support

| split      |   opportunities |   exploration |   exploration_rate |
|:-----------|----------------:|--------------:|-------------------:|
| TEST       |            5440 |          1363 |             0.2506 |
| TRAIN      |           25385 |          6411 |             0.2526 |
| VALIDATION |            5440 |          1394 |             0.2562 |

Held-out exploration cohort by action:

| action_taken          |   n |
|:----------------------|----:|
| DO_NOTHING            | 179 |
| FREE_SHIPPING         | 171 |
| SMALL_DISCOUNT        | 162 |
| MEDIUM_DISCOUNT       | 176 |
| PAYMENT_METHOD_SWITCH | 158 |
| IMMEDIATE_RETRY       |  78 |
| DELAYED_RETRY         |  72 |
| PAYMENT_LINK          | 175 |
| HUMAN_ESCALATION      | 192 |


Held-out rows: **5,440** · exploration rows: **1,363**

Kish ESS of uniform-target importance weights: **2,127** (max weight 51.1, min propensity 0.0196)

## 4. Propensity distribution

|       |   action_propensity |
|:------|--------------------:|
| count |          36265.0000 |
| mean  |              0.2474 |
| std   |              0.1429 |
| min   |              0.0196 |
| 1%    |              0.0197 |
| 5%    |              0.0356 |
| 50%   |              0.2069 |
| 95%   |              0.4447 |
| 99%   |              0.4638 |
| max   |              0.5024 |

## 5. Logged action distribution

| action_taken          |     n |   share |   conversion |   mean_contribution |
|:----------------------|------:|--------:|-------------:|--------------------:|
| DO_NOTHING            | 12120 |  0.3342 |       0.3460 |            661.4756 |
| SMALL_DISCOUNT        |  7841 |  0.2162 |       0.3519 |            808.6976 |
| FREE_SHIPPING         |  4560 |  0.1257 |       0.4783 |            472.1180 |
| MEDIUM_DISCOUNT       |  3580 |  0.0987 |       0.3704 |            634.1307 |
| PAYMENT_METHOD_SWITCH |  2299 |  0.0634 |       0.3201 |            748.2591 |
| HUMAN_ESCALATION      |  1736 |  0.0479 |       0.3485 |            692.8912 |
| PAYMENT_LINK          |  1637 |  0.0451 |       0.3293 |            780.5068 |
| IMMEDIATE_RETRY       |  1431 |  0.0395 |       0.4032 |            909.6085 |
| DELAYED_RETRY         |  1061 |  0.0293 |       0.4995 |           1028.3512 |


> Naive conversion rates above are **confounded** with context — the historical policy chose actions based on cart value and failure reason. They are not causal estimates and must not be read as action effectiveness.

## 6. Structure check — oracle uplift by segment

| segment               |    n |   FREE_SHIPPING |   SMALL_DISCOUNT |   MEDIUM_DISCOUNT |   PAYMENT_METHOD_SWITCH |   DELAYED_RETRY |
|:----------------------|-----:|----------------:|-----------------:|------------------:|------------------------:|----------------:|
| AT_RISK               | 3909 |          0.0898 |           0.0385 |            0.0651 |                  0.0210 |          0.0982 |
| CONVENIENCE_SENSITIVE | 5031 |          0.0834 |          -0.0357 |           -0.0156 |                 -0.0416 |          0.1243 |
| DEAL_SEEKER           | 5118 |          0.1046 |           0.0925 |            0.1389 |                  0.0143 |          0.1129 |
| HIGH_LTV              | 2773 |         -0.0315 |          -0.0930 |           -0.0799 |                 -0.0758 |          0.1294 |
| LOYAL                 | 3702 |         -0.0576 |          -0.1002 |           -0.0851 |                 -0.0935 |          0.1181 |
| NEW_CUSTOMER          | 4586 |          0.0744 |           0.0181 |            0.0450 |                 -0.0011 |          0.1275 |
| PAYMENT_FRICTION      | 4540 |          0.0365 |          -0.0082 |            0.0129 |                  0.0272 |          0.1038 |
| PRICE_SENSITIVE       | 6606 |          0.0638 |           0.0658 |            0.1080 |                 -0.0042 |          0.1224 |


> True ΔP(recovery) vs DO_NOTHING. Distinct rows here are the signal the model is supposed to learn. Flat rows mean the simulator has no structure.

## 7a. Oracle best action — PROBABILITY (argmax P(recovery|a))

| prob_max_action       |   share |
|:----------------------|--------:|
| FREE_SHIPPING         |  0.3855 |
| MEDIUM_DISCOUNT       |  0.3212 |
| DELAYED_RETRY         |  0.1260 |
| DO_NOTHING            |  0.0990 |
| PAYMENT_METHOD_SWITCH |  0.0478 |
| HUMAN_ESCALATION      |  0.0198 |
| PAYMENT_LINK          |  0.0008 |


## 7b. Oracle best action — ECONOMICS (argmax dEV, incl. DO_NOTHING)

| ev_max_action         |   share |
|:----------------------|--------:|
| FREE_SHIPPING         |  0.3275 |
| DO_NOTHING            |  0.2595 |
| DELAYED_RETRY         |  0.1588 |
| SMALL_DISCOUNT        |  0.1313 |
| PAYMENT_METHOD_SWITCH |  0.0675 |
| MEDIUM_DISCOUNT       |  0.0217 |
| HUMAN_ESCALATION      |  0.0173 |
| PAYMENT_LINK          |  0.0165 |


Computed on 4,000 sampled opportunities via the canonical financial engine.

## 8. Intelligent restraint

| metric                                                                |   value |
|:----------------------------------------------------------------------|--------:|
| share_of_opportunities_where_all_interventions_have_delta_ev_lte_zero |  0.2595 |
| share_where_probability_max_action_differs_from_ev_max_action         |  0.4100 |
| share_where_DO_NOTHING_is_economics_optimal                           |  0.2595 |


> The second row is the headline RevenueOS metric: on that share of opportunities, maximising conversion and maximising contribution select **different actions**.

## 9. Flat-10%-discount sanity comparison

Expected values under three fixed strategies, same opportunities:

| strategy            |   conversion |   recovered_gmv_net |   incentive_cost |   net_contribution |
|:--------------------|-------------:|--------------------:|-----------------:|-------------------:|
| DO_NOTHING          |       0.3445 |           2323.7175 |           0.0000 |           732.3407 |
| FLAT_10PCT_DISCOUNT |       0.3825 |           2352.5125 |         263.3903 |           558.9104 |
| ORACLE_ECONOMIC     |       0.4601 |           2888.9240 |          46.9968 |           876.2076 |


- Flat 10% converts better than the economic action in **32.6%** of opportunities

- Flat 10% yields lower net contribution in **97.8%**

- **Both simultaneously (converts better, earns less): 32.6%** — these are the cases that demonstrate the thesis

## 10. Key distributions

|       |   cart_value |   shipping_fee_charged |   shipping_cost |   base_contribution_margin |   minutes_since_event |
|:------|-------------:|-----------------------:|----------------:|---------------------------:|----------------------:|
| count |     36265.00 |               36265.00 |        36265.00 |                   36265.00 |              36265.00 |
| mean  |      7625.47 |                  45.13 |          138.68 |                    2382.39 |                 17.54 |
| std   |      9537.61 |                  59.34 |          155.14 |                    2892.54 |                  8.37 |
| min   |       309.00 |                   0.00 |           37.21 |                     104.11 |                  2.00 |
| 25%   |      1977.00 |                   0.00 |           56.45 |                     609.16 |                 11.00 |
| 50%   |      4338.00 |                  34.00 |           93.42 |                    1261.60 |                 16.00 |
| 75%   |      9738.00 |                  68.97 |          155.72 |                    3015.38 |                 24.00 |
| max   |     87236.00 |                 482.72 |         2659.74 |                   23707.08 |                 38.00 |


Contribution margin %: mean 33.2%, p05 21.5%, p95 49.5%


### Opportunity type mix

| opportunity_type     |     n |
|:---------------------|------:|
| CHECKOUT_ABANDONMENT | 26021 |
| PAYMENT_FAILURE      | 10244 |


### Failure reason mix (payment failures)

| failure_reason         |    n |
|:-----------------------|-----:|
| INSUFFICIENT_FUNDS     | 1879 |
| UPI_TIMEOUT            | 1650 |
| CARD_DECLINED          | 1522 |
| BANK_TIMEOUT           | 1448 |
| AUTHENTICATION_FAILURE | 1107 |
| USER_CANCELLED         | 1001 |
| NETWORK_ERROR          |  996 |
| UNKNOWN                |  641 |

## 11. Hidden environmental mechanisms

These windows shift outcomes but are never exposed as features:

```json
{
  "bank_outages": [
    [
      "2025-06-17 00:00:00",
      "2025-06-17 09:00:00"
    ],
    [
      "2025-08-17 18:00:00",
      "2025-08-18 03:00:00"
    ],
    [
      "2025-08-18 20:00:00",
      "2025-08-19 05:00:00"
    ],
    [
      "2025-09-26 13:00:00",
      "2025-09-26 22:00:00"
    ],
    [
      "2025-10-18 00:00:00",
      "2025-10-18 09:00:00"
    ],
    [
      "2025-11-02 05:00:00",
      "2025-11-02 14:00:00"
    ]
  ],
  "competitor_sales": [
    [
      "2025-06-16 05:00:00",
      "2025-06-19 05:00:00"
    ],
    [
      "2025-06-17 16:00:00",
      "2025-06-20 16:00:00"
    ],
    [
      "2025-07-06 15:00:00",
      "2025-07-09 15:00:00"
    ],
    [
      "2025-10-02 10:00:00",
      "2025-10-05 10:00:00"
    ]
  ],
  "courier_disruptions": [
    [
      "2025-09-01 15:00:00",
      "2025-09-05 15:00:00"
    ],
    [
      "2025-10-08 11:00:00",
      "2025-10-12 11:00:00"
    ],
    [
      "2025-11-19 17:00:00",
      "2025-11-23 17:00:00"
    ]
  ],
  "payday_days": [
    1,
    2,
    3,
    28,
    29,
    30
  ]
}
```

## 12. Leakage checks

- PASS — no hidden_* columns in observable customers
- PASS — oracle stored in a separate quarantined file
- PASS — opportunities sorted chronologically
- PASS — all propensities > 0
- PASS — all propensities <= 1
- PASS — train precedes validation precedes test

## 13. Is the simulator too easy?

- Oracle P(recovery) mean 0.360, sd 0.162

- Realised conversion rate: 0.371


> The response surface carries shared logit noise plus unobserved environment windows. If a trained model later reaches test ROC-AUC above ~0.85 or near-perfect calibration, treat that as a simulator defect and raise `response_logit_noise_sd`.

## Verdict

**No blocking warnings. Dataset is ready for feature engineering (Phase 3).**
