# RevenueOS — Model Card

## Model overview
Action-conditioned recovery-probability model estimating `P(recovery | context, action)`
for ecommerce checkout abandonment and payment failure. XGBoost with isotonic
calibration. Model hash `cdfba209ae9dc97e`, trained on simulator `1.1.0`, seed 42.

## Intended use
Scoring candidate recovery actions so a separate deterministic financial engine
can rank them by incremental expected contribution margin. The model produces
probabilities only; it never selects actions and never authorizes spending.

## Non-intended use
Not for real-merchant deployment. Trained entirely on a synthetic behavioural
environment; its probabilities reflect modelled assumptions, not measured
customer behaviour. Not a causal-inference tool. Not for credit, pricing, or any
decision about an individual person's eligibility.

## Training data
25,385 logged recovery opportunities, one row per (opportunity,
historically-taken action). Counterfactual expansion using oracle labels was
explicitly NOT performed — the model must infer action response from logged
decisions under a confounded historical policy.

## Temporal split
Chronological. TRAIN ends 2025-10-04, VALIDATION ends 2025-10-31,
TEST runs to 2025-11-28. 5,440 validation / 5,440 test rows.

## Features
58 features across six groups: customer history, action-response
history (finite counts with empirical-Bayes smoothing, alpha=5), checkout
context, payment context, temporal context, and candidate-action economics.

## Excluded features
All `hidden_*` simulator latents, all oracle probabilities, all realised outcomes.
Enforced programmatically by `assert_no_leakage` and by `tests/test_phase3.py`.

## Target
Binary recovery of the checkout/payment following the logged intervention.
Training prevalence 0.371. No class weighting applied —
prevalence is not severely imbalanced and weighting degraded calibration.

## Calibration
Isotonic, fitted on VALIDATION predictions only. TEST was never used for
fitting, tuning, or threshold selection. Selection was calibration-first
(Brier, then ECE); isotonic was not assumed superior.

## Held-out metrics (TEST)
| metric | value |
|---|---|
| ROC-AUC | 0.5942 |
| PR-AUC | 0.4593 |
| Brier | 0.2287 |
| Log loss | 0.6586 |
| ECE | 0.0204 |

ROC-AUC is modest by classification standards and this is expected: the
simulator injects shared logit noise and hidden environment windows that cap
achievable accuracy. Calibration and relative treatment response matter more
here than ranking, because decisions are expected-value comparisons.

## Action-specific response (synthetic oracle)
Predicted vs true dP correlation is strongest for discounts (Pearson ~0.48) and
free shipping (~0.33), weak for retries (~0.15-0.22), and **fails for
PAYMENT_METHOD_SWITCH (Pearson -0.046)**. See limitations.

## Policy evaluation (synthetic oracle, TEST)
| policy | conversion | incentive cost/opp | net contribution/opp |
|---|---:|---:|---:|
| DO_NOTHING | 0.3450 | 0.00 | 761.25 |
| FLAT_10_PERCENT | 0.3821 | 269.04 | 573.06 |
| RULES | 0.4153 | 56.43 | 819.41 |
| MODEL_CONVERSION_MAX | 0.4560 | 112.80 | 806.15 |
| REVENUEOS | 0.4385 | 45.81 | 838.48 |
| ORACLE_ECONOMIC | 0.4634 | 44.82 | 912.86 |

## Off-policy evaluation
Doubly robust estimate for RevenueOS: computed from logged actions,
propensities and outcomes only, with cross-fitted reward model (K=5). Agreement
with the oracle is within 2.7% for RevenueOS and within 11% for all policies.
Clipping sensitivity (unclipped / clip20 / clip10) is reported in full.

## Ablations
Removing action-response history degrades oracle dP MAE; removing customer
history, economic context, or time context has negligible effect on Brier. This
suggests the response signal the model actually uses is narrower than the full
feature set implies.

## Robustness
Perturbing probabilities by ±0.02 changes the selected action on ~25% of
opportunities; ±0.05 changes ~40%. Decisions are frequently close calls in dEV
terms, which is why a conservative variant with a minimum-dEV threshold is
provided.

## Known limitations
1. **PAYMENT_METHOD_SWITCH response is not learned.** Near-zero correlation with
   oracle dP and a negative bias, so the policy never selects it despite it being
   genuinely effective for CARD_DECLINED. This costs real contribution.
2. **PAYMENT_LINK and HUMAN_ESCALATION are never selected**, so the effective
   action space is narrower than the nominal nine.
3. Synthetic environment throughout; results are policy evaluation under a
   documented behavioural model, not real-world causal uplift.
4. Off-policy estimates depend on overlap; match rates range 10-34%.
5. Cold-start customers carry population priors by design and are harder to score.
6. Decision instability under small probability perturbation is non-trivial.

## Ethical and financial considerations
The model cannot spend money. Every monetary action passes a deterministic
policy engine that the model cannot influence, and maximum downside per
opportunity is bounded and recorded. Discount decisions are made on economics,
not on any protected or inferred demographic attribute. All customers are
synthetic; no real personal or financial data is used.