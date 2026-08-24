# RevenueOS — Product Specification (frozen contract)

Changes to this document after Phase 2 require an explicit, recorded reason.

---

## 1. What it is

RevenueOS detects where ecommerce revenue is slipping away, diagnoses why,
estimates which recovery intervention creates the greatest **incremental
contribution margin**, enforces merchant financial policy, executes through
Razorpay-compatible workflows, and verifies whether the money was recovered.

Primary track: **Track 03 — AI Revenue Recovery**.
Differentiator: a narrowly scoped Track 01 agentic-commerce extension reusing
the same economics and policy layer.

## 2. Scope (frozen)

In scope — exactly two revenue-loss classes:

1. **Checkout abandonment**
2. **Payment failure**

Out of scope: CRM, churn platform, loyalty, marketing automation, returns
optimisation, subscriptions, merchant prospecting, multi-merchant SaaS.

## 3. Core loop

```
DETECT → DIAGNOSE → GENERATE CANDIDATES → PREDICT RESPONSE
→ CALCULATE ΔEV → POLICY CHECK → ACT → VERIFY → MEASURE → AUDIT
```

## 4. Governing principle

> ML predicts. Economics ranks. LLM explains. Policy authorizes.
> Razorpay executes. Webhooks verify. Audit records everything.

The LLM **may propose**. It **may not authorize money**.

## 5. Financial objective (canonical)

```
EV(a) = P(recovery | context, a)
        * ( base_contribution_margin
            - incentive_cost_if_recovered(a)
            - expected_return_loss(a)
            - expected_cancellation_loss(a) )
        - fixed_action_cost(a)

ΔEV(a) = EV(a) - EV(DO_NOTHING)
```

Selection is by highest **positive** ΔEV. If no action has ΔEV > 0, select
`DO_NOTHING`. Incentive costs are conditional on recovery; fixed costs are not;
an incentive is subtracted exactly once; reported GMV is net of discount.

Implemented in `ml/financial_engine.py`, covered by 28 unit tests.

## 6. Action space (closed)

`DO_NOTHING`, `FREE_SHIPPING`, `SMALL_DISCOUNT`, `MEDIUM_DISCOUNT`,
`PAYMENT_METHOD_SWITCH`, `IMMEDIATE_RETRY`, `DELAYED_RETRY`, `PAYMENT_LINK`,
`HUMAN_ESCALATION`.

Retry actions are ineligible for pure abandonment. `MEMBERSHIP_OFFER` is a
stretch goal, deliberately excluded.

## 7. Merchant policy (deterministic)

```json
{
  "max_autonomous_discount_percent": 7,
  "max_autonomous_discount_amount": 300,
  "max_free_shipping_cost": 150,
  "minimum_contribution_margin_percent": 15,
  "max_recovery_attempts": 2,
  "minimum_action_confidence": 0.65,
  "human_approval_required_above_discount": 250
}
```

Outcomes: `PASS` / `REJECT` / `REQUIRE_APPROVAL`. Every decision names the exact
rule triggered. The proposal layer *is allowed* to propose violating actions —
blocked-violation counts are what prove the gate is load-bearing.

## 8. State machine

```
DETECTED → DIAGNOSING → CANDIDATES_GENERATED → SCORED → OPTIMIZED
→ POLICY_CHECKED → [AWAITING_APPROVAL] → EXECUTING → AWAITING_OUTCOME → RECOVERED
```

Terminal: `RECOVERED`, `NOT_RECOVERED`, `STOPPED`, `ESCALATED`, `EXPIRED`,
`EXECUTION_FAILED`.

## 9. Protected features (never cut)

Opportunity Detail screen · Audit Timeline · ΔEV optimisation · calibrated
probabilities · propensity logging · exploration cohort · off-policy evaluation
· Razorpay test flow · policy engine · adversarial tests · baseline comparison.

## 10. Cut from MVP

Membership offers · customer clustering · LangGraph · MLflow · DVC · uplift/
T-learner · contextual bandits · forecasting · what-if simulator · voice ·
WhatsApp.

## 11. Phase status

| Phase | Status |
|---|---|
| 1 — Contract freeze + financial engine | **Done** |
| 2 — Data simulator + validation gate | **Done** |
| 3 — Features + model + calibration | Next |
| 4 — Policy engine + adversarial suite | |
| 5 — Off-policy + oracle evaluation | |
| 6 — Razorpay Test Mode | |
| 7 — Frontend (detail + audit first) | |
| 8 — Agentic commerce (timeboxed) | |
