# RevenueOS — Backend Architecture (Phase 5/6)

## Authority chain

```mermaid
flowchart TD
    API[FastAPI] --> WF[Recovery Workflow]
    WF --> P[Frozen ML Predictor]
    P --> FE[Financial Engine]
    FE --> PE[Deterministic Policy Engine]
    PE -->|PASS| EX[Execution Provider]
    PE -->|REQUIRE_APPROVAL| HA[Awaiting Approval]
    PE -->|REJECT / STOP| ST[Stopped]
    EX --> SIM[Simulator]
    EX --> RZP[Razorpay Test Mode]
    RZP --> WH[Webhook Inbox]
    WH --> REC[Reconciler]
    REC --> DB[(PostgreSQL / SQLite)]
    WF --> AUD[Audit Store] --> DB
```

**ML predicts. Economics ranks. Policy authorizes. Razorpay executes. Webhooks
verify. Audit records.** No component assumes another's authority: the predictor
returns probabilities and nothing else, the financial engine takes no model, and
the policy engine takes no text.

## Successful recovery

```mermaid
sequenceDiagram
    participant U as Customer
    participant API as RevenueOS API
    participant M as Predictor
    participant F as Financial Engine
    participant P as Policy Engine
    participant R as Razorpay
    participant W as Webhook
    API->>M: score all eligible actions
    M-->>API: P(recovery | action) per action
    API->>F: EV and ΔEV vs DO_NOTHING
    F-->>API: ranked candidates
    API->>P: evaluate best candidate
    P-->>API: PASS + max authorized downside
    API->>API: commit idempotency key (before any external call)
    API->>R: create Test Mode order
    R-->>U: checkout
    U->>R: pays
    R->>W: payment.captured
    W->>API: verify signature over raw body
    API->>API: RECOVERED, outcome row written once
```

## Failure then recovery

```mermaid
sequenceDiagram
    participant R as Razorpay
    participant API as RevenueOS
    R->>API: payment.failed
    API->>API: PAYMENT_FAILED_RECOVERABLE (not terminal)
    API->>API: re-analyze, attempt 2, new idempotency key
    R->>API: payment.captured
    API->>API: RECOVERED
```

A failed payment is never treated as final: the same order may still be captured.

## Duplicate webhook

```mermaid
sequenceDiagram
    participant R as Razorpay
    participant API as RevenueOS
    R->>API: event evt_1 (payment.captured)
    API->>API: insert inbox row, process, RECOVERED
    R->>API: event evt_1 again (retry)
    API->>API: UNIQUE(provider, event_id) conflict
    API-->>R: 200 duplicate, no state change
```

## Tables

`opportunities`, `action_predictions`, `action_financial_evaluations`,
`policy_evaluations`, `policy_rule_evaluations`, `recovery_executions`,
`recovery_outcomes`, `payment_failures`, `audit_events`, `webhook_inbox`.

Key constraints: `recovery_executions.idempotency_key` UNIQUE;
`webhook_inbox (provider, event_id)` UNIQUE; `audit_events (opportunity_id,
sequence_number)` UNIQUE; `recovery_outcomes.opportunity_id` PRIMARY KEY, which
is what makes double-counting structurally impossible.

## Research vs live metrics

`evaluation/results/` holds frozen research evaluation on held-out TEST data.
`/api/dashboard/metrics` reports only live and seeded execution outcomes and
labels itself `LIVE_OPERATIONAL`. These are different classes of evidence and
are never summed together.