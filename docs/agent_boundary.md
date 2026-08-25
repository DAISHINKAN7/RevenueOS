# Agent Authority Boundary

> The LLM planner decides which bounded workflow tool to request next based on
> current state and evidence. It cannot select or override monetary amounts,
> recovery probabilities, expected-value calculations, policy decisions, payment
> states, workflow transitions, or Razorpay authorization.

## Allowed

- Read workflow state, policy summary, audit summary, payment history
- Request a structured diagnosis
- Request analysis (scoring + economics + policy check) as one bounded operation
- Request execution of an action **policy has already authorized**
- Flag that human approval is needed
- Choose `WAIT` when external evidence is pending
- Choose `STOP` when the opportunity is terminal or no safe step remains

## Forbidden

| The planner cannot | Enforced by |
|---|---|
| Invent an action type | `Literal` in `PlannerDecision.next_tool` |
| Set a discount or amount | argument screening in `AgentToolAuthorizer` |
| Change model probabilities | no tool accepts a probability |
| Alter ΔEV or policy outcomes | financial engine and policy engine take no planner input |
| Authorize its own execution | `request_execution` re-reads the policy row server-side |
| Mark a payment successful | payment state changes only on verified provider evidence |
| Change workflow state | transitions are backend-owned |
| Call Razorpay directly | no client access in any planner-facing tool |
| Modify audit history | audit is append-only |

## Tool surface

Eight tools plus two verbs. Read-only: `get_opportunity_state`,
`get_policy_summary`, `get_audit_summary`. Advisory:
`diagnose_recovery_context`. Mutating: `analyze_opportunity`,
`request_execution`, `request_human_approval`, `stop_workflow`.

`request_execution` accepts no action, no amount and no approval flag. It names
a policy-evaluation id; the backend re-reads that evaluation, re-checks the
workflow state, confirms the action still matches, and executes only what policy
authorized.

## State gating

Tools are permitted per workflow state. `request_execution` is reachable from
exactly one state — `AUTHORIZED` — which only the policy engine can produce.
Terminal states (`RECOVERED`, `NOT_RECOVERED`, `STOPPED`, `EXPIRED`,
`ESCALATED`) expose read-only tools only.

## Approval behaviour

When policy returns `REQUIRE_APPROVAL` the workflow moves to
`AWAITING_APPROVAL`, where `request_execution` is not permitted. The agent may
call `request_human_approval`, which records the need and explicitly returns
`approved: false`. Only an authenticated operator acting through the API can
approve.

## Failure behaviour

| Failure | Response |
|---|---|
| Planner timeout or error | `AGENT_PLANNER_FAILED`, deterministic fallback router |
| Malformed or unparseable output | `AGENT_OUTPUT_INVALID`, deterministic fallback |
| Unknown tool name | schema rejection; never executed |
| Unauthorized tool for the state | `AGENT_TOOL_BLOCKED`, then replan |
| Budget exhausted | `STOPPED_BUDGET`, no financial action |
| No progress across calls | `STOPPED_NO_PROGRESS` |

With `AGENT_LLM_ENABLED=false` the deterministic backend behaves exactly as it
did before the agent layer existed. The agent is an enhancement, never a
dependency for correctness.

## Audit behaviour

Every proposal is recorded as a chain: `AGENT_TOOL_PROPOSED` →
`AGENT_TOOL_ALLOWED` or `AGENT_TOOL_BLOCKED` → `AGENT_TOOL_RESULT`. Blocked
calls carry the state, the permitted tool set, the reason and an arguments hash.
Only concise decision summaries are persisted — never hidden chain-of-thought.