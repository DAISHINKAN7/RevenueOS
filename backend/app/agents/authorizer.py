"""Planner schema and tool authorization layer (spec §5-10, §48-50).

This module is the choke point between the language model and every
deterministic system behind it. Three separate defences, in order:

1. **Schema validation** — the planner's output is parsed into a Pydantic model
   whose `next_tool` is a `Literal`. An unknown tool name cannot survive
   parsing, so a hallucinated tool never reaches a lookup.
2. **State gating** — a tool valid in one workflow state is refused in another.
   The language model is never trusted to respect the state machine.
3. **Argument screening** — any argument whose name resembles a monetary or
   authorization override is rejected outright, even on an allowed tool.

The planner chooses *which permitted operation happens next*. It cannot choose
amounts, probabilities, policy outcomes, payment states, or transitions.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from backend.app.domain import State

AUTHORIZER_VERSION = "agent-tool-authorizer-1.0.0"

# The complete LLM-facing surface. Deliberately small: every entry here is an
# attack surface, so tools exist only where the planner genuinely needs a choice.
PlannerTool = Literal[
    "get_opportunity_state",
    "diagnose_recovery_context",
    "analyze_opportunity",
    "request_execution",
    "request_human_approval",
    "stop_workflow",
    "get_audit_summary",
    "get_policy_summary",
    "WAIT",
    "STOP",
]

PLANNER_TOOLS: frozenset[str] = frozenset({
    "get_opportunity_state", "diagnose_recovery_context", "analyze_opportunity",
    "request_execution", "request_human_approval", "stop_workflow",
    "get_audit_summary", "get_policy_summary",
})

READ_ONLY_TOOLS: frozenset[str] = frozenset({
    "get_opportunity_state", "get_audit_summary", "get_policy_summary",
})

MUTATING_TOOLS: frozenset[str] = PLANNER_TOOLS - READ_ONLY_TOOLS - {
    "diagnose_recovery_context"}

VERBS: frozenset[str] = frozenset({"WAIT", "STOP"})

# Argument names that would represent a financial or authorization override.
# No LLM-facing tool may accept any of these, on any tool, ever (spec §48).
FORBIDDEN_ARGUMENT_SUBSTRINGS: tuple[str, ...] = (
    "amount", "discount", "probability", "expected_value", "delta_ev", "ev",
    "payment_status", "workflow_state", "state", "policy", "override",
    "approved", "authorize", "authorized", "price", "cost", "margin",
    "capture", "refund", "razorpay", "order_id", "payment_id",
)


class Disposition(str, Enum):
    """Explicit agent outcomes (spec §41). Never arbitrary strings."""

    COMPLETED_RECOVERED = "COMPLETED_RECOVERED"
    WAITING_AWAITING_PAYMENT = "WAITING_AWAITING_PAYMENT"
    WAITING_FOR_HUMAN_APPROVAL = "WAITING_FOR_HUMAN_APPROVAL"
    STOPPED_POLICY = "STOPPED_POLICY"
    STOPPED_TERMINAL = "STOPPED_TERMINAL"
    STOPPED_BUDGET = "STOPPED_BUDGET"
    STOPPED_NO_PROGRESS = "STOPPED_NO_PROGRESS"
    FAILED_PLANNER = "FAILED_PLANNER"
    FAILED_TOOL = "FAILED_TOOL"
    REPLANNED = "REPLANNED"
    TERMINAL_NO_ACTION = "TERMINAL_NO_ACTION"


class PlannerDecision(BaseModel):
    """Schema-constrained planner output (spec §9).

    `next_tool` is a Literal, so an unknown tool fails validation rather than
    reaching the authorizer. `model_config` forbids extra fields, which is what
    blocks an attempt to smuggle `discount_percent` alongside a valid tool.
    """

    model_config = {"extra": "forbid"}

    observation: str = Field(default="", max_length=600)
    next_tool: PlannerTool
    reason: str = Field(default="", max_length=300)
    expected_state: str | None = Field(default=None, max_length=64)

    @classmethod
    def parse_planner_output(cls, raw: dict) -> tuple["PlannerDecision | None", str | None]:
        """Return (decision, error). Never raises: a bad planner is expected."""
        if not isinstance(raw, dict):
            return None, "planner output was not a JSON object"

        # Tolerate the older field names without accepting unknown extras.
        normalized = {
            "observation": str(raw.get("observation")
                               or raw.get("observation_summary") or "")[:600],
            "next_tool": raw.get("next_tool") or raw.get("next_step"),
            "reason": str(raw.get("reason") or raw.get("reason_code") or "")[:300],
            "expected_state": raw.get("expected_state"),
        }
        # Anything else the planner sent is an override attempt or noise; record
        # it for the audit trail rather than silently discarding it.
        extras = sorted(set(raw) - {
            "observation", "observation_summary", "next_tool", "next_step",
            "reason", "reason_code", "expected_state",
            "requires_financial_authorization", "goal", "tool_input"})
        try:
            decision = cls(**normalized)
        except ValidationError as exc:
            first = exc.errors()[0]
            return None, f"{'.'.join(str(x) for x in first['loc'])}: {first['msg']}"[:200]
        if extras:
            return decision, f"ignored unexpected planner fields: {extras}"
        return decision, None


@dataclass(frozen=True)
class AuthorizationResult:
    allowed: bool
    tool: str
    state: str
    allowed_tools: tuple[str, ...]
    reason: str
    arguments_hash: str

    def as_dict(self) -> dict:
        return {
            "allowed": self.allowed, "tool": self.tool, "state": self.state,
            "allowed_tools": list(self.allowed_tools), "reason": self.reason,
            "arguments_hash": self.arguments_hash,
            "authorizer_version": AUTHORIZER_VERSION,
        }


class AgentToolAuthorizer:
    """Deterministic state → permitted tool mapping.

    Terminal states expose read-only tools only, so no mutation is reachable
    once an opportunity is settled. `request_execution` appears in exactly one
    state — AUTHORIZED — which is the state only the policy engine can produce.
    """

    STATE_TOOLS: dict[str, frozenset[str]] = {
        State.DETECTED.value: frozenset({
            "get_opportunity_state", "diagnose_recovery_context",
            "analyze_opportunity", "stop_workflow"}),
        State.ANALYZING.value: frozenset({"get_opportunity_state"}),
        State.CANDIDATES_SCORED.value: frozenset({"get_opportunity_state"}),
        State.ECONOMICALLY_RANKED.value: frozenset({"get_opportunity_state"}),
        State.POLICY_CHECKED.value: frozenset({
            "get_opportunity_state", "get_policy_summary"}),
        # Execution is reachable only from the state the policy engine sets.
        State.AUTHORIZED.value: frozenset({
            "get_opportunity_state", "get_policy_summary",
            "request_execution", "stop_workflow"}),
        # Approval must come from a human; the agent may only look and wait.
        State.AWAITING_APPROVAL.value: frozenset({
            "get_opportunity_state", "get_policy_summary",
            "request_human_approval"}),
        State.AWAITING_PAYMENT.value: frozenset({"get_opportunity_state"}),
        State.EXECUTING.value: frozenset({"get_opportunity_state"}),
        State.EXECUTION_PENDING.value: frozenset({"get_opportunity_state"}),
        State.PAYMENT_FAILED_RECOVERABLE.value: frozenset({
            "get_opportunity_state", "diagnose_recovery_context",
            "analyze_opportunity", "stop_workflow"}),
        State.EXECUTION_FAILED.value: frozenset({
            "get_opportunity_state", "diagnose_recovery_context",
            "analyze_opportunity", "stop_workflow"}),
        State.NOT_RECOVERED.value: frozenset({
            "get_opportunity_state", "get_audit_summary"}),
        # Terminal: read-only, no exceptions.
        State.RECOVERED.value: frozenset({"get_opportunity_state", "get_audit_summary"}),
        State.STOPPED.value: frozenset({"get_opportunity_state", "get_audit_summary"}),
        State.EXPIRED.value: frozenset({"get_opportunity_state", "get_audit_summary"}),
        State.ESCALATED.value: frozenset({"get_opportunity_state", "get_audit_summary"}),
    }

    TERMINAL_STATES = frozenset({
        State.RECOVERED.value, State.NOT_RECOVERED.value, State.STOPPED.value,
        State.EXPIRED.value, State.ESCALATED.value})

    WAIT_STATES = frozenset({
        State.AWAITING_PAYMENT.value, State.AWAITING_APPROVAL.value,
        State.EXECUTING.value, State.EXECUTION_PENDING.value})

    # Reads whose answer cannot change while the workflow state is unchanged.
    # Re-offering one after it has been called invites the planner to dither
    # instead of taking the single productive step available to it.
    IDEMPOTENT_READS: frozenset[str] = frozenset({
        "get_opportunity_state", "get_policy_summary", "get_audit_summary",
        "diagnose_recovery_context",
    })

    def allowed_tools_for_state(self, state: str,
                                already_called: set[str] | None = None) -> set[str]:
        tools = set(self.STATE_TOOLS.get(state, {"get_opportunity_state"}))
        if already_called:
            tools -= (set(already_called) & self.IDEMPOTENT_READS)
        # Never return an empty menu: with nothing left to read, a read is still
        # a harmless answer, and the verbs remain available regardless.
        return tools or {"get_opportunity_state"}

    @staticmethod
    def screen_arguments(arguments: dict | None) -> str | None:
        """Reject any argument resembling a financial or authorization override."""
        if not arguments:
            return None
        for key in arguments:
            low = str(key).lower()
            for bad in FORBIDDEN_ARGUMENT_SUBSTRINGS:
                if bad in low:
                    return f"argument {key!r} resembles a financial or authorization override"
        blob = json.dumps(arguments, default=str)
        if len(blob) > 2000:
            return "arguments payload is oversized"
        return None

    def authorize(self, tool: str, state: str, arguments: dict | None = None,
                  already_called: set[str] | None = None) -> AuthorizationResult:
        allowed_tools = self.allowed_tools_for_state(state, already_called)
        args_hash = hashlib.sha256(
            json.dumps(arguments or {}, sort_keys=True, default=str).encode()
        ).hexdigest()[:16]

        def result(ok: bool, reason: str) -> AuthorizationResult:
            return AuthorizationResult(
                allowed=ok, tool=tool, state=state,
                allowed_tools=tuple(sorted(allowed_tools)), reason=reason,
                arguments_hash=args_hash)

        if tool in VERBS:
            return result(True, f"{tool} is always a permitted planner decision")
        if tool not in PLANNER_TOOLS:
            return result(False, f"{tool!r} is not on the planner tool allowlist")

        arg_error = self.screen_arguments(arguments)
        if arg_error:
            return result(False, arg_error)

        if state in self.TERMINAL_STATES and tool in MUTATING_TOOLS:
            return result(False,
                          f"{tool} is a mutating tool and {state} is terminal")
        if tool not in allowed_tools:
            return result(False,
                          f"{tool} is not permitted in state {state}; "
                          f"permitted: {sorted(allowed_tools)}")
        return result(True, f"{tool} is permitted in state {state}")


def fallback_decision(state: str, already_called: set[str] | None = None) -> PlannerDecision:
    """Deterministic state router (spec §24).

    Proves RevenueOS stays operational with no language model at all. Used when
    the planner times out, returns malformed output, or proposes something the
    authorizer refuses.
    """
    called = already_called or set()
    auth = AgentToolAuthorizer()

    if state in auth.TERMINAL_STATES:
        return PlannerDecision(observation=f"State {state} is terminal.",
                               next_tool="STOP", reason="TERMINAL_STATE")
    if state in auth.WAIT_STATES:
        return PlannerDecision(
            observation=f"State {state} depends on an external party.",
            next_tool="WAIT", reason="EXTERNAL_PENDING")
    if state in (State.DETECTED.value, State.PAYMENT_FAILED_RECOVERABLE.value,
                 State.EXECUTION_FAILED.value):
        if "diagnose_recovery_context" not in called:
            return PlannerDecision(
                observation="Gathering structured diagnosis before acting.",
                next_tool="diagnose_recovery_context", reason="NEED_DIAGNOSIS")
        return PlannerDecision(
            observation="Diagnosis complete; scoring actions and checking policy.",
            next_tool="analyze_opportunity", reason="READY_TO_SCORE")
    if state == State.AUTHORIZED.value:
        return PlannerDecision(
            observation="Policy authorized an action; requesting execution.",
            next_tool="request_execution", reason="POLICY_AUTHORIZED")
    return PlannerDecision(observation=f"No safe step available in {state}.",
                           next_tool="STOP", reason="NO_SAFE_STEP")