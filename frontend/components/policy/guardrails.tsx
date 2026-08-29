"use client";

import { useState } from "react";
import { AlertTriangle, Check, ChevronRight, Shield, X } from "lucide-react";
import { Card, PolicyBadge, Tip } from "@/components/ui/primitives";
import { formatINR, titleCase } from "@/lib/format";
import type { PolicyDecision } from "@/lib/types";

const RULE_LABELS: Record<string, string> = {
  RULE_WORKFLOW_STATE_VALID: "Workflow state valid",
  RULE_INVALID_MODEL_OUTPUT: "Model output valid",
  RULE_DUPLICATE_ACTION_PREVENTION: "No duplicate execution",
  RULE_MAX_RECOVERY_ATTEMPTS: "Within attempt limit",
  RULE_OPPORTUNITY_EXPIRED: "Opportunity not expired",
  RULE_CUSTOMER_DECLINED: "Customer has not declined",
  RULE_MINIMUM_DELTA_EV: "ΔEV threshold met",
  RULE_MINIMUM_MODEL_CONFIDENCE: "Recovery probability above minimum",
  RULE_MINIMUM_DECISION_MARGIN: "Decision margin sufficient",
  RULE_DISCOUNT_PERCENT_LIMIT: "Discount percent within limit",
  RULE_DISCOUNT_AMOUNT_LIMIT: "Discount amount within limit",
  RULE_HUMAN_APPROVAL_DISCOUNT_AMOUNT: "Below discount approval threshold",
  RULE_FREE_SHIPPING_LIMIT: "Shipping subsidy within limit",
  RULE_MINIMUM_MARGIN: "Contribution margin protected",
  RULE_HIGH_VALUE_REQUIRES_APPROVAL: "Below high-value approval threshold",
};

export function PolicyPanel({ policy }: { policy: PolicyDecision | null }) {
  const [expanded, setExpanded] = useState<string | null>(null);

  if (!policy)
    return (
      <Card title="Policy guardrails">
        <p className="text-[12px] text-muted-dim">
          No policy evaluation yet. Run analysis to score actions and apply merchant policy.
        </p>
      </Card>
    );

  const failed = policy.rules.filter((r) => !r.passed);
  const passed = policy.rules.length - failed.length;

  return (
    <Card
      title="Policy guardrails"
      subtitle={`${passed} of ${policy.rules.length} rules passed · ${policy.policy_version}`}
      actions={<PolicyBadge status={policy.decision} />}
    >
      <div className="mb-4 rounded-lg border border-ink-700 bg-ink-850 p-3.5">
        <div className="flex items-center gap-1.5 label">
          Maximum authorized downside
          <Tip text="The most this action can cost the merchant. RevenueOS will not commit more without violating policy." />
        </div>
        <div className="mt-1.5 text-metric font-semibold tabular-nums text-[#e6e9ef]">
          {formatINR(policy.maximum_authorized_downside)}
        </div>
      </div>

      {policy.decision !== "PASS" && (
        <div className={`mb-4 flex items-start gap-2.5 rounded-lg border p-3 ${
          policy.decision === "REQUIRE_APPROVAL"
            ? "border-warn/30 bg-warn-soft" : "border-neg/30 bg-neg-soft"}`}>
          <AlertTriangle size={14} className={
            policy.decision === "REQUIRE_APPROVAL" ? "mt-0.5 text-warn" : "mt-0.5 text-neg"} />
          <div>
            <div className="text-[12px] font-medium text-[#e6e9ef]">
              {policy.decision === "REQUIRE_APPROVAL"
                ? "Human approval required" : "Action blocked by policy"}
            </div>
            <div className="mono mt-0.5 text-muted">{policy.reason_code}</div>
          </div>
        </div>
      )}

      <ul className="space-y-0.5">
        {policy.rules.map((rule) => {
          const open = expanded === rule.rule_id;
          return (
            <li key={rule.rule_id}>
              <button onClick={() => setExpanded(open ? null : rule.rule_id)}
                className="flex w-full items-center gap-2.5 rounded-md px-2 py-1.5 text-left hover:bg-ink-850">
                {rule.passed
                  ? <Check size={13} className="shrink-0 text-pos" />
                  : <X size={13} className="shrink-0 text-neg" />}
                <span className={`flex-1 text-[12px] ${rule.passed ? "text-muted" : "text-neg"}`}>
                  {RULE_LABELS[rule.rule_id] ?? titleCase(rule.rule_id.replace("RULE_", ""))}
                </span>
                {rule.input && (
                  <span className="mono shrink-0 text-muted-dim">
                    {rule.input}{rule.threshold && ` / ${rule.threshold}`}
                  </span>
                )}
                <ChevronRight size={12}
                  className={`shrink-0 text-muted-dim transition-transform ${open ? "rotate-90" : ""}`} />
              </button>
              {open && (
                <div className="ml-6 mb-1 rounded-md border border-ink-700 bg-ink-850 p-3 animate-fade-up">
                  <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1.5 text-[11px]">
                    <dt className="text-muted-dim">Rule</dt>
                    <dd className="mono text-muted">{rule.rule_id}</dd>
                    <dt className="text-muted-dim">Input</dt>
                    <dd className="tabular-nums text-[#e6e9ef]">{rule.input ?? "—"}</dd>
                    <dt className="text-muted-dim">Threshold</dt>
                    <dd className="tabular-nums text-[#e6e9ef]">{rule.threshold ?? "—"}</dd>
                    <dt className="text-muted-dim">Result</dt>
                    <dd className={rule.passed ? "text-pos" : "text-neg"}>
                      {rule.passed ? "PASS" : rule.decision}
                    </dd>
                    <dt className="text-muted-dim">Reason</dt>
                    <dd className="text-muted">{rule.reason}</dd>
                  </dl>
                </div>
              )}
            </li>
          );
        })}
      </ul>

      <p className="mt-4 flex items-start gap-2 border-t border-ink-700 pt-3 text-[11px] leading-relaxed text-muted-dim">
        <Shield size={12} className="mt-0.5 shrink-0" />
        Policy is deterministic and takes no model or language input. The planner may propose;
        only these rules authorize.
      </p>
    </Card>
  );
}