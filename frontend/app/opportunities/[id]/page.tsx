"use client";

import { useCallback, useEffect, useState } from "react";
import { use } from "react";
import Link from "next/link";
import { ArrowLeft, Check, ShieldAlert, X } from "lucide-react";
import { AuditTimeline } from "@/components/audit/timeline";
import { CandidateTable, ConversionVsEconomics, WhyNot } from "@/components/decision/candidates";
import { LiveRun } from "@/components/decision/live-run";
import { SelectedAction } from "@/components/decision/selected";
import { ExecutionPanel, RetryComparison, WorkflowProgress } from "@/components/payments/execution";
import { PolicyPanel } from "@/components/policy/guardrails";
import {
  Button, Card, ErrorState, Mono, Skeleton, StateBadge, TestModeBadge,
} from "@/components/ui/primitives";
import { api, ApiError } from "@/lib/api";
import { formatINR, formatAge, titleCase } from "@/lib/format";
import type { OpportunityDetail } from "@/lib/types";

/** Polling is limited to states where an external party may change things. */
const POLL_STATES = new Set(["AWAITING_PAYMENT", "EXECUTING", "EXECUTION_PENDING"]);

export default function OpportunityPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const [detail, setDetail] = useState<OpportunityDetail | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      setDetail(await api.opportunity(id));
      setError(null);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Failed to load opportunity");
    }
  }, [id]);

  useEffect(() => { load(); }, [load]);

  useEffect(() => {
    if (!detail || !POLL_STATES.has(detail.opportunity.state)) return;
    const t = setInterval(load, 3000);
    return () => clearInterval(t);
  }, [detail, load]);

  async function act(kind: "analyze" | "approve" | "reject") {
    setBusy(kind);
    try {
      if (kind === "analyze") await api.analyze(id);
      if (kind === "approve") await api.approve(id);
      if (kind === "reject") await api.reject(id);
      await load();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Action failed");
    } finally { setBusy(null); }
  }

  if (error && !detail) return <ErrorState message={error} onRetry={load} />;
  if (!detail)
    return (
      <div className="space-y-4">
        <Skeleton className="h-24 w-full" />
        <div className="grid gap-4 lg:grid-cols-3">
          <Skeleton className="h-72 lg:col-span-2" /><Skeleton className="h-72" />
        </div>
      </div>
    );

  const { opportunity: o, candidate_actions, policy_decision, checkout_summary,
          payment_summary, customer_summary, execution, audit_timeline } = detail;
  const needsApproval = o.state === "AWAITING_APPROVAL";
  const canAnalyze = ["DETECTED", "PAYMENT_FAILED_RECOVERABLE", "NOT_RECOVERED",
                      "EXECUTION_FAILED"].includes(o.state);

  return (
    <div className="space-y-4">
      <Link href="/opportunities"
        className="inline-flex items-center gap-1.5 text-[12px] text-muted-dim hover:text-muted">
        <ArrowLeft size={13} /> Opportunities
      </Link>

      {/* ---- header ---- */}
      <div className="surface p-5">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <div className="flex flex-wrap items-center gap-2.5">
              <Mono copyable>{o.opportunity_id}</Mono>
              <StateBadge state={o.state} size="md" />
              {o.execution_mode === "RAZORPAY_TEST" && <TestModeBadge />}
            </div>
            <div className="mt-3 flex items-baseline gap-3">
              <span className="text-metric-lg font-semibold tabular-nums text-[#e6e9ef]">
                {formatINR(o.revenue_at_risk)}
              </span>
              <span className="text-[12px] text-muted">revenue at risk</span>
            </div>
            <div className="mt-1 text-[12px] text-muted-dim">
              {titleCase(o.type)} · attempt {o.attempt} · detected {formatAge(o.detected_at)} ago
              {payment_summary.failure_reason &&
                payment_summary.failure_reason !== "NO_PAYMENT_FAILURE" && (
                <> · <span className="text-neg">{titleCase(payment_summary.failure_reason)}</span></>
              )}
            </div>
          </div>

          <div className="flex flex-col items-end gap-2">
            {canAnalyze && (
              <Button variant="primary" onClick={() => act("analyze")} loading={busy === "analyze"}>
                Run analysis
              </Button>
            )}
            <dl className="grid grid-cols-2 gap-x-6 gap-y-1 text-right text-[12px]">
              <dt className="text-muted-dim">Contribution at risk</dt>
              <dd className="tabular-nums text-[#e6e9ef]">
                {formatINR(o.contribution_margin_at_risk)}</dd>
              <dt className="text-muted-dim">Customer segment</dt>
              <dd className="text-[#e6e9ef]">
                {String(customer_summary.segment ?? "—")}</dd>
              <dt className="text-muted-dim">Shipping fee</dt>
              <dd className="tabular-nums text-[#e6e9ef]">
                {formatINR(checkout_summary.shipping_fee_charged ?? 0)}</dd>
            </dl>
          </div>
        </div>
        <div className="mt-4 border-t border-ink-700 pt-3.5">
          <WorkflowProgress state={o.state} attempt={o.attempt} />
        </div>
      </div>

      {/* ---- approval gate ---- */}
      {needsApproval && (
        <Card className="border-warn/30">
          <div className="flex flex-wrap items-start justify-between gap-5">
            <div className="flex items-start gap-3">
              <ShieldAlert size={18} className="mt-0.5 shrink-0 text-warn" />
              <div>
                <h3 className="text-[15px] font-semibold text-[#e6e9ef]">Human approval required</h3>
                <p className="mt-1 max-w-xl text-[12px] leading-relaxed text-muted">
                  {policy_decision?.reason_code === "RULE_HIGH_VALUE_REQUIRES_APPROVAL"
                    ? `Order value of ${formatINR(o.revenue_at_risk)} exceeds the merchant high-value approval threshold.`
                    : `Policy returned ${policy_decision?.reason_code ?? "REQUIRE_APPROVAL"}.`}
                  {" "}No payment execution can occur before approval.
                </p>
                <p className="mt-2 text-[12px] text-muted-dim">
                  Proposed: <span className="text-[#e6e9ef]">{titleCase(detail.selected_action ?? "—")}</span>
                  {" "}· max downside{" "}
                  {formatINR(policy_decision?.maximum_authorized_downside ?? 0)}
                </p>
              </div>
            </div>
            <div className="flex gap-2">
              <Button variant="primary" onClick={() => act("approve")} loading={busy === "approve"}>
                <Check size={13} /> Approve
              </Button>
              <Button variant="danger" onClick={() => act("reject")} loading={busy === "reject"}>
                <X size={13} /> Reject
              </Button>
            </div>
          </div>
        </Card>
      )}

      {/* ---- decision intelligence | policy ---- */}
      <div className="grid gap-4 lg:grid-cols-3">
        <div className="space-y-4 lg:col-span-2">
          <LiveRun opportunityId={o.opportunity_id} state={o.state} onComplete={load} />
          <SelectedAction selected={detail.selected_action}
            candidates={candidate_actions} policy={policy_decision} />
          <ConversionVsEconomics candidates={candidate_actions} selected={detail.selected_action} />
          <CandidateTable candidates={candidate_actions} selected={detail.selected_action}
            policyDecision={policy_decision?.decision ?? null} />
          <WhyNot candidates={candidate_actions} selected={detail.selected_action} />
        </div>
        <div className="space-y-4">
          <PolicyPanel policy={policy_decision} />
        </div>
      </div>

      <RetryComparison executions={execution}
        failureReason={payment_summary.failure_reason} />
      <ExecutionPanel detail={detail} onRefresh={load} />
      <AuditTimeline events={audit_timeline} />
    </div>
  );
}