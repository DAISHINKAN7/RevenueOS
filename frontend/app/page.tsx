"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { ArrowRight, TrendingDown } from "lucide-react";
import { LiveFeed } from "@/components/dashboard/live-feed";
import { PageHeader } from "@/components/ui/shell";
import {
  Card, EmptyState, ErrorState, Metric, Skeleton, StateBadge, Tip,
} from "@/components/ui/primitives";
import { api } from "@/lib/api";
import { formatINR, formatPercent, formatNumber, formatAge, titleCase } from "@/lib/format";
import type { DashboardMetrics, EvaluationSummary, OpportunityListItem } from "@/lib/types";

function Funnel({ m }: { m: DashboardMetrics }) {
  const stages = [
    { label: "Revenue at risk", value: m.revenue_at_risk },
    { label: "Contribution at risk", value: m.contribution_margin_at_risk },
    { label: "Recovered GMV", value: m.recovered_gmv },
    { label: "Net contribution recovered", value: m.net_contribution_recovered },
  ];
  const max = Math.max(...stages.map((s) => s.value), 1);
  return (
    <div className="space-y-2.5">
      {stages.map((s) => (
        <div key={s.label}>
          <div className="mb-1 flex items-baseline justify-between">
            <span className="text-[12px] text-muted">{s.label}</span>
            <span className="tabular-nums text-[12px] font-medium text-[#e6e9ef]">
              {formatINR(s.value)}
            </span>
          </div>
          <div className="h-1.5 overflow-hidden rounded-full bg-ink-800">
            <div className="h-full rounded-full bg-accent/50"
              style={{ width: `${Math.max(2, (s.value / max) * 100)}%` }} />
          </div>
        </div>
      ))}
    </div>
  );
}

export default function DashboardPage() {
  const [metrics, setMetrics] = useState<DashboardMetrics | null>(null);
  const [evaluation, setEvaluation] = useState<EvaluationSummary | null>(null);
  const [queue, setQueue] = useState<OpportunityListItem[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = () => {
    Promise.all([api.dashboard(), api.evaluation(), api.opportunities()])
      .then(([m, e, q]) => { setMetrics(m); setEvaluation(e); setQueue(q); setError(null); })
      .catch((e) => setError(e.message));
  };
  useEffect(() => { load(); }, []);

  if (error) return <ErrorState message={error} onRetry={load} />;
  if (!metrics)
    return <div className="grid gap-4 sm:grid-cols-4">
      {[0,1,2,3].map((i) => <Skeleton key={i} className="h-28" />)}</div>;

  const divergence = evaluation?.divergence;

  return (
    <>
      <PageHeader
        title="Overview"
        subtitle="Detect revenue leakage, choose the highest-value intervention, and execute bounded recovery workflows."
      />

      {/* ---- headline metrics: live operational only ---- */}
      <div className="mb-2 flex items-center gap-2">
        <span className="label">Live demo · Razorpay Test Mode</span>
        <Tip text="These are operational outcomes from seeded and Test Mode executions. Research evaluation is reported separately." />
      </div>
      <div className="mb-5 grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <Card><Metric label="Revenue at risk" value={formatINR(metrics.revenue_at_risk, true)}
          hint={`${formatNumber(metrics.opportunities)} opportunities`} /></Card>
        <Card><Metric label="Contribution at risk"
          value={formatINR(metrics.contribution_margin_at_risk, true)} /></Card>
        <Card><Metric label="Recovered GMV" value={formatINR(metrics.recovered_gmv, true)}
          tone={metrics.recovered_gmv > 0 ? "pos" : "neutral"}
          hint={`Recovery rate ${formatPercent(metrics.recovery_rate)}`} /></Card>
        <Card><Metric label="Net contribution recovered"
          value={formatINR(metrics.net_contribution_recovered, true)}
          tone={metrics.net_contribution_recovered > 0 ? "pos" : "neutral"}
          hint={`Incentive spend ${formatINR(metrics.intervention_cost)}`} /></Card>
      </div>

      {/* ---- hero insight ---- */}
      {divergence !== undefined && (
        <Card className="mb-5 border-accent/25">
          <div className="flex flex-wrap items-center gap-6">
            <div className="flex items-center gap-4">
              <TrendingDown size={20} className="text-accent" />
              <div>
                <div className="text-metric-lg font-semibold tabular-nums text-accent">
                  {formatPercent(divergence)}
                </div>
                <div className="label mt-1">of decisions change</div>
              </div>
            </div>
            <p className="max-w-xl flex-1 text-[13px] leading-relaxed text-muted">
              …when optimizing for <strong className="text-[#e6e9ef]">contribution</strong> instead of
              conversion. RevenueOS often chooses a lower-converting action because it preserves more
              merchant margin — a blanket discount can lift conversions while destroying profit.
            </p>
            <Link href="/evaluation"
              className="inline-flex items-center gap-1.5 rounded-md border border-ink-600 px-3 py-1.5
                text-[12px] text-muted hover:bg-ink-800">
              Evidence <ArrowRight size={12} />
            </Link>
          </div>
        </Card>
      )}

      <div className="mb-4"><LiveFeed /></div>

      <div className="grid gap-4 lg:grid-cols-3">
        <Card title="Recovery funnel" subtitle="Live operational values" className="lg:col-span-1">
          <Funnel m={metrics} />
          <dl className="mt-4 grid grid-cols-2 gap-x-4 gap-y-2 border-t border-ink-700 pt-3 text-[12px]">
            <dt className="text-muted-dim">Policy blocks</dt>
            <dd className="tabular-nums text-[#e6e9ef]">{metrics.number_of_policy_blocks}</dd>
            <dt className="text-muted-dim">Approvals required</dt>
            <dd className="tabular-nums text-[#e6e9ef]">{metrics.number_of_approval_cases}</dd>
            <dt className="text-muted-dim">Do-nothing decisions</dt>
            <dd className="tabular-nums text-[#e6e9ef]">{metrics.number_of_do_nothing_decisions}</dd>
          </dl>
        </Card>

        <Card title="Opportunity queue" subtitle="Most recent" className="lg:col-span-2"
          actions={<Link href="/opportunities"
            className="text-[12px] text-accent hover:underline">View all</Link>}>
          {!queue?.length ? (
            <EmptyState title="No opportunities yet"
              hint="Run `make demo-reset` on the backend to seed demo scenarios." />
          ) : (
            <div className="-mx-5 -my-1">
              {queue.slice(0, 6).map((r) => (
                <Link key={r.opportunity_id} href={`/opportunities/${r.opportunity_id}`}
                  className="flex items-center justify-between gap-4 border-b border-ink-800 px-5 py-2.5
                    last:border-0 hover:bg-ink-850">
                  <div className="min-w-0">
                    <div className="mono truncate text-accent">{r.opportunity_id}</div>
                    <div className="mt-0.5 text-[11px] text-muted-dim">
                      {titleCase(r.opportunity_type)} · {formatAge(r.detected_at)} ago
                    </div>
                  </div>
                  <div className="flex shrink-0 items-center gap-4">
                    <span className="tabular-nums text-[13px] font-medium text-[#e6e9ef]">
                      {formatINR(r.revenue_at_risk)}
                    </span>
                    <StateBadge state={r.state} />
                  </div>
                </Link>
              ))}
            </div>
          )}
        </Card>
      </div>
    </>
  );
}