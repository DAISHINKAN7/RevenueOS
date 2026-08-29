"use client";

import { useEffect, useState } from "react";
import { FlaskConical } from "lucide-react";
import { PageHeader } from "@/components/ui/shell";
import { Card, EmptyState, ErrorState, Metric, Skeleton, Tip } from "@/components/ui/primitives";
import { api } from "@/lib/api";
import { formatINR, formatNumber, formatPercent, titleCase } from "@/lib/format";
import type { EvaluationSummary } from "@/lib/types";

const POLICY_LABELS: Record<string, string> = {
  DO_NOTHING: "Do nothing", FLAT_10_PERCENT: "Flat 10% discount",
  RULES: "Rule baseline", MODEL_CONVERSION_MAX: "Conversion-max",
  REVENUEOS: "RevenueOS", ORACLE_ECONOMIC: "Oracle (upper bound)",
};

function PolicyChart({ policies }: { policies: NonNullable<EvaluationSummary["policies"]> }) {
  const max = Math.max(...policies.map((p) => p.net_contribution_per_opp), 1);
  return (
    <div className="space-y-2.5">
      {policies.map((p) => {
        const isRevenueOS = p.policy === "REVENUEOS";
        const isOracle = p.policy === "ORACLE_ECONOMIC";
        return (
          <div key={p.policy} className="flex items-center gap-3">
            <span className={`w-40 shrink-0 truncate text-[12px] ${
              isRevenueOS ? "font-semibold text-[#e6e9ef]" : "text-muted"}`}>
              {POLICY_LABELS[p.policy] ?? titleCase(p.policy)}
            </span>
            <div className="h-5 flex-1 overflow-hidden rounded bg-ink-850">
              <div className={`h-full rounded ${
                isRevenueOS ? "bg-accent/70" : isOracle ? "bg-ink-500" : "bg-ink-600"}`}
                style={{ width: `${Math.max(1, (p.net_contribution_per_opp / max) * 100)}%` }} />
            </div>
            <span className={`w-24 shrink-0 text-right text-[12px] tabular-nums ${
              isRevenueOS ? "font-semibold text-accent" : "text-muted"}`}>
              {formatINR(p.net_contribution_per_opp)}
            </span>
          </div>
        );
      })}
    </div>
  );
}

export default function EvaluationPage() {
  const [data, setData] = useState<EvaluationSummary | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = () => api.evaluation().then(setData).catch((e) => setError(e.message));
  useEffect(() => { load(); }, []);

  if (error) return <ErrorState message={error} onRetry={load} />;
  if (!data) return <div className="grid gap-4 sm:grid-cols-4">
    {[0,1,2,3].map((i) => <Skeleton key={i} className="h-28" />)}</div>;

  if (!data.available)
    return (
      <>
        <PageHeader title="Evaluation" />
        <div className="surface">
          <EmptyState icon={<FlaskConical size={20} />}
            title="No evaluation artifacts found"
            hint="Run `make model-gate` on the backend to generate held-out evaluation results." />
        </div>
      </>
    );

  const h = data.headline ?? {};
  const revenueos = data.policies?.find((p) => p.policy === "REVENUEOS");
  const flat = data.policies?.find((p) => p.policy === "FLAT_10_PERCENT");
  const nothing = data.policies?.find((p) => p.policy === "DO_NOTHING");

  return (
    <>
      <PageHeader title="Evaluation"
        subtitle="RevenueOS is evaluated on economic policy value, not conversion accuracy alone." />

      <div className="mb-2 flex items-center gap-2">
        <span className="label">Synthetic held-out evaluation</span>
        <Tip text="Frozen research results on the chronological held-out test set. Never mixed with live Test Mode outcomes." />
      </div>

      <div className="mb-5 grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <Card><Metric label="vs flat 10% discount"
          value={revenueos && flat
            ? formatINR(revenueos.net_contribution_per_opp - flat.net_contribution_per_opp)
            : h["RevenueOS vs Flat 10%"] ?? "—"}
          tone="pos" hint="per opportunity" /></Card>
        <Card><Metric label="vs doing nothing"
          value={revenueos && nothing
            ? formatINR(revenueos.net_contribution_per_opp - nothing.net_contribution_per_opp)
            : h["RevenueOS vs DO_NOTHING"] ?? "—"}
          tone="pos" hint="per opportunity" /></Card>
        <Card><Metric label="Conversion / economics divergence"
          value={formatPercent(data.divergence)} tone="accent"
          hint="decisions that change objective" /></Card>
        <Card><Metric label="Intelligent restraint"
          value={formatPercent(data.do_nothing_rate)}
          hint="do-nothing selection rate" /></Card>
      </div>

      <div className="grid gap-4 lg:grid-cols-3">
        <Card title="Net contribution per opportunity"
          subtitle="Policy comparison on the held-out test set" className="lg:col-span-2">
          {data.policies ? <PolicyChart policies={data.policies} /> :
            <p className="text-[12px] text-muted-dim">Policy comparison unavailable.</p>}
          {revenueos && flat && flat.conversion > revenueos.conversion && (
            <p className="mt-4 rounded-lg border border-ink-700 bg-ink-850 p-3 text-[12px] leading-relaxed text-muted">
              Flat 10% converts at {formatPercent(flat.conversion)} versus RevenueOS at{" "}
              {formatPercent(revenueos.conversion)} — higher conversion, yet{" "}
              <strong className="text-neg">
                {formatINR(revenueos.net_contribution_per_opp - flat.net_contribution_per_opp)}
              </strong>{" "}less contribution per opportunity. Conversion is not the objective.
            </p>
          )}
        </Card>

        <div className="space-y-4">
          <Card title="Off-policy evidence"
            subtitle="Estimated from logged data only">
            <dl className="space-y-3 text-[12px]">
              {[["Oracle policy value", h["RevenueOS Oracle Value / Opportunity"]],
                ["Doubly robust estimate", h["RevenueOS DR Value / Opportunity"]],
                ["DR vs oracle error", h["DR vs Oracle Relative Error"]],
                ["Oracle value captured", h["Oracle Incremental Value Captured"]]].map(([k, v]) => (
                <div key={k as string} className="flex items-baseline justify-between gap-3">
                  <dt className="text-muted-dim">{k}</dt>
                  <dd className="tabular-nums font-medium text-[#e6e9ef]">{v ?? "—"}</dd>
                </div>
              ))}
            </dl>
            <p className="mt-3.5 border-t border-ink-700 pt-3 text-[11px] leading-relaxed text-muted-dim">
              The doubly robust estimate uses only logged actions, outcomes and action propensities.
              Synthetic oracle evaluation is reported separately and is not a causal claim.
            </p>
          </Card>

          <Card title="Model quality" subtitle="Calibration matters more than AUC here">
            <dl className="grid grid-cols-2 gap-x-4 gap-y-2.5 text-[12px]">
              {data.model && Object.entries({
                "ROC-AUC": data.model.roc_auc, "PR-AUC": data.model.pr_auc,
                "Brier": data.model.brier, "ECE": data.model.ece,
              }).map(([k, v]) => (
                <div key={k}>
                  <dt className="text-muted-dim">{k}</dt>
                  <dd className="mt-0.5 tabular-nums font-medium text-[#e6e9ef]">{v?.toFixed(4)}</dd>
                </div>
              ))}
            </dl>
            <p className="mt-3 border-t border-ink-700 pt-3 text-[11px] leading-relaxed text-muted-dim">
              Decisions are expected-value comparisons, so probability magnitude matters more than
              ranking. Modest AUC is expected: the environment carries deliberate noise.
            </p>
          </Card>
        </div>
      </div>

      {data.data && (
        <Card className="mt-4" title="Data integrity">
          <dl className="grid grid-cols-2 gap-4 text-[12px] sm:grid-cols-5">
            {[["Train", formatNumber(data.data.train_rows)],
              ["Validation", formatNumber(data.data.validation_rows)],
              ["Test", formatNumber(data.data.test_rows)],
              ["Simulator", data.data.simulator_version],
              ["Oracle access", titleCase(data.data.oracle_access_policy)]].map(([k, v]) => (
              <div key={k}>
                <dt className="label">{k}</dt>
                <dd className="mt-1 tabular-nums text-[13px] text-[#e6e9ef]">{v}</dd>
              </div>
            ))}
          </dl>
          <p className="mt-3.5 border-t border-ink-700 pt-3 text-[11px] leading-relaxed text-muted-dim">
            Chronological split. Calibration fitted on a held-out slice of validation, never on test.
            Counterfactual oracle data is quarantined and used only after model freeze.
          </p>
        </Card>
      )}
    </>
  );
}