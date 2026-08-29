"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { ClipboardList } from "lucide-react";
import { PageHeader } from "@/components/ui/shell";
import {
  EmptyState, ErrorState, Mono, Skeleton, StateBadge,
} from "@/components/ui/primitives";
import { api } from "@/lib/api";
import { formatAge, formatINR, titleCase } from "@/lib/format";
import type { OpportunityListItem } from "@/lib/types";

const FILTERS = [
  { key: "all", label: "All" },
  { key: "DETECTED", label: "At risk" },
  { key: "AWAITING_APPROVAL", label: "Awaiting approval" },
  { key: "AWAITING_PAYMENT", label: "Awaiting payment" },
  { key: "PAYMENT_FAILED_RECOVERABLE", label: "Recoverable failure" },
  { key: "RECOVERED", label: "Recovered" },
];

export default function OpportunitiesPage() {
  const [rows, setRows] = useState<OpportunityListItem[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [filter, setFilter] = useState("all");

  const load = () =>
    api.opportunities()
      .then((r) => { setRows(r); setError(null); })
      .catch((e) => setError(e.message));

  useEffect(() => { load(); }, []);

  const visible = useMemo(
    () => (rows ?? []).filter((r) => filter === "all" || r.state === filter),
    [rows, filter]);

  if (error) return <ErrorState message={error} onRetry={load} />;

  return (
    <>
      <PageHeader
        title="Opportunities"
        subtitle="Compare recovery probability, financial impact and policy constraints before any action is executed."
      />

      <div className="mb-4 flex flex-wrap gap-1.5">
        {FILTERS.map((f) => {
          const count = f.key === "all"
            ? rows?.length ?? 0
            : (rows ?? []).filter((r) => r.state === f.key).length;
          return (
            <button key={f.key} onClick={() => setFilter(f.key)}
              className={`rounded-md border px-2.5 py-1 text-[12px] transition-colors ${
                filter === f.key
                  ? "border-ink-500 bg-ink-800 text-[#e6e9ef]"
                  : "border-ink-700 text-muted hover:bg-ink-850"}`}>
              {f.label}
              <span className="ml-1.5 tabular-nums text-muted-dim">{count}</span>
            </button>
          );
        })}
      </div>

      {!rows ? (
        <div className="space-y-2">{[0, 1, 2, 3].map((i) => <Skeleton key={i} className="h-12 w-full" />)}</div>
      ) : visible.length === 0 ? (
        <div className="surface">
          <EmptyState icon={<ClipboardList size={20} />}
            title={filter === "all" ? "No opportunities yet" : `No opportunities are ${FILTERS.find((f) => f.key === filter)?.label.toLowerCase()}.`}
            hint={filter === "all" ? "Seed demo data with `make demo-reset` on the backend." : undefined} />
        </div>
      ) : (
        <div className="surface overflow-hidden">
          <table className="w-full text-[12px]">
            <thead>
              <tr className="border-b border-ink-700 text-left">
                {["Opportunity", "Type", "Revenue at risk", "State", "Selected action",
                  "Attempt", "Mode", "Age"].map((h) => (
                  <th key={h} className="px-4 py-2.5 font-medium text-muted-dim">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {visible.map((r) => (
                <tr key={r.opportunity_id}
                  className="border-b border-ink-800 last:border-0 hover:bg-ink-850">
                  <td className="px-4 py-3">
                    <Link href={`/opportunities/${r.opportunity_id}`}
                      className="mono text-accent hover:underline">{r.opportunity_id}</Link>
                    {r.customer_segment && (
                      <div className="mt-0.5 text-[11px] text-muted-dim">{titleCase(r.customer_segment)}</div>
                    )}
                  </td>
                  <td className="px-4 py-3 text-muted">{titleCase(r.opportunity_type)}</td>
                  <td className="px-4 py-3 tabular-nums font-medium text-[#e6e9ef]">
                    {formatINR(r.revenue_at_risk)}</td>
                  <td className="px-4 py-3"><StateBadge state={r.state} /></td>
                  <td className="px-4 py-3 text-muted">
                    {r.selected_action ? titleCase(r.selected_action) : "—"}</td>
                  <td className="px-4 py-3 tabular-nums text-muted">{r.attempt}</td>
                  <td className="px-4 py-3">
                    <span className={`text-[11px] ${
                      r.execution_mode === "RAZORPAY_TEST" ? "text-warn" : "text-muted-dim"}`}>
                      {r.execution_mode === "RAZORPAY_TEST" ? "Test Mode" : "Simulator"}
                    </span>
                  </td>
                  <td className="px-4 py-3 text-muted-dim">{formatAge(r.detected_at)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </>
  );
}