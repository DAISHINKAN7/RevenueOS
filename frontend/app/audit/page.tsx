"use client";

import { useEffect, useState } from "react";
import { Search } from "lucide-react";
import { AuditTimeline } from "@/components/audit/timeline";
import { PageHeader } from "@/components/ui/shell";
import { Card, EmptyState, ErrorState, Skeleton, StateBadge } from "@/components/ui/primitives";
import { api } from "@/lib/api";
import { formatINR } from "@/lib/format";
import type { AuditEvent, OpportunityListItem } from "@/lib/types";

export default function AuditPage() {
  const [rows, setRows] = useState<OpportunityListItem[] | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [events, setEvents] = useState<AuditEvent[] | null>(null);
  const [query, setQuery] = useState("");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.opportunities()
      .then((r) => { setRows(r); if (r.length && !selected) setSelected(r[0].opportunity_id); })
      .catch((e) => setError(e.message));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (!selected) return;
    setEvents(null);
    api.opportunity(selected)
      .then((d) => setEvents(d.audit_timeline))
      .catch((e) => setError(e.message));
  }, [selected]);

  if (error) return <ErrorState message={error} />;

  const visible = (rows ?? []).filter((r) =>
    !query || r.opportunity_id.toLowerCase().includes(query.toLowerCase()));

  return (
    <>
      <PageHeader title="Audit"
        subtitle="Every material transition is recorded append-only, with the model and policy version in force at the time." />

      <div className="grid gap-4 lg:grid-cols-4">
        <div className="lg:col-span-1">
          <Card title="Opportunities">
            <div className="relative mb-3">
              <Search size={13} className="absolute left-2.5 top-1/2 -translate-y-1/2 text-muted-dim" />
              <input value={query} onChange={(e) => setQuery(e.target.value)}
                placeholder="Search by ID"
                aria-label="Search opportunities by ID"
                className="w-full rounded-md border border-ink-700 bg-ink-850 py-1.5 pl-8 pr-2.5
                  text-[12px] text-[#e6e9ef] placeholder:text-muted-dim focus:border-accent/50" />
            </div>
            {!rows ? (
              <div className="space-y-2">{[0,1,2].map((i) => <Skeleton key={i} className="h-10" />)}</div>
            ) : visible.length === 0 ? (
              <EmptyState title="No matching opportunities" />
            ) : (
              <div className="-mx-5 max-h-[28rem] overflow-auto">
                {visible.map((r) => (
                  <button key={r.opportunity_id} onClick={() => setSelected(r.opportunity_id)}
                    className={`block w-full border-b border-ink-800 px-5 py-2.5 text-left last:border-0
                      hover:bg-ink-850 ${selected === r.opportunity_id ? "bg-ink-850" : ""}`}>
                    <div className="mono truncate text-accent">{r.opportunity_id}</div>
                    <div className="mt-1 flex items-center justify-between gap-2">
                      <StateBadge state={r.state} />
                      <span className="tabular-nums text-[11px] text-muted-dim">
                        {formatINR(r.revenue_at_risk, true)}
                      </span>
                    </div>
                  </button>
                ))}
              </div>
            )}
          </Card>
        </div>

        <div className="lg:col-span-3">
          {!selected ? (
            <div className="surface"><EmptyState title="Select an opportunity" /></div>
          ) : !events ? (
            <Skeleton className="h-96 w-full" />
          ) : (
            <AuditTimeline events={events} />
          )}
        </div>
      </div>
    </>
  );
}