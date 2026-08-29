"use client";

import { useEffect, useState } from "react";
import { PageHeader } from "@/components/ui/shell";
import { Card, ErrorState, Skeleton } from "@/components/ui/primitives";
import { api } from "@/lib/api";
import { titleCase } from "@/lib/format";
import type { MerchantPolicyConfig } from "@/lib/types";

const UNITS: Record<string, (v: string | number | boolean) => string> = {
  max_autonomous_discount_percent: (v) => `${v}%`,
  minimum_remaining_contribution_margin_percent: (v) => `${v}%`,
  minimum_recovery_probability: (v) => `${(Number(v) * 100).toFixed(0)}%`,
  max_autonomous_discount_amount: (v) => `₹${v}`,
  max_free_shipping_cost: (v) => `₹${v}`,
  minimum_incremental_expected_value: (v) => `₹${v}`,
  minimum_decision_margin: (v) => `₹${v}`,
  human_approval_required_above_discount_amount: (v) => `₹${v}`,
  high_value_order_threshold: (v) => `₹${Number(v).toLocaleString("en-IN")}`,
  opportunity_ttl_minutes: (v) => `${v} min`,
};

export default function SettingsPage() {
  const [policy, setPolicy] = useState<MerchantPolicyConfig | null>(null);
  const [versions, setVersions] = useState<Record<string, string> | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = () =>
    Promise.all([api.policy(), api.version()])
      .then(([p, v]) => { setPolicy(p); setVersions(v); })
      .catch((e) => setError(e.message));
  useEffect(() => { load(); }, []);

  if (error) return <ErrorState message={error} onRetry={load} />;
  if (!policy || !versions)
    return <div className="grid gap-4 lg:grid-cols-2"><Skeleton className="h-80" /><Skeleton className="h-80" /></div>;

  return (
    <>
      <PageHeader title="Settings"
        subtitle="Merchant policy is versioned and read-only here. Every authorization decision cites the version in force." />
      <div className="grid gap-4 lg:grid-cols-2">
        <Card title="Merchant policy" subtitle={String(policy.policy_version)}>
          <dl className="space-y-2.5 text-[12px]">
            {Object.entries(policy).filter(([k]) => k !== "policy_version").map(([k, v]) => (
              <div key={k} className="flex items-baseline justify-between gap-4 border-b border-ink-800
                pb-2 last:border-0">
                <dt className="text-muted">{titleCase(k)}</dt>
                <dd className="shrink-0 tabular-nums font-medium text-[#e6e9ef]">
                  {typeof v === "boolean" ? (v ? "Required" : "Not required")
                    : UNITS[k]?.(v) ?? String(v)}
                </dd>
              </div>
            ))}
          </dl>
        </Card>

        <Card title="System versions" subtitle="For auditability">
          <dl className="space-y-2.5 text-[12px]">
            {Object.entries(versions).map(([k, v]) => (
              <div key={k} className="flex items-baseline justify-between gap-4 border-b border-ink-800
                pb-2 last:border-0">
                <dt className="text-muted">{titleCase(k)}</dt>
                <dd className="mono shrink-0 text-[#e6e9ef]">{v}</dd>
              </div>
            ))}
          </dl>
        </Card>
      </div>
    </>
  );
}