"use client";

import { useState } from "react";
import {
  Ban, Bot, Check, CreditCard, Radio, Shield, Sparkles, Zap, type LucideIcon,
} from "lucide-react";
import { Card, Mono } from "@/components/ui/primitives";
import { formatClock, formatTimestamp, titleCase } from "@/lib/format";
import type { AuditEvent } from "@/lib/types";

type Category = "Intelligence" | "Policy" | "Execution" | "Payment" | "Agent" | "Audit";

const EVENT_META: Record<string, { icon: LucideIcon; category: Category; tone: string }> = {
  OPPORTUNITY_DETECTED: { icon: Sparkles, category: "Audit", tone: "text-muted" },
  ANALYSIS_STARTED: { icon: Sparkles, category: "Intelligence", tone: "text-accent" },
  ACTION_SCORED: { icon: Sparkles, category: "Intelligence", tone: "text-accent" },
  ACTIONS_RANKED: { icon: Sparkles, category: "Intelligence", tone: "text-accent" },
  ADAPTIVE_ADJUSTMENT_APPLIED: { icon: Sparkles, category: "Intelligence", tone: "text-accent" },
  POLICY_EVALUATED: { icon: Shield, category: "Policy", tone: "text-warn" },
  ACTION_AUTHORIZED: { icon: Shield, category: "Policy", tone: "text-pos" },
  ACTION_REJECTED: { icon: Ban, category: "Policy", tone: "text-neg" },
  APPROVAL_REQUIRED: { icon: Shield, category: "Policy", tone: "text-warn" },
  EXECUTION_CREATED: { icon: Zap, category: "Execution", tone: "text-accent" },
  EXECUTION_ERROR: { icon: Ban, category: "Execution", tone: "text-neg" },
  RAZORPAY_ORDER_CREATED: { icon: CreditCard, category: "Payment", tone: "text-accent" },
  CHECKOUT_STARTED: { icon: CreditCard, category: "Payment", tone: "text-accent" },
  WEBHOOK_RECEIVED: { icon: Radio, category: "Payment", tone: "text-accent" },
  PAYMENT_FAILED: { icon: Ban, category: "Payment", tone: "text-neg" },
  RECOVERY_CONFIRMED: { icon: Check, category: "Payment", tone: "text-pos" },
  AUDIT_CORRECTION: { icon: Shield, category: "Audit", tone: "text-muted" },
  WORKFLOW_STOPPED: { icon: Ban, category: "Audit", tone: "text-muted" },
  AGENT_RUN_STARTED: { icon: Bot, category: "Agent", tone: "text-muted" },
  AGENT_TOOL_BLOCKED: { icon: Ban, category: "Agent", tone: "text-neg" },
  AGENT_STOPPED: { icon: Bot, category: "Agent", tone: "text-muted" },
};

const CATEGORIES: Category[] = ["Intelligence", "Policy", "Execution", "Payment", "Agent", "Audit"];

export function AuditTimeline({ events }: { events: AuditEvent[] }) {
  const [filter, setFilter] = useState<Category | "All">("All");
  const [open, setOpen] = useState<number | null>(null);

  const visible = events.filter((e) => {
    if (filter === "All") return true;
    return (EVENT_META[e.event_type]?.category ?? "Audit") === filter;
  });

  return (
    <Card
      title="Audit timeline"
      subtitle={`${events.length} append-only events`}
      actions={
        <div className="flex flex-wrap gap-1">
          {(["All", ...CATEGORIES] as const).map((c) => (
            <button key={c} onClick={() => setFilter(c)}
              className={`rounded px-2 py-0.5 text-[11px] transition-colors ${
                filter === c ? "bg-ink-700 text-[#e6e9ef]" : "text-muted-dim hover:text-muted"}`}>
              {c}
            </button>
          ))}
        </div>
      }
    >
      {visible.length === 0 ? (
        <p className="py-6 text-center text-[12px] text-muted-dim">No events in this category.</p>
      ) : (
        <ol className="relative">
          <div className="absolute bottom-2 left-[13px] top-2 w-px bg-ink-700" aria-hidden />
          {visible.map((e) => {
            const meta = EVENT_META[e.event_type] ?? { icon: Sparkles, category: "Audit" as Category, tone: "text-muted" };
            const Icon = meta.icon;
            const isOpen = open === e.sequence;
            return (
              <li key={e.sequence} className="relative animate-fade-up pb-1 pl-9">
                <span className={`absolute left-0 top-1 flex h-[27px] w-[27px] items-center justify-center
                  rounded-full border border-ink-600 bg-ink-900 ${meta.tone}`}>
                  <Icon size={12} />
                </span>
                <button onClick={() => setOpen(isOpen ? null : e.sequence)}
                  className="w-full rounded-md px-2 py-1.5 text-left hover:bg-ink-850">
                  <div className="flex items-baseline justify-between gap-3">
                    <span className="text-[12px] font-medium text-[#e6e9ef]">
                      {titleCase(e.event_type)}
                    </span>
                    <span className="mono shrink-0 text-muted-dim">{formatClock(e.timestamp)}</span>
                  </div>
                  <p className="mt-0.5 text-[12px] leading-relaxed text-muted">{e.summary}</p>
                  {e.state_before && e.state_after && (
                    <p className="mono mt-1 text-muted-dim">
                      {e.state_before} → {e.state_after}
                    </p>
                  )}
                </button>
                {isOpen && (
                  <div className="mb-2 ml-2 rounded-lg border border-ink-700 bg-ink-850 p-3 animate-fade-up">
                    <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-[11px]">
                      <dt className="text-muted-dim">Sequence</dt><dd className="mono">{e.sequence}</dd>
                      <dt className="text-muted-dim">Timestamp</dt>
                      <dd className="text-muted">{formatTimestamp(e.timestamp)}</dd>
                      <dt className="text-muted-dim">Actor</dt><dd className="text-muted">{e.actor}</dd>
                      {e.execution_id && (<>
                        <dt className="text-muted-dim">Execution</dt>
                        <dd><Mono copyable>{e.execution_id}</Mono></dd></>)}
                    </dl>
                    {Object.keys(e.payload ?? {}).length > 0 && (
                      <details className="mt-2.5">
                        <summary className="cursor-pointer text-[11px] text-muted-dim hover:text-muted">
                          Developer details
                        </summary>
                        <pre className="mono mt-2 max-h-56 overflow-auto rounded bg-ink-950 p-2.5
                          text-[10px] leading-relaxed text-muted">
{JSON.stringify(e.payload, null, 2)}
                        </pre>
                      </details>
                    )}
                  </div>
                )}
              </li>
            );
          })}
        </ol>
      )}
    </Card>
  );
}