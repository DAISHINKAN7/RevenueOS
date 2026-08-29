"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import { Radio } from "lucide-react";
import { Card, EmptyState } from "@/components/ui/primitives";
import { API_BASE } from "@/lib/api";
import { formatClock, titleCase } from "@/lib/format";

interface FeedEvent {
  audit_id: number; opportunity_id: string; event_type: string;
  summary: string; timestamp: string; state_after: string | null;
}

const TONE: Record<string, string> = {
  RECOVERY_CONFIRMED: "text-pos", PAYMENT_FAILED: "text-neg",
  ACTION_REJECTED: "text-neg", AGENT_TOOL_BLOCKED: "text-warn",
  APPROVAL_REQUIRED: "text-warn", RAZORPAY_ORDER_CREATED: "text-accent",
  WEBHOOK_RECEIVED: "text-accent", ACTION_AUTHORIZED: "text-accent",
};

/**
 * Cross-opportunity activity, polled with a watermark so only new rows are
 * fetched. Arrivals animate in; nothing is fabricated between polls.
 */
export function LiveFeed({ pollMs = 3000 }: { pollMs?: number }) {
  const [events, setEvents] = useState<FeedEvent[]>([]);
  const [live, setLive] = useState(false);
  const watermark = useRef(0);

  useEffect(() => {
    let alive = true;
    const tick = async () => {
      try {
        const res = await fetch(
          `${API_BASE}/api/events/recent?limit=25${watermark.current ? `&since=${watermark.current}` : ""}`,
          { cache: "no-store" });
        if (!res.ok) throw new Error();
        const data = await res.json();
        if (!alive) return;
        setLive(true);
        if (data.events?.length) {
          watermark.current = Math.max(watermark.current, data.watermark);
          setEvents((prev) => [...data.events, ...prev].slice(0, 25));
        }
      } catch { if (alive) setLive(false); }
    };
    tick();
    const t = setInterval(tick, pollMs);
    return () => { alive = false; clearInterval(t); };
  }, [pollMs]);

  return (
    <Card
      title="Live activity"
      subtitle="Recovery events across all opportunities"
      actions={
        <span className="inline-flex items-center gap-1.5 text-[11px] text-muted-dim">
          <span className={`h-1.5 w-1.5 rounded-full ${
            live ? "animate-pulse-soft bg-pos" : "bg-ink-500"}`} />
          {live ? "Live" : "Offline"}
        </span>
      }
    >
      {events.length === 0 ? (
        <EmptyState icon={<Radio size={18} />} title="Waiting for activity"
          hint="Run an analysis or execute a recovery to see events stream in." />
      ) : (
        <ol className="-my-1 max-h-[22rem] overflow-auto">
          {events.map((e) => (
            <li key={e.audit_id}
              className="animate-fade-up border-b border-ink-800 py-2 last:border-0">
              <div className="flex items-baseline justify-between gap-3">
                <span className={`text-[12px] font-medium ${TONE[e.event_type] ?? "text-muted"}`}>
                  {titleCase(e.event_type)}
                </span>
                <span className="mono shrink-0 text-muted-dim">{formatClock(e.timestamp)}</span>
              </div>
              <p className="mt-0.5 truncate text-[11px] text-muted">{e.summary}</p>
              <Link href={`/opportunities/${e.opportunity_id}`}
                className="mono mt-0.5 inline-block text-[10px] text-accent hover:underline">
                {e.opportunity_id}
              </Link>
            </li>
          ))}
        </ol>
      )}
    </Card>
  );
}