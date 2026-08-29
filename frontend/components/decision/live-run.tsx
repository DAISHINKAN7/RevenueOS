"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { Activity, Check, Play, Shield, Sparkles, Zap } from "lucide-react";
import { Button, Card, DeltaEV, PolicyBadge } from "@/components/ui/primitives";
import { API_BASE } from "@/lib/api";
import { formatINR, formatProbability, titleCase } from "@/lib/format";

const TOKEN = process.env.NEXT_PUBLIC_ADMIN_TOKEN ?? "";

type Stage = "idle" | "running" | "done" | "failed";

interface ScoredAction { action: string; probability: number | null; valid: boolean }
interface RankedAction {
  action: string; probability: number | null;
  incremental_expected_value: number; incentive_cost_if_recovered: number;
}
interface Step { id: number; kind: string; label: string; detail?: string }

/**
 * Streams a live analysis over server-sent events.
 *
 * This is the same `analyze()` the REST route runs — the workflow reports stage
 * completions through a callback, so what appears here is the decision actually
 * being made, not a replay of a stored result.
 */
export function LiveRun({ opportunityId, state, onComplete }: {
  opportunityId: string; state: string; onComplete: () => void;
}) {
  const [stage, setStage] = useState<Stage>("idle");
  const [steps, setSteps] = useState<Step[]>([]);
  const [scored, setScored] = useState<ScoredAction[]>([]);
  const [ranked, setRanked] = useState<RankedAction[]>([]);
  const [policy, setPolicy] = useState<{ decision: string; reason_code: string;
    maximum_authorized_downside: number } | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const esRef = useRef<EventSource | null>(null);
  const seq = useRef(0);

  const canRun = ["DETECTED", "PAYMENT_FAILED_RECOVERABLE", "NOT_RECOVERED",
                  "EXECUTION_FAILED"].includes(state);

  useEffect(() => () => esRef.current?.close(), []);

  const push = (kind: string, label: string, detail?: string) =>
    setSteps((s) => [...s, { id: seq.current++, kind, label, detail }]);

  const start = useCallback(() => {
    setStage("running"); setSteps([]); setScored([]); setRanked([]);
    setPolicy(null); setSelected(null); setError(null);
    seq.current = 0;

    const url = `${API_BASE}/api/opportunities/${opportunityId}/analyze/stream` +
      `?token=${encodeURIComponent(TOKEN)}`;
    const es = new EventSource(url);
    esRef.current = es;

    es.addEventListener("analysis_started", (e) => {
      const d = JSON.parse((e as MessageEvent).data);
      push("start", `Analysis started — attempt ${d.attempt}`,
        `${d.eligible_actions.length} eligible actions · ${formatINR(d.revenue_at_risk)} at risk`);
    });

    es.addEventListener("action_scored", (e) => {
      const d = JSON.parse((e as MessageEvent).data) as ScoredAction;
      setScored((s) => [...s, d]);
      push("score", `Scored ${titleCase(d.action)}`,
        d.valid && d.probability !== null
          ? `recovery probability ${formatProbability(d.probability)}`
          : "prediction invalid");
    });

    es.addEventListener("actions_ranked", (e) => {
      const d = JSON.parse((e as MessageEvent).data);
      setRanked(d.ranking as RankedAction[]);
      push("rank", "Ranked by incremental expected value",
        `${d.ranking.length} candidates compared against doing nothing`);
    });

    es.addEventListener("policy_evaluated", (e) => {
      const d = JSON.parse((e as MessageEvent).data);
      setPolicy(d);
      const failed = (d.rules ?? []).filter((r: { passed: boolean }) => !r.passed).length;
      push("policy", `Policy ${d.decision}`,
        failed === 0
          ? `all ${d.rules.length} rules passed · downside capped at ${formatINR(d.maximum_authorized_downside)}`
          : `${failed} rule(s) triggered — ${d.reason_code}`);
    });

    es.addEventListener("decision_complete", (e) => {
      const d = JSON.parse((e as MessageEvent).data);
      setSelected(d.selected_action);
      push("done", `Selected ${titleCase(d.selected_action)}`, `state → ${d.state}`);
    });

    es.addEventListener("done", () => {
      es.close(); setStage("done"); onComplete();
    });
    es.addEventListener("failed", (e) => {
      const d = JSON.parse((e as MessageEvent).data);
      setError(d.message); es.close(); setStage("failed");
    });
    es.onerror = () => {
      es.close();
      setStage((s) => (s === "running" ? "failed" : s));
      setError((prev) => prev ?? "Stream interrupted");
    };
  }, [opportunityId, onComplete]);

  if (!canRun && stage === "idle") return null;

  const ICONS: Record<string, typeof Sparkles> = {
    start: Activity, score: Sparkles, rank: Zap, policy: Shield, done: Check,
  };

  return (
    <Card
      title="Live decision"
      subtitle={stage === "running"
        ? "Streaming each stage as it completes"
        : "Watch the model score, the economics rank and the policy gate decide"}
      actions={
        canRun ? (
          <Button variant="primary" onClick={start} loading={stage === "running"}>
            <Play size={12} /> {stage === "idle" ? "Run live analysis" : "Run again"}
          </Button>
        ) : undefined
      }
    >
      {stage === "idle" ? (
        <p className="text-[12px] leading-relaxed text-muted-dim">
          Nothing has been decided for this opportunity yet. Running the analysis streams every
          stage — candidate scoring, economic ranking and the policy gate — as it happens.
        </p>
      ) : (
        <div className="grid gap-5 lg:grid-cols-2">
          {/* ---- event stream ---- */}
          <ol className="relative space-y-0.5">
            <div className="absolute bottom-2 left-[11px] top-2 w-px bg-ink-700" aria-hidden />
            {steps.map((s) => {
              const Icon = ICONS[s.kind] ?? Sparkles;
              return (
                <li key={s.id} className="relative animate-fade-up pl-8">
                  <span className={`absolute left-0 top-0.5 flex h-[23px] w-[23px] items-center
                    justify-center rounded-full border border-ink-600 bg-ink-900 ${
                      s.kind === "done" ? "text-pos"
                      : s.kind === "policy" ? "text-warn" : "text-accent"}`}>
                    <Icon size={11} />
                  </span>
                  <div className="pb-2">
                    <div className="text-[12px] font-medium text-[#e6e9ef]">{s.label}</div>
                    {s.detail && <div className="mt-0.5 text-[11px] text-muted">{s.detail}</div>}
                  </div>
                </li>
              );
            })}
            {stage === "running" && (
              <li className="relative pl-8">
                <span className="absolute left-0 top-0.5 flex h-[23px] w-[23px] items-center
                  justify-center rounded-full border border-ink-600 bg-ink-900">
                  <span className="h-2.5 w-2.5 animate-spin rounded-full border border-accent
                    border-t-transparent" />
                </span>
                <span className="text-[12px] text-muted-dim">Working…</span>
              </li>
            )}
            {error && <li className="pl-8 text-[12px] text-neg">{error}</li>}
          </ol>

          {/* ---- live results ---- */}
          <div className="space-y-3">
            {scored.length > 0 && ranked.length === 0 && (
              <div>
                <div className="label mb-2">Scoring candidates</div>
                <div className="space-y-1">
                  {scored.map((a) => (
                    <div key={a.action}
                      className="flex animate-fade-up items-center justify-between gap-3 rounded-md
                        border border-ink-700 bg-ink-850 px-2.5 py-1.5">
                      <span className="text-[12px] text-muted">{titleCase(a.action)}</span>
                      <span className="tabular-nums text-[12px] text-[#e6e9ef]">
                        {formatProbability(a.probability)}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {ranked.length > 0 && (
              <div className="animate-fade-up">
                <div className="label mb-2">Ranked by incremental value</div>
                <div className="space-y-1">
                  {ranked.slice(0, 6).map((a) => (
                    <div key={a.action}
                      className={`flex items-center justify-between gap-3 rounded-md border px-2.5 py-1.5 ${
                        a.action === selected
                          ? "border-accent/40 bg-accent-soft" : "border-ink-700 bg-ink-850"}`}>
                      <span className={`text-[12px] ${
                        a.action === selected ? "font-semibold text-[#e6e9ef]" : "text-muted"}`}>
                        {titleCase(a.action)}
                      </span>
                      <span className="flex items-center gap-3">
                        <span className="tabular-nums text-[11px] text-muted-dim">
                          {formatProbability(a.probability)}
                        </span>
                        <DeltaEV value={a.incremental_expected_value} />
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {policy && (
              <div className="animate-fade-up rounded-lg border border-ink-700 bg-ink-850 p-3">
                <div className="flex items-center justify-between">
                  <span className="label">Policy gate</span>
                  <PolicyBadge status={policy.decision} />
                </div>
                <div className="mt-2 text-[12px] text-muted">
                  Maximum authorized downside{" "}
                  <span className="tabular-nums font-medium text-[#e6e9ef]">
                    {formatINR(policy.maximum_authorized_downside)}
                  </span>
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </Card>
  );
}