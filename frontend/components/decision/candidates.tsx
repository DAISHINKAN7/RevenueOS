"use client";

import { useState } from "react";
import { ChevronDown, TrendingUp, Wallet } from "lucide-react";
import { Card, DeltaEV, Probability, Tip } from "@/components/ui/primitives";
import { formatDeltaEV, formatINR, formatProbability, titleCase } from "@/lib/format";
import type { CandidateAction } from "@/lib/types";

const DEV_TIP =
  "ΔEV is the expected value of an action minus the expected value of doing nothing. " +
  "It is what the system ranks on — not raw conversion probability.";

/**
 * The signature visual. Conversion-max and economics-max are shown side by
 * side because the whole product argument is that they are not the same action.
 */
export function ConversionVsEconomics({ candidates, selected }: {
  candidates: CandidateAction[]; selected: string | null;
}) {
  const scored = candidates.filter((c) => c.probability !== null);
  if (scored.length < 2) return null;

  const byProb = [...scored].sort((a, b) => b.probability - a.probability)[0];
  const byEV = [...scored].sort(
    (a, b) => b.incremental_expected_value - a.incremental_expected_value)[0];
  const diverges = byProb.action !== byEV.action;
  const gap = byEV.incremental_expected_value - byProb.incremental_expected_value;

  return (
    <Card
      title="Conversion versus economics"
      subtitle={diverges
        ? "The highest-converting action is not the most profitable action."
        : "For this opportunity both objectives agree."}
    >
      <div className="grid gap-4 sm:grid-cols-2">
        <div className="rounded-lg border border-ink-700 bg-ink-850 p-4">
          <div className="flex items-center gap-2 label"><TrendingUp size={12} /> Highest conversion</div>
          <div className="mt-2 text-[15px] font-semibold text-[#e6e9ef]">{titleCase(byProb.action)}</div>
          <div className="mt-2 flex items-baseline gap-3">
            <span className="text-metric font-semibold tabular-nums text-[#e6e9ef]">
              {formatProbability(byProb.probability)}
            </span>
            <span className="text-[11px] text-muted-dim">predicted recovery</span>
          </div>
          <div className="mt-3 border-t border-ink-700 pt-3">
            <span className="text-[12px] text-muted">Economic value </span>
            <DeltaEV value={byProb.incremental_expected_value} />
          </div>
        </div>

        <div className={`rounded-lg border p-4 ${
          diverges ? "border-accent/40 bg-accent-soft" : "border-ink-700 bg-ink-850"}`}>
          <div className="flex items-center gap-2 label"><Wallet size={12} /> Best economic outcome</div>
          <div className="mt-2 text-[15px] font-semibold text-[#e6e9ef]">{titleCase(byEV.action)}</div>
          <div className="mt-2 flex items-baseline gap-3">
            <DeltaEV value={byEV.incremental_expected_value} size="lg" />
          </div>
          <div className="mt-3 border-t border-ink-700 pt-3">
            <span className="text-[12px] text-muted">Predicted recovery </span>
            <span className="tabular-nums text-[13px] text-[#e6e9ef]">
              {formatProbability(byEV.probability)}
            </span>
          </div>
        </div>
      </div>

      {diverges && (
        <p className="mt-4 rounded-lg border border-ink-700 bg-ink-850 p-3 text-[12px] leading-relaxed text-muted">
          <strong className="text-[#e6e9ef]">{titleCase(byProb.action)}</strong> converts{" "}
          {formatProbability(byProb.probability - byEV.probability, 1).replace("−", "")} better, but its
          incentive cost of {formatINR(byProb.incentive_cost)} leaves{" "}
          {formatDeltaEV(byProb.incremental_expected_value)} of incremental value.{" "}
          <strong className="text-[#e6e9ef]">{titleCase(byEV.action)}</strong> creates{" "}
          <strong className="text-pos">{formatINR(Math.abs(gap))} more</strong> economic value
          {selected === byEV.action ? " and was selected." : "."}
        </p>
      )}
    </Card>
  );
}

/** Horizontal ΔEV plot with an explicit zero line. */
function DeltaEVChart({ candidates }: { candidates: CandidateAction[] }) {
  const vals = candidates.map((c) => c.incremental_expected_value);
  const max = Math.max(...vals.map(Math.abs), 1);
  const zeroPct = 50;

  return (
    <div className="space-y-2">
      {candidates.map((c) => {
        const pct = (Math.abs(c.incremental_expected_value) / max) * 48;
        const pos = c.incremental_expected_value >= 0;
        return (
          <div key={c.action} className="flex items-center gap-3">
            <span className="w-40 shrink-0 truncate text-[12px] text-muted">{titleCase(c.action)}</span>
            <div className="relative h-5 flex-1 rounded bg-ink-850">
              <div className="absolute inset-y-0 w-px bg-ink-500" style={{ left: `${zeroPct}%` }} aria-hidden />
              <div className={`absolute inset-y-1 rounded-sm ${pos ? "bg-pos/60" : "bg-neg/60"}`}
                style={pos ? { left: `${zeroPct}%`, width: `${pct}%` }
                           : { right: `${100 - zeroPct}%`, width: `${pct}%` }} />
            </div>
            <span className={`w-24 shrink-0 text-right text-[12px] font-medium tabular-nums ${
              pos ? "text-pos" : "text-neg"}`}>
              {formatDeltaEV(c.incremental_expected_value)}
            </span>
          </div>
        );
      })}
    </div>
  );
}

export function CandidateTable({ candidates, selected, policyDecision }: {
  candidates: CandidateAction[]; selected: string | null; policyDecision: string | null;
}) {
  const [chart, setChart] = useState(false);
  const sorted = [...candidates].sort(
    (a, b) => b.incremental_expected_value - a.incremental_expected_value);

  return (
    <Card
      title={<span className="inline-flex items-center gap-1.5">Candidate actions <Tip text={DEV_TIP} /></span>}
      subtitle={`${candidates.length} actions scored, ranked by incremental expected value`}
      actions={
        <button onClick={() => setChart((v) => !v)}
          className="rounded-md border border-ink-600 px-2.5 py-1 text-[11px] text-muted hover:bg-ink-800">
          {chart ? "Table" : "Chart"}
        </button>
      }
    >
      {chart ? (
        <DeltaEVChart candidates={sorted} />
      ) : (
        <div className="-mx-5 overflow-x-auto">
          <table className="w-full text-[12px]">
            <thead>
              <tr className="border-b border-ink-700 text-left">
                <th className="px-5 py-2 font-medium text-muted-dim">Action</th>
                <th className="px-3 py-2 font-medium text-muted-dim">Recovery probability</th>
                <th className="px-3 py-2 text-right font-medium text-muted-dim">Incentive cost</th>
                <th className="px-3 py-2 text-right font-medium text-muted-dim">Expected value</th>
                <th className="px-5 py-2 text-right font-medium text-muted-dim">ΔEV</th>
              </tr>
            </thead>
            <tbody>
              {sorted.map((c) => {
                const isSelected = c.action === selected;
                return (
                  <tr key={c.action}
                    className={`border-b border-ink-800 last:border-0 ${
                      isSelected ? "bg-accent-soft/40" : ""}`}>
                    <td className="px-5 py-2.5">
                      <div className="flex items-center gap-2">
                        {isSelected && <span className="h-3 w-0.5 rounded-full bg-accent" aria-hidden />}
                        <span className={isSelected ? "font-semibold text-[#e6e9ef]" : "text-muted"}>
                          {titleCase(c.action)}
                        </span>
                        {isSelected && (
                          <span className="rounded border border-accent/40 px-1.5 py-0.5 text-[10px]
                            font-medium uppercase tracking-wide text-accent">
                            {policyDecision === "PASS" ? "Selected" : "Proposed"}
                          </span>
                        )}
                      </div>
                    </td>
                    <td className="px-3 py-2.5"><Probability value={c.probability} /></td>
                    <td className="px-3 py-2.5 text-right tabular-nums text-muted">
                      {c.incentive_cost > 0 ? formatINR(c.incentive_cost) : "—"}
                    </td>
                    <td className="px-3 py-2.5 text-right tabular-nums text-muted">
                      {formatINR(c.expected_value)}
                    </td>
                    <td className="px-5 py-2.5 text-right"><DeltaEV value={c.incremental_expected_value} /></td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </Card>
  );
}

export function WhyNot({ candidates, selected }: {
  candidates: CandidateAction[]; selected: string | null;
}) {
  const [open, setOpen] = useState(false);
  const others = [...candidates]
    .filter((c) => c.action !== selected)
    .sort((a, b) => b.incremental_expected_value - a.incremental_expected_value)
    .slice(0, 4);
  if (!others.length) return null;

  const sel = candidates.find((c) => c.action === selected);

  return (
    <div className="surface">
      <button onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center justify-between px-5 py-3.5 text-left">
        <span className="text-[13px] font-semibold text-[#e6e9ef]">Why not the alternatives?</span>
        <ChevronDown size={14}
          className={`text-muted-dim transition-transform ${open ? "rotate-180" : ""}`} />
      </button>
      {open && (
        <div className="space-y-2.5 border-t border-ink-700 p-5 animate-fade-up">
          {others.map((c) => {
            const negative = c.incremental_expected_value < 0;
            const belowSelected = sel && c.incremental_expected_value < sel.incremental_expected_value;
            return (
              <div key={c.action} className="rounded-lg border border-ink-700 bg-ink-850 p-3">
                <div className="flex items-center justify-between gap-3">
                  <span className="text-[12px] font-medium text-[#e6e9ef]">{titleCase(c.action)}</span>
                  <DeltaEV value={c.incremental_expected_value} />
                </div>
                <p className="mt-1.5 text-[12px] leading-relaxed text-muted">
                  Predicted recovery {formatProbability(c.probability)}
                  {c.incentive_cost > 0 && <> with an incentive cost of {formatINR(c.incentive_cost)}</>}.{" "}
                  {negative
                    ? "The cost exceeds the value the uplift creates, so it would destroy margin."
                    : belowSelected
                      ? "Positive, but below the selected action."
                      : "Not selected."}
                </p>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}