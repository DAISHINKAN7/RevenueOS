"use client";

import { CircleSlash, Sparkles } from "lucide-react";
import { Card, DeltaEV, Tip } from "@/components/ui/primitives";
import { formatINR, formatProbability, titleCase } from "@/lib/format";
import type { CandidateAction, PolicyDecision } from "@/lib/types";

export function SelectedAction({ selected, candidates, policy }: {
  selected: string | null; candidates: CandidateAction[]; policy: PolicyDecision | null;
}) {
  const action = candidates.find((c) => c.action === selected);
  if (!selected)
    return (
      <Card title="Decision">
        <p className="text-[12px] text-muted-dim">
          No action selected yet. Run analysis to score candidates.
        </p>
      </Card>
    );

  const isDoNothing = selected === "DO_NOTHING";
  const runnerUp = [...candidates]
    .filter((c) => c.action !== selected && c.action !== "DO_NOTHING")
    .sort((a, b) => b.incremental_expected_value - a.incremental_expected_value)[0];

  return (
    <Card className="border-accent/25">
      <div className="flex items-start justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 label">
            {isDoNothing ? <CircleSlash size={12} /> : <Sparkles size={12} />}
            {isDoNothing ? "Intelligent restraint" : "Selected action"}
          </div>
          <h3 className="mt-2 text-[24px] font-semibold tracking-tight text-[#e6e9ef]">
            {titleCase(selected)}
          </h3>
        </div>
        {!isDoNothing && (
          <span className="shrink-0 rounded-md border border-accent/40 bg-accent-soft px-2.5 py-1
            text-[11px] font-medium text-accent">
            Best economic action
          </span>
        )}
      </div>

      <div className="mt-5 grid grid-cols-2 gap-5 border-t border-ink-700 pt-4 sm:grid-cols-4">
        <div>
          <div className="label">Predicted recovery</div>
          <div className="mt-1 text-[17px] font-semibold tabular-nums text-[#e6e9ef]">
            {formatProbability(action?.probability ?? null)}
          </div>
        </div>
        <div>
          <div className="flex items-center gap-1 label">ΔEV <Tip text="Expected value relative to doing nothing." /></div>
          <div className="mt-1"><DeltaEV value={action?.incremental_expected_value ?? null} size="md" /></div>
        </div>
        <div>
          <div className="label">Intervention cost</div>
          <div className="mt-1 text-[17px] font-semibold tabular-nums text-muted">
            {formatINR(action?.incentive_cost ?? 0)}
          </div>
        </div>
        <div>
          <div className="label">Max downside</div>
          <div className="mt-1 text-[17px] font-semibold tabular-nums text-muted">
            {formatINR(policy?.maximum_authorized_downside ?? 0)}
          </div>
        </div>
      </div>

      <div className="mt-4 rounded-lg border border-ink-700 bg-ink-850 p-3.5">
        <div className="label mb-2">Why this action</div>
        <ul className="space-y-1.5 text-[12px] leading-relaxed text-muted">
          {isDoNothing ? (
            <li>· No intervention produced positive incremental expected value, so the system declined to spend.</li>
          ) : (
            <>
              <li>· Highest valid ΔEV at {formatINR(action?.incremental_expected_value ?? 0)}</li>
              <li>· Predicted recovery probability {formatProbability(action?.probability ?? null)}</li>
              <li>· Intervention costs {formatINR(action?.incentive_cost ?? 0)} only if recovery succeeds</li>
              {runnerUp && (
                <li>· Next best was {titleCase(runnerUp.action)} at{" "}
                  {formatINR(runnerUp.incremental_expected_value)}</li>
              )}
            </>
          )}
          {policy && (
            <li>· Policy {policy.decision === "PASS" ? "passed all" : `returned ${policy.decision} across`}{" "}
              {policy.rules.length} rules, downside capped at {formatINR(policy.maximum_authorized_downside)}</li>
          )}
        </ul>
      </div>
    </Card>
  );
}