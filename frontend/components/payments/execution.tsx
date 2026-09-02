"use client";

import { useState } from "react";
import { ArrowRight, CheckCircle2, CreditCard, Loader2, Radio, XCircle } from "lucide-react";
import { Button, Card, Mono, TestModeBadge } from "@/components/ui/primitives";
import { formatINR, titleCase } from "@/lib/format";
import type { ExecutionRecord, OpportunityDetail } from "@/lib/types";

declare global { interface Window { Razorpay?: new (o: unknown) => { open: () => void } } }

const PAYMENT_STATUS: Record<string, string> = {
  CAPTURED: "text-pos", FAILED: "text-neg", SUBMITTED: "text-accent",
  PENDING: "text-muted", AUTHORIZED: "text-accent",
};

/** The workflow stepper. Failure is shown as a branch, not a dead end. */
export function WorkflowProgress({ state, attempt }: { state: string; attempt: number }) {
  const steps = ["DETECTED", "AUTHORIZED", "AWAITING_PAYMENT", "RECOVERED"];
  const failed = state === "PAYMENT_FAILED_RECOVERABLE" || state === "EXECUTION_FAILED";
  const approval = state === "AWAITING_APPROVAL";
  const idx = steps.indexOf(state);
  const current = idx >= 0 ? idx : state === "RECOVERED" ? 3 : failed ? 2 : approval ? 1 : 0;

  return (
    <div className="flex items-center gap-1.5 overflow-x-auto">
      {steps.map((s, i) => {
        const done = i < current || state === "RECOVERED";
        const active = i === current;
        return (
          <div key={s} className="flex shrink-0 items-center gap-1.5">
            <div className={`flex items-center gap-1.5 rounded-md border px-2 py-1 text-[11px] ${
              state === "RECOVERED" && i === 3 ? "border-pos/40 bg-pos-soft text-pos"
              : active && failed ? "border-neg/40 bg-neg-soft text-neg"
              : active && approval ? "border-warn/40 bg-warn-soft text-warn"
              : active ? "border-accent/40 bg-accent-soft text-accent"
              : done ? "border-ink-600 bg-ink-800 text-muted"
              : "border-ink-700 text-muted-dim"}`}>
              <span className="h-1.5 w-1.5 rounded-full bg-current opacity-70" aria-hidden />
              {titleCase(s)}
            </div>
            {i < steps.length - 1 && <ArrowRight size={11} className="text-ink-500" />}
          </div>
        );
      })}
      {failed && (
        <>
          <ArrowRight size={11} className="text-ink-500" />
          <div className="shrink-0 rounded-md border border-warn/40 bg-warn-soft px-2 py-1 text-[11px] text-warn">
            Attempt {attempt + 1} available
          </div>
        </>
      )}
    </div>
  );
}

/** Shown when attempt 2 differs from attempt 1 — the adaptive-retry story. */
export function RetryComparison({ executions, failureReason }: {
  executions: ExecutionRecord[]; failureReason: string | null;
}) {
  if (executions.length < 2) return null;
  const [first, second] = [executions[0], executions[executions.length - 1]];
  const changed = first.action !== second.action;

  return (
    <Card
      title={changed ? "Strategy changed after new evidence" : "Second attempt"}
      subtitle={changed
        ? "The observed blocker moved the decision to a different family of action."
        : "The same action was retried under a fresh policy check."}
    >
      <div className="grid items-center gap-3 sm:grid-cols-[1fr_auto_1fr]">
        <div className="rounded-lg border border-neg/25 bg-neg-soft/40 p-3.5">
          <div className="label">Attempt {first.attempt}</div>
          <div className="mt-1.5 text-[14px] font-semibold text-[#e6e9ef]">{titleCase(first.action)}</div>
          <div className="mt-1 text-[12px] text-neg">{first.status}</div>
          <div className="mt-2 border-t border-ink-700 pt-2">
            <Mono copyable>{first.execution_id}</Mono>
          </div>
        </div>

        <div className="flex flex-col items-center gap-1 px-2 text-center">
          <Radio size={14} className="text-warn" />
          <div className="text-[10px] uppercase tracking-wide text-muted-dim">New evidence</div>
          <div className="mono text-warn">{failureReason ?? "payment failure"}</div>
        </div>

        <div className="rounded-lg border border-accent/25 bg-accent-soft/40 p-3.5">
          <div className="label">Attempt {second.attempt}</div>
          <div className="mt-1.5 text-[14px] font-semibold text-[#e6e9ef]">{titleCase(second.action)}</div>
          <div className="mt-1 text-[12px] text-accent">{second.status}</div>
          <div className="mt-2 border-t border-ink-700 pt-2">
            <Mono copyable>{second.execution_id}</Mono>
          </div>
        </div>
      </div>
      <p className="mt-3 text-[11px] text-muted-dim">
        Each attempt carries a distinct idempotency key, so a repeated request cannot create a second order.
      </p>
    </Card>
  );
}

/**
 * Chooses how the next execution runs. Offline is the default so the whole
 * demo works with no tunnel and no credentials; live creates a real Test Mode
 * order. Locked once an execution exists, so the record of how money moved
 * cannot be rewritten after the fact.
 */
function ModeSelector({ opportunityId, mode, onChange }: {
  opportunityId: string; mode: string; onChange: () => void;
}) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function set(next: "SIMULATOR" | "RAZORPAY_TEST") {
    if (next === mode) return;
    setBusy(true); setError(null);
    try {
      const { api } = await import("@/lib/api");
      await api.setExecutionMode(opportunityId, next);
      onChange();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not change mode");
    } finally { setBusy(false); }
  }

  const options = [
    { key: "SIMULATOR" as const, label: "Offline",
      hint: "Completes locally · no tunnel needed" },
    { key: "RAZORPAY_TEST" as const, label: "Razorpay Test Mode",
      hint: "Real Test Mode order · needs a webhook tunnel" },
  ];

  return (
    <div className="mb-4 border-b border-ink-700 pb-4">
      <div className="label mb-2">Execution mode</div>
      <div className="flex flex-wrap gap-2">
        {options.map((o) => (
          <button key={o.key} onClick={() => set(o.key)} disabled={busy}
            className={`flex-1 rounded-md border px-3 py-2 text-left transition-colors
              disabled:opacity-50 ${
                mode === o.key
                  ? "border-accent/40 bg-accent-soft"
                  : "border-ink-600 hover:bg-ink-800"}`}>
            <div className={`text-[12px] font-medium ${
              mode === o.key ? "text-accent" : "text-[#e6e9ef]"}`}>{o.label}</div>
            <div className="mt-0.5 text-[11px] text-muted-dim">{o.hint}</div>
          </button>
        ))}
      </div>
      {error && <p className="mt-2 text-[11px] text-neg">{error}</p>}
    </div>
  );
}

/**
 * Offline demo control. Drives the same reconciliation path a webhook uses, but
 * every record it produces is marked as simulated — the UI says so, and so does
 * the audit trail. It exists because a verified webhook needs a public tunnel,
 * which is not always available.
 */
function SimulatePayment({ opportunityId, onDone }: {
  opportunityId: string; onDone: () => void;
}) {
  const [busy, setBusy] = useState<string | null>(null);
  const [mode, setMode] = useState("card_declined");
  const [error, setError] = useState<string | null>(null);

  async function run(outcome: "success" | "failure") {
    setBusy(outcome); setError(null);
    try {
      const { api } = await import("@/lib/api");
      await api.simulatePayment(opportunityId, outcome, mode);
      onDone();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Simulation failed");
    } finally { setBusy(null); }
  }

  return (
    <div className="rounded-lg border border-dashed border-ink-600 bg-ink-850 p-3.5">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <div className="label">Simulated payment · offline demo</div>
          <p className="mt-1 max-w-md text-[11px] leading-relaxed text-muted-dim">
            Applies the same state transitions and outcome booking a verified webhook would.
            Recorded as <span className="mono">provider: SIMULATOR</span>, never as a Razorpay event.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <select value={mode} onChange={(e) => setMode(e.target.value)}
            aria-label="Simulated failure mode"
            className="rounded-md border border-ink-600 bg-ink-800 px-2 py-1.5 text-[11px] text-muted">
            <option value="card_declined">Card declined</option>
            <option value="bank_timeout">Bank timeout</option>
            <option value="insufficient_funds">Insufficient funds</option>
            <option value="authentication">Authentication failed</option>
            <option value="user_cancelled">Customer cancelled</option>
          </select>
          <Button onClick={() => run("failure")} loading={busy === "failure"}>
            <XCircle size={12} /> Fail
          </Button>
          <Button variant="primary" onClick={() => run("success")} loading={busy === "success"}>
            <CheckCircle2 size={12} /> Succeed
          </Button>
        </div>
      </div>
      {error && <p className="mt-2 text-[11px] text-neg">{error}</p>}
    </div>
  );
}

export function ExecutionPanel({ detail, onRefresh }: {
  detail: OpportunityDetail; onRefresh: () => void;
}) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [submitted, setSubmitted] = useState(false);

  const { opportunity, execution, outcome, selected_action, policy_decision } = detail;
  // Provenance matters: a simulated recovery must never look like a verified one.
  const simulated = detail.audit_timeline.some(
    (e) => e.event_type === "SIMULATED_PAYMENT_EVENT");
  const canExecute = opportunity.state === "AUTHORIZED";
  const waiting = opportunity.state === "AWAITING_PAYMENT";

  async function handleExecute() {
    setBusy(true); setError(null);
    try {
      const { api } = await import("@/lib/api");
      const res = await api.execute(opportunity.opportunity_id);
      // Offline executions never return a checkout payload, so the Razorpay
      // modal is never opened and the simulated panel takes over instead.
      if (res.checkout?.razorpay_order_id) {
        await launchCheckout(res.checkout);
        setSubmitted(true);
      }
      onRefresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Execution failed");
    } finally { setBusy(false); }
  }

  async function launchCheckout(c: NonNullable<Awaited<ReturnType<typeof import("@/lib/api").api.execute>>["checkout"]>) {
    if (!window.Razorpay) {
      await new Promise<void>((resolve, reject) => {
        const s = document.createElement("script");
        s.src = "https://checkout.razorpay.com/v1/checkout.js";
        s.onload = () => resolve(); s.onerror = () => reject();
        document.body.appendChild(s);
      });
    }
    new window.Razorpay!({
      key: c.razorpay_key_id, order_id: c.razorpay_order_id,
      amount: c.amount_paise, currency: c.currency, name: c.display_name,
      description: `Recovery — ${selected_action}`,
      handler: () => { setSubmitted(true); onRefresh(); },
    }).open();
  }

  return (
    <Card title="Execution and payment" actions={<TestModeBadge />}>
      {canExecute && (
        <div className="mb-4 rounded-lg border border-ink-700 bg-ink-850 p-4">
          <ModeSelector opportunityId={opportunity.opportunity_id}
            mode={opportunity.execution_mode} onChange={onRefresh} />
          <div className="label mb-2.5">Confirm before executing</div>
          <dl className="grid grid-cols-2 gap-x-6 gap-y-2 text-[12px] sm:grid-cols-4">
            <div><dt className="text-muted-dim">Action</dt>
              <dd className="mt-0.5 font-medium text-[#e6e9ef]">{titleCase(selected_action ?? "—")}</dd></div>
            <div><dt className="text-muted-dim">Amount</dt>
              <dd className="mt-0.5 tabular-nums text-[#e6e9ef]">
                {formatINR(detail.checkout_summary.cart_value ?? null)}</dd></div>
            <div><dt className="text-muted-dim">Max downside</dt>
              <dd className="mt-0.5 tabular-nums text-[#e6e9ef]">
                {formatINR(policy_decision?.maximum_authorized_downside ?? 0)}</dd></div>
            <div><dt className="text-muted-dim">Environment</dt>
              <dd className="mt-0.5 text-warn">Test Mode</dd></div>
          </dl>
          <div className="mt-3.5 flex items-center gap-3">
            <Button variant="primary" onClick={handleExecute} loading={busy}>
              <CreditCard size={13} /> Execute recovery
            </Button>
            {error && <span className="text-[12px] text-neg">{error}</span>}
          </div>
        </div>
      )}

      {waiting && (
        <div className="mb-4 space-y-3">
          <div className="flex items-start gap-3 rounded-lg border border-accent/25 bg-accent-soft p-3.5">
            <Loader2 size={14} className="mt-0.5 animate-spin text-accent" />
            <div>
              <div className="text-[12px] font-medium text-[#e6e9ef]">
                {submitted ? "Payment submitted — awaiting verified webhook" : "Waiting for payment confirmation"}
              </div>
              <p className="mt-1 text-[11px] leading-relaxed text-muted">
                Order created. Recovery is confirmed only by a signature-verified webhook,
                never by the browser callback.
              </p>
            </div>
          </div>
          <SimulatePayment opportunityId={opportunity.opportunity_id} onDone={onRefresh} />
        </div>
      )}

      {outcome && (
        <div className="mb-4 rounded-lg border border-pos/25 bg-pos-soft p-4">
          <div className="flex items-center gap-2">
            <span className="label text-pos">Recovery confirmed</span>
            {simulated && (
              <span className="rounded border border-ink-600 px-1.5 py-0.5 text-[10px]
                uppercase tracking-wide text-muted-dim">Simulated</span>
            )}
          </div>
          <div className="mt-2 grid grid-cols-2 gap-4 sm:grid-cols-3">
            <div><div className="text-[11px] text-muted-dim">Net recovered GMV</div>
              <div className="mt-0.5 text-[17px] font-semibold tabular-nums text-pos">
                {formatINR(outcome.net_recovered_gmv)}</div></div>
            <div><div className="text-[11px] text-muted-dim">Realized contribution</div>
              <div className="mt-0.5 text-[17px] font-semibold tabular-nums text-[#e6e9ef]">
                {formatINR(outcome.realized_contribution)}</div></div>
            <div><div className="text-[11px] text-muted-dim">Discount given</div>
              <div className="mt-0.5 text-[17px] font-semibold tabular-nums text-muted">
                {formatINR(outcome.discount_amount)}</div></div>
          </div>
        </div>
      )}

      {execution.length > 0 ? (
        <div className="-mx-5 overflow-x-auto">
          <table className="w-full text-[12px]">
            <thead>
              <tr className="border-b border-ink-700 text-left">
                {["Attempt", "Action", "Status", "Order", "Payment", "Amount"].map((h) => (
                  <th key={h} className="px-3 py-2 font-medium text-muted-dim first:pl-5 last:pr-5">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {execution.map((e) => (
                <tr key={e.execution_id} className="border-b border-ink-800 last:border-0">
                  <td className="py-2.5 pl-5 pr-3 tabular-nums text-muted">{e.attempt}</td>
                  <td className="px-3 py-2.5 text-[#e6e9ef]">{titleCase(e.action)}</td>
                  <td className={`px-3 py-2.5 font-medium ${PAYMENT_STATUS[e.status] ?? "text-muted"}`}>
                    {e.status}</td>
                  <td className="px-3 py-2.5">{e.order_id ? <Mono copyable>{e.order_id}</Mono> : "—"}</td>
                  <td className="px-3 py-2.5">{e.payment_id ? <Mono copyable>{e.payment_id}</Mono> : "—"}</td>
                  <td className="py-2.5 pl-3 pr-5 tabular-nums text-muted">{formatINR(e.amount)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        !canExecute && <p className="text-[12px] text-muted-dim">No execution has been created yet.</p>
      )}
    </Card>
  );
}