"use client";

/**
 * Agentic commerce — the Track 01 screen.
 *
 * Deliberately self-contained: only Tailwind core utilities and React state, no
 * imports from the shared primitives, so it cannot break if those change.
 *
 * The screen's whole job is to make one thing legible on video: the buyer asks
 * for a price, and the counter-offer that comes back is the *merchant's reserve*
 * — with the binding constraint named. Not a haggle. The rule ledger is on
 * screen for exactly that reason.
 */

import { useState } from "react";

const API =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

type Product = {
  product_id: string;
  name: string;
  category: string;
  subcategory: string;
  list_price: number;
  in_stock: boolean;
  inventory_level: number;
};

type Rule = {
  rule_id: string;
  passed: boolean;
  reason: string;
  input_value: string | null;
  threshold: string | null;
};

type Offer = {
  decision: "ACCEPT" | "COUNTER" | "REJECT" | "REQUIRE_APPROVAL";
  reason_code: string;
  list_price: number;
  requested_unit_price: number;
  final_unit_price: number;
  reserve_unit_price: number;
  binding_constraint: string;
  order_total: number;
  net_contribution: number;
  margin_percent: number;
  discount_percent: number;
  discount_amount: number;
  message: string;
  round: number;
  rounds_remaining: number;
  session_status?: string;
  rules: Rule[];
};

type Turn = {
  actor: "BUYER" | "MERCHANT";
  round: number;
  decision: string | null;
  requested_unit_price: number | null;
  offered_unit_price: number | null;
  message: string;
  message_source: string;
};

const inr = (n: number) =>
  "₹" + n.toLocaleString("en-IN", { maximumFractionDigits: 0 });

const DECISION_STYLE: Record<string, string> = {
  ACCEPT: "bg-pos-soft text-pos border-pos/30",
  COUNTER: "bg-warn-soft text-warn border-warn/30",
  REJECT: "bg-neg-soft text-neg border-neg/30",
  REQUIRE_APPROVAL: "bg-accent-soft text-accent border-accent/30",
};

const humanRule = (id: string) =>
  id.replace("RULE_AC_", "").replaceAll("_", " ").toLowerCase();

export default function CommercePage() {
  const [request, setRequest] = useState(
    "I need wireless headphones under Rs 6000 with good battery life"
  );
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [constraints, setConstraints] = useState<Record<string, unknown> | null>(
    null
  );
  const [constraintsSource, setConstraintsSource] = useState<string>("");
  const [matches, setMatches] = useState<Product[]>([]);
  const [selected, setSelected] = useState<Product | null>(null);
  const [askPrice, setAskPrice] = useState<string>("");
  const [offer, setOffer] = useState<Offer | null>(null);
  const [turns, setTurns] = useState<Turn[]>([]);
  const [bridge, setBridge] = useState<{ opportunity_id: string } | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function call(path: string, body?: unknown) {
    const res = await fetch(`${API}${path}`, {
      method: body ? "POST" : "GET",
      headers: { "Content-Type": "application/json" },
      body: body ? JSON.stringify(body) : undefined,
    });
    const json = await res.json().catch(() => ({}));
    if (!res.ok) {
      throw new Error(
        json?.detail?.message ?? json?.detail?.error_code ?? `HTTP ${res.status}`
      );
    }
    return json;
  }

  function reset() {
    setSessionId(null);
    setConstraints(null);
    setMatches([]);
    setSelected(null);
    setOffer(null);
    setTurns([]);
    setBridge(null);
    setError(null);
    setAskPrice("");
  }

  async function startSession() {
    setBusy("session");
    setError(null);
    try {
      reset();
      const j = await call("/api/agent-commerce/session", { request });
      setSessionId(j.session_id);
      setConstraints(j.constraints);
      setConstraintsSource(j.constraints_source);
      setMatches(j.matches);
      if (j.matches.length === 0)
        setError("No catalog product matches those constraints.");
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(null);
    }
  }

  function pick(p: Product) {
    setSelected(p);
    setOffer(null);
    // Open aggressively by default — the point of the demo is the gate biting.
    setAskPrice(String(Math.round(p.list_price * 0.82)));
  }

  async function sendOffer(price?: number) {
    if (!selected) return;
    setBusy("offer");
    setError(null);
    try {
      const j: Offer = await call("/api/agent-commerce/offer", {
        session_id: sessionId,
        product_id: selected.product_id,
        requested_price: price ?? Number(askPrice),
        quantity: 1,
      });
      setOffer(j);
      await refreshTurns();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(null);
    }
  }

  async function buyerResponds() {
    if (!sessionId) return;
    setBusy("respond");
    setError(null);
    try {
      await call("/api/agent-commerce/respond", { session_id: sessionId });
      await refreshTurns();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(null);
    }
  }

  async function refreshTurns() {
    if (!sessionId) return;
    try {
      const j = await call(`/api/agent-commerce/session/${sessionId}`);
      setTurns(j.turns ?? []);
    } catch {
      /* transcript is decoration; never block the flow on it */
    }
  }

  async function goToCheckout() {
    if (!sessionId) return;
    setBusy("checkout");
    setError(null);
    try {
      const j = await call("/api/agent-commerce/checkout", {
        session_id: sessionId,
        execution_mode: "SIMULATOR",
      });
      setBridge({ opportunity_id: j.opportunity_id });
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(null);
    }
  }

  const canCheckout = offer?.decision === "ACCEPT" && !bridge;

  return (
    <div className="mx-auto max-w-6xl px-6 py-10 text-[#e6e9ef]">
      <header className="mb-8">
        <p className="text-xs uppercase tracking-[0.2em] text-muted-dim">
          Track 01 · Agentic Commerce
        </p>
        <h1 className="mt-2 text-3xl font-semibold text-[#e6e9ef]">
          AI buyer, bounded merchant
        </h1>
        <p className="mt-3 max-w-3xl text-sm leading-relaxed text-muted">
          An AI buyer states constraints in natural language. The merchant agent
          searches the catalog and rules on a requested price. Every counter-offer
          is the merchant&apos;s{" "}
          <span className="text-[#e6e9ef]">reserve price</span> — computed from
          COGS, fulfilment cost, return rate and merchant policy. The language
          model writes the sentences. It never touches the arithmetic.
        </p>
      </header>

      {error && (
        <div className="mb-6 rounded-lg border border-neg/30 bg-neg-soft px-4 py-3 text-sm text-neg">
          {error}
        </div>
      )}

      {/* 1 — buyer request */}
      <section className="mb-6 rounded-xl border border-ink-700 bg-ink-900 p-5">
        <div className="mb-3 flex items-center gap-2">
          <span className="flex h-6 w-6 items-center justify-center rounded-full bg-ink-800 text-xs font-semibold">
            1
          </span>
          <h2 className="text-sm font-medium text-[#c7ccd8]">Buyer request</h2>
        </div>
        <div className="flex flex-col gap-3 sm:flex-row">
          <input
            value={request}
            onChange={(e) => setRequest(e.target.value)}
            className="flex-1 rounded-lg border border-ink-600 bg-ink-950 px-4 py-2.5 text-sm text-[#e6e9ef] outline-none focus:border-accent"
            placeholder="e.g. I need a cookware set under ₹4,000"
          />
          <button
            onClick={startSession}
            disabled={busy !== null}
            className="rounded-lg bg-accent px-5 py-2.5 text-sm font-medium text-ink-950 hover:bg-accent/90 disabled:opacity-40"
          >
            {busy === "session" ? "Parsing…" : "Send to merchant"}
          </button>
        </div>

        {constraints && (
          <div className="mt-4 flex flex-wrap items-center gap-2 text-xs">
            <span className="text-muted-dim">parsed constraints</span>
            <span className="rounded border border-ink-600 px-2 py-0.5 text-muted">
              {constraintsSource === "LLM" ? "LLM parse" : "rule parse"}
            </span>
            {Object.entries(constraints)
              .filter(([, v]) => v !== null && v !== undefined && v !== "")
              .map(([k, v]) => (
                <span
                  key={k}
                  className="rounded bg-ink-800 px-2 py-0.5 font-mono text-[#c7ccd8]"
                >
                  {k}={String(v)}
                </span>
              ))}
          </div>
        )}
      </section>

      {/* 2 — catalog matches */}
      {matches.length > 0 && (
        <section className="mb-6 rounded-xl border border-ink-700 bg-ink-900 p-5">
          <div className="mb-3 flex items-center gap-2">
            <span className="flex h-6 w-6 items-center justify-center rounded-full bg-ink-800 text-xs font-semibold">
              2
            </span>
            <h2 className="text-sm font-medium text-[#c7ccd8]">
              Catalog matches
            </h2>
            <span className="text-xs text-muted-dim">
              buyer sees list price only — never COGS
            </span>
          </div>
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {matches.map((p) => (
              <button
                key={p.product_id}
                onClick={() => pick(p)}
                className={`rounded-lg border p-4 text-left transition ${
                  selected?.product_id === p.product_id
                    ? "border-accent bg-ink-800"
                    : "border-ink-700 bg-ink-950 hover:border-ink-500"
                }`}
              >
                <p className="font-mono text-[11px] text-muted-dim">
                  {p.product_id}
                </p>
                <p className="mt-1 text-sm font-medium text-[#e6e9ef]">
                  {p.name}
                </p>
                <p className="text-xs text-muted-dim">{p.subcategory}</p>
                <p className="mt-2 text-lg font-semibold text-[#e6e9ef]">
                  {inr(p.list_price)}
                </p>
                <p className="text-[11px] text-muted-dim">
                  {p.inventory_level} in stock
                </p>
              </button>
            ))}
          </div>
        </section>
      )}

      {/* 3 — the offer */}
      {selected && (
        <section className="mb-6 rounded-xl border border-ink-700 bg-ink-900 p-5">
          <div className="mb-3 flex items-center gap-2">
            <span className="flex h-6 w-6 items-center justify-center rounded-full bg-ink-800 text-xs font-semibold">
              3
            </span>
            <h2 className="text-sm font-medium text-[#c7ccd8]">
              Request a price for {selected.product_id}
            </h2>
          </div>
          <div className="flex flex-col gap-3 sm:flex-row sm:items-center">
            <div className="flex items-center gap-2">
              <span className="text-sm text-muted-dim">
                list {inr(selected.list_price)} · asking
              </span>
              <input
                value={askPrice}
                onChange={(e) => setAskPrice(e.target.value)}
                inputMode="numeric"
                className="w-32 rounded-lg border border-ink-600 bg-ink-950 px-3 py-2 font-mono text-sm text-[#e6e9ef] outline-none focus:border-accent"
              />
            </div>
            <button
              onClick={() => sendOffer()}
              disabled={busy !== null}
              className="rounded-lg bg-accent px-5 py-2 text-sm font-medium text-ink-950 hover:bg-accent/90 disabled:opacity-40"
            >
              {busy === "offer" ? "Evaluating…" : "Make offer"}
            </button>
            {offer?.decision === "COUNTER" && (
              <button
                onClick={() => sendOffer(offer.final_unit_price)}
                disabled={busy !== null}
                className="rounded-lg border border-pos/40 bg-pos-soft px-5 py-2 text-sm font-medium text-pos hover:bg-pos/20 disabled:opacity-40"
              >
                Accept counter at {inr(offer.final_unit_price)}
              </button>
            )}
            <button
              onClick={buyerResponds}
              disabled={busy !== null || !offer}
              className="rounded-lg border border-ink-600 px-4 py-2 text-sm text-[#c7ccd8] hover:border-accent disabled:opacity-40"
            >
              Let the AI buyer decide
            </button>
          </div>
        </section>
      )}

      {/* 4 — the ruling */}
      {offer && (
        <section className="mb-6 grid gap-6 lg:grid-cols-5">
          <div className="lg:col-span-2 rounded-xl border border-ink-700 bg-ink-900 p-5">
            <div className="mb-4 flex items-center justify-between">
              <h2 className="text-sm font-medium text-[#c7ccd8]">
                Merchant ruling
              </h2>
              <span
                className={`rounded border px-2.5 py-1 text-xs font-semibold ${
                  DECISION_STYLE[offer.decision]
                }`}
              >
                {offer.decision}
              </span>
            </div>

            <p className="mb-5 text-sm leading-relaxed text-[#c7ccd8]">
              {offer.message}
            </p>

            <dl className="space-y-2 text-sm">
              {[
                ["Buyer asked", inr(offer.requested_unit_price)],
                ["List price", inr(offer.list_price)],
                ["Merchant price", inr(offer.final_unit_price)],
                ["Order total", inr(offer.order_total)],
                [
                  "Discount",
                  `${inr(offer.discount_amount)} (${offer.discount_percent.toFixed(
                    1
                  )}%)`,
                ],
                ["Net contribution", inr(offer.net_contribution)],
                ["Margin", `${offer.margin_percent.toFixed(1)}%`],
              ].map(([k, v]) => (
                <div
                  key={k}
                  className="flex justify-between border-b border-ink-700 pb-1.5"
                >
                  <dt className="text-muted-dim">{k}</dt>
                  <dd className="font-mono text-[#e6e9ef]">{v}</dd>
                </div>
              ))}
            </dl>

            <div className="mt-5 rounded-lg border border-ink-700 bg-ink-950 p-3">
              <p className="text-[11px] uppercase tracking-wider text-muted-dim">
                Binding constraint
              </p>
              <p className="mt-1 font-mono text-xs text-warn">
                {offer.binding_constraint}
              </p>
              <p className="mt-2 text-xs leading-relaxed text-muted-dim">
                The counter is the lowest price at which {humanRule(offer.binding_constraint)}{" "}
                still holds. Ask again and you get the same number.
              </p>
            </div>

            {canCheckout && (
              <button
                onClick={goToCheckout}
                disabled={busy !== null}
                className="mt-5 w-full rounded-lg bg-pos px-4 py-2.5 text-sm font-semibold text-ink-950 hover:bg-pos/85 disabled:opacity-40"
              >
                {busy === "checkout"
                  ? "Creating…"
                  : "Continue to checkout →"}
              </button>
            )}

            {bridge && (
              <div className="mt-5 rounded-lg border border-pos/30 bg-pos-soft p-4">
                <p className="text-xs uppercase tracking-wider text-pos">
                  Handed to the recovery engine
                </p>
                <p className="mt-2 text-sm text-[#c7ccd8]">
                  The negotiated cart is now an ordinary opportunity. Same
                  predictor, same policy gate, same Razorpay path — and if the
                  payment fails, the same recovery loop.
                </p>
                <a
                  href={`/opportunities/${bridge.opportunity_id}`}
                  className="mt-3 inline-block rounded bg-pos px-4 py-2 text-xs font-semibold text-ink-950 hover:bg-pos/85"
                >
                  Open {bridge.opportunity_id} →
                </a>
              </div>
            )}
          </div>

          {/* rule ledger */}
          <div className="lg:col-span-3 rounded-xl border border-ink-700 bg-ink-900 p-5">
            <h2 className="mb-1 text-sm font-medium text-[#c7ccd8]">
              Policy rules evaluated
            </h2>
            <p className="mb-4 text-xs text-muted-dim">
              Round {offer.round} of {offer.round + offer.rounds_remaining} ·
              deterministic · no model output on this path
            </p>
            <div className="space-y-1.5">
              {offer.rules.map((r) => (
                <div
                  key={r.rule_id}
                  className="flex items-start gap-3 rounded-lg border border-ink-700 bg-ink-950 px-3 py-2.5"
                >
                  <span
                    className={`mt-0.5 text-xs font-bold ${
                      r.passed ? "text-pos" : "text-neg"
                    }`}
                  >
                    {r.passed ? "PASS" : "FAIL"}
                  </span>
                  <div className="min-w-0 flex-1">
                    <p className="font-mono text-[11px] text-muted">
                      {r.rule_id}
                    </p>
                    <p className="mt-0.5 text-xs text-muted">{r.reason}</p>
                  </div>
                  {r.threshold && (
                    <span className="shrink-0 font-mono text-[11px] text-muted-dim">
                      {r.input_value} / {r.threshold}
                    </span>
                  )}
                </div>
              ))}
            </div>
          </div>
        </section>
      )}

      {/* 5 — transcript */}
      {turns.length > 0 && (
        <section className="rounded-xl border border-ink-700 bg-ink-900 p-5">
          <h2 className="mb-1 text-sm font-medium text-[#c7ccd8]">
            Negotiation transcript
          </h2>
          <p className="mb-4 text-xs text-muted-dim">
            Persisted turn by turn — this is audit history, not chat scrollback.
          </p>
          <div className="space-y-3">
            {turns.map((t, i) => (
              <div
                key={i}
                className={`flex ${
                  t.actor === "BUYER" ? "justify-start" : "justify-end"
                }`}
              >
                <div
                  className={`max-w-[75%] rounded-lg border px-4 py-2.5 ${
                    t.actor === "BUYER"
                      ? "border-ink-600 bg-ink-950"
                      : "border-ink-600 bg-ink-800"
                  }`}
                >
                  <p className="mb-1 flex items-center gap-2 text-[11px] uppercase tracking-wider text-muted-dim">
                    {t.actor === "BUYER" ? "AI buyer" : "Merchant agent"}
                    {t.decision && (
                      <span className="font-mono normal-case text-muted">
                        {t.decision}
                      </span>
                    )}
                    <span className="rounded bg-ink-800 px-1.5 text-[10px] normal-case text-muted-dim">
                      {t.message_source}
                    </span>
                  </p>
                  <p className="text-sm text-[#c7ccd8]">{t.message}</p>
                </div>
              </div>
            ))}
          </div>
        </section>
      )}
    </div>
  );
}