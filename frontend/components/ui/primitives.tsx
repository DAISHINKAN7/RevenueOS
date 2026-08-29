"use client";

import { Check, Copy, Info, X } from "lucide-react";
import { useState, type ReactNode } from "react";
import { formatDeltaEV, formatProbability, titleCase } from "@/lib/format";
import type { PolicyStatus, WorkflowState } from "@/lib/types";

/* ------------------------------------------------------------------ state */

const STATE_STYLES: Record<string, string> = {
  RECOVERED: "bg-pos-soft text-pos border-pos/30",
  NOT_RECOVERED: "bg-ink-800 text-muted border-ink-600",
  AWAITING_PAYMENT: "bg-accent-soft text-accent border-accent/30",
  AWAITING_APPROVAL: "bg-warn-soft text-warn border-warn/30",
  PAYMENT_FAILED_RECOVERABLE: "bg-neg-soft text-neg border-neg/30",
  EXECUTION_FAILED: "bg-neg-soft text-neg border-neg/30",
  AUTHORIZED: "bg-accent-soft text-accent border-accent/30",
  DETECTED: "bg-ink-800 text-muted border-ink-600",
  STOPPED: "bg-ink-800 text-muted-dim border-ink-600",
  EXPIRED: "bg-ink-800 text-muted-dim border-ink-600",
};

export function StateBadge({ state, size = "sm" }: { state: WorkflowState | string; size?: "sm" | "md" }) {
  const cls = STATE_STYLES[state] ?? "bg-ink-800 text-muted border-ink-600";
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-md border font-medium ${cls} ${
        size === "md" ? "px-2.5 py-1 text-[12px]" : "px-2 py-0.5 text-[11px]"
      }`}
    >
      {/* Never colour-only: the label carries the meaning too. */}
      <span className="h-1.5 w-1.5 rounded-full bg-current opacity-70" aria-hidden />
      {titleCase(state)}
    </span>
  );
}

const POLICY_STYLES: Record<string, string> = {
  PASS: "bg-pos-soft text-pos border-pos/30",
  REQUIRE_APPROVAL: "bg-warn-soft text-warn border-warn/30",
  REJECT: "bg-neg-soft text-neg border-neg/30",
  STOP: "bg-neg-soft text-neg border-neg/30",
};

export function PolicyBadge({ status }: { status: PolicyStatus | string }) {
  return (
    <span className={`inline-flex items-center rounded-md border px-2 py-0.5 text-[11px] font-medium ${
      POLICY_STYLES[status] ?? "bg-ink-800 text-muted border-ink-600"}`}>
      {titleCase(status)}
    </span>
  );
}

/* --------------------------------------------------------------- economics */

/**
 * Economic value carries semantic colour; probability never does. A
 * high-converting action with negative value must not read as good.
 */
export function DeltaEV({ value, size = "sm", showLabel = false }:
  { value: number | null; size?: "sm" | "md" | "lg"; showLabel?: boolean }) {
  const tone = value === null ? "text-muted"
    : value > 0 ? "text-pos" : value < 0 ? "text-neg" : "text-muted";
  const scale = size === "lg" ? "text-metric" : size === "md" ? "text-[17px]" : "text-[13px]";
  return (
    <span className="inline-flex flex-col">
      <span className={`font-semibold tabular-nums ${tone} ${scale}`}>{formatDeltaEV(value)}</span>
      {showLabel && <span className="label mt-0.5">Incremental expected value</span>}
    </span>
  );
}

/** Neutral by design — see DeltaEV. */
export function Probability({ value, withBar = true }: { value: number | null; withBar?: boolean }) {
  const pct = value === null ? 0 : Math.min(1, Math.max(0, value));
  return (
    <span className="inline-flex items-center gap-2">
      <span className="tabular-nums text-[13px] text-[#e6e9ef]">{formatProbability(value)}</span>
      {withBar && (
        <span className="h-1 w-12 overflow-hidden rounded-full bg-ink-700" aria-hidden>
          <span className="block h-full rounded-full bg-muted-dim"
                style={{ width: `${pct * 100}%` }} />
        </span>
      )}
    </span>
  );
}

/* ------------------------------------------------------------------ shell */

export function Card({ title, subtitle, actions, children, className = "" }: {
  title?: ReactNode; subtitle?: ReactNode; actions?: ReactNode;
  children: ReactNode; className?: string;
}) {
  return (
    <section className={`surface ${className}`}>
      {(title || actions) && (
        <header className="flex items-start justify-between gap-4 border-b border-ink-700 px-5 py-3.5">
          <div>
            {title && <h2 className="text-[13px] font-semibold text-[#e6e9ef]">{title}</h2>}
            {subtitle && <p className="mt-0.5 text-[12px] text-muted-dim">{subtitle}</p>}
          </div>
          {actions}
        </header>
      )}
      <div className="p-5">{children}</div>
    </section>
  );
}

export function Metric({ label, value, hint, tone = "neutral", size = "md" }: {
  label: string; value: ReactNode; hint?: string;
  tone?: "neutral" | "pos" | "neg" | "accent"; size?: "md" | "lg";
}) {
  const tones = { neutral: "text-[#e6e9ef]", pos: "text-pos", neg: "text-neg", accent: "text-accent" };
  return (
    <div>
      <div className="label">{label}</div>
      <div className={`mt-1.5 font-semibold tabular-nums ${tones[tone]} ${
        size === "lg" ? "text-metric-lg" : "text-metric"}`}>{value}</div>
      {hint && <div className="mt-1 text-[12px] text-muted-dim">{hint}</div>}
    </div>
  );
}

export function Mono({ children, copyable = false }: { children: string; copyable?: boolean }) {
  const [copied, setCopied] = useState(false);
  if (!copyable) return <span className="mono text-muted">{children}</span>;
  return (
    <button
      onClick={() => { navigator.clipboard?.writeText(children); setCopied(true);
                       setTimeout(() => setCopied(false), 1200); }}
      className="group inline-flex items-center gap-1.5 rounded px-1 py-0.5 mono text-muted hover:bg-ink-800 hover:text-[#e6e9ef]"
      title="Copy"
    >
      {children}
      {copied ? <Check size={11} className="text-pos" /> :
        <Copy size={11} className="opacity-0 group-hover:opacity-60" />}
    </button>
  );
}

export function Tip({ text }: { text: string }) {
  return (
    <span className="group relative inline-flex align-middle">
      <Info size={12} className="text-muted-dim hover:text-muted" />
      <span role="tooltip" className="pointer-events-none absolute bottom-full left-1/2 z-30 mb-2 w-60
        -translate-x-1/2 rounded-lg border border-ink-600 bg-ink-850 p-2.5 text-[11px] leading-relaxed
        text-muted opacity-0 transition-opacity group-hover:opacity-100">
        {text}
      </span>
    </span>
  );
}

export function EmptyState({ title, hint, icon }: { title: string; hint?: string; icon?: ReactNode }) {
  return (
    <div className="flex flex-col items-center justify-center gap-2 py-12 text-center">
      {icon && <div className="text-muted-dim">{icon}</div>}
      <p className="text-[13px] font-medium text-muted">{title}</p>
      {hint && <p className="max-w-sm text-[12px] text-muted-dim">{hint}</p>}
    </div>
  );
}

export function ErrorState({ message, onRetry }: { message: string; onRetry?: () => void }) {
  return (
    <div className="surface flex flex-col items-center gap-3 p-10 text-center">
      <X size={18} className="text-neg" />
      <p className="text-[13px] font-medium text-[#e6e9ef]">{message}</p>
      <p className="text-[12px] text-muted-dim">
        Confirm the backend is running and NEXT_PUBLIC_API_BASE_URL is correct.
      </p>
      {onRetry && (
        <button onClick={onRetry}
          className="mt-1 rounded-md border border-ink-600 px-3 py-1.5 text-[12px] text-muted hover:bg-ink-800">
          Retry
        </button>
      )}
    </div>
  );
}

export function Skeleton({ className = "" }: { className?: string }) {
  return <div className={`animate-pulse-soft rounded bg-ink-800 ${className}`} />;
}

export function TestModeBadge() {
  return (
    <span className="inline-flex items-center gap-1.5 rounded-md border border-warn/30 bg-warn-soft
      px-2 py-0.5 text-[10px] font-semibold uppercase tracking-[0.08em] text-warn">
      Razorpay Test Mode
    </span>
  );
}

export function Button({ children, onClick, variant = "default", disabled, loading, className = "" }: {
  children: ReactNode; onClick?: () => void;
  variant?: "default" | "primary" | "danger"; disabled?: boolean;
  loading?: boolean; className?: string;
}) {
  const variants = {
    default: "border-ink-600 bg-ink-800 text-[#e6e9ef] hover:bg-ink-700",
    primary: "border-accent/40 bg-accent/15 text-accent hover:bg-accent/25",
    danger: "border-neg/40 bg-neg-soft text-neg hover:bg-neg/20",
  };
  return (
    <button onClick={onClick} disabled={disabled || loading}
      className={`inline-flex items-center gap-2 rounded-md border px-3 py-1.5 text-[12px] font-medium
        transition-colors disabled:cursor-not-allowed disabled:opacity-40 ${variants[variant]} ${className}`}>
      {loading && <span className="h-3 w-3 animate-spin rounded-full border border-current border-t-transparent" />}
      {children}
    </button>
  );
}