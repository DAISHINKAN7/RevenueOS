/**
 * Shared formatters. Every number the user sees passes through here so
 * rounding is consistent across the whole product; IDs are never reformatted.
 */

const INR = new Intl.NumberFormat("en-IN", {
  style: "currency", currency: "INR", maximumFractionDigits: 2,
});
const INR0 = new Intl.NumberFormat("en-IN", {
  style: "currency", currency: "INR", maximumFractionDigits: 0,
});

export function formatINR(value: number | null | undefined, compact = false): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  if (compact && Math.abs(value) >= 100000) return `₹${(value / 100000).toFixed(2)}L`;
  return (compact ? INR0 : INR).format(value);
}

export function formatDeltaEV(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  const sign = value > 0 ? "+" : value < 0 ? "−" : "";
  return `${sign}${INR.format(Math.abs(value))}`;
}

/** Probability is presented as a percentage but never coloured. */
export function formatProbability(p: number | null | undefined, digits = 1): string {
  if (p === null || p === undefined || Number.isNaN(p)) return "—";
  return `${(p * 100).toFixed(digits)}%`;
}

export function formatPercent(p: number | null | undefined, digits = 1): string {
  if (p === null || p === undefined || Number.isNaN(p)) return "—";
  return `${(p * 100).toFixed(digits)}%`;
}

export function formatNumber(n: number | null | undefined): string {
  if (n === null || n === undefined || Number.isNaN(n)) return "—";
  return new Intl.NumberFormat("en-IN").format(n);
}

export function formatTimestamp(iso: string | null | undefined): string {
  if (!iso) return "—";
  const d = new Date(iso.endsWith("Z") || iso.includes("+") ? iso : `${iso}Z`);
  return d.toLocaleString("en-IN", {
    day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit",
    second: "2-digit", hour12: false,
  });
}

export function formatClock(iso: string | null | undefined): string {
  if (!iso) return "—";
  const d = new Date(iso.endsWith("Z") || iso.includes("+") ? iso : `${iso}Z`);
  return d.toLocaleTimeString("en-IN", {
    hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false,
  });
}

export function formatAge(iso: string | null | undefined): string {
  if (!iso) return "—";
  const d = new Date(iso.endsWith("Z") || iso.includes("+") ? iso : `${iso}Z`);
  const mins = Math.max(0, Math.round((Date.now() - d.getTime()) / 60000));
  if (mins < 60) return `${mins}m`;
  if (mins < 1440) return `${Math.round(mins / 60)}h`;
  return `${Math.round(mins / 1440)}d`;
}

export function titleCase(s: string): string {
  return s.replace(/_/g, " ").toLowerCase().replace(/\b\w/g, (c) => c.toUpperCase());
}