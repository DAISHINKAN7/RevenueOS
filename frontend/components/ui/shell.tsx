"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import {
  Activity, ClipboardList, Handshake, LayoutDashboard, ScrollText, Settings,
  ShieldCheck,
} from "lucide-react";
import { api } from "@/lib/api";
import type { HealthStatus } from "@/lib/types";

const NAV = [
  { href: "/", label: "Overview", icon: LayoutDashboard },
  { href: "/opportunities", label: "Opportunities", icon: ClipboardList },
  { href: "/evaluation", label: "Evaluation", icon: Activity },
  { href: "/agent", label: "Agent", icon: ShieldCheck },
  { href: "/commerce", label: "Agentic Commerce", icon: Handshake },
  { href: "/audit", label: "Audit", icon: ScrollText },
  { href: "/settings", label: "Settings", icon: Settings },
];

function HealthPill() {
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [offline, setOffline] = useState(false);

  useEffect(() => {
    let alive = true;
    const load = () =>
      api.health()
        .then((h) => { if (alive) { setHealth(h); setOffline(false); } })
        .catch(() => { if (alive) setOffline(true); });
    load();
    const t = setInterval(load, 15000);
    return () => { alive = false; clearInterval(t); };
  }, []);

  if (offline)
    return (
      <span className="inline-flex items-center gap-2 text-[11px] text-neg">
        <span className="h-1.5 w-1.5 rounded-full bg-neg" /> Backend unavailable
      </span>
    );
  if (!health) return <span className="text-[11px] text-muted-dim">Checking…</span>;

  const ok = health.backend_status === "ok";
  return (
    <div className="flex items-center gap-4 text-[11px]">
      <span className="inline-flex items-center gap-2 text-muted">
        <span className={`h-1.5 w-1.5 rounded-full ${ok ? "bg-pos" : "bg-warn"}`} />
        {ok ? "System healthy" : "Degraded"}
      </span>
      <span className="text-muted-dim">
        Model {health.model_loaded ? "loaded" : "unavailable"}
      </span>
      <span className="rounded border border-warn/30 bg-warn-soft px-1.5 py-0.5 font-semibold
        uppercase tracking-[0.06em] text-warn">
        {health.payment_environment} mode
      </span>
    </div>
  );
}

export function Shell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  return (
    <div className="flex min-h-screen">
      <aside className="hidden w-56 shrink-0 flex-col border-r border-ink-700 bg-ink-900 lg:flex">
        <div className="border-b border-ink-700 px-5 py-4">
          <div className="text-[14px] font-semibold tracking-tight text-[#e6e9ef]">RevenueOS</div>
          <div className="mt-0.5 text-[11px] text-muted-dim">Autonomous Revenue Recovery</div>
        </div>
        <nav className="flex-1 p-3">
          {NAV.map(({ href, label, icon: Icon }) => {
            const active = href === "/" ? pathname === "/" : pathname.startsWith(href);
            return (
              <Link key={href} href={href}
                className={`mb-0.5 flex items-center gap-2.5 rounded-md px-3 py-2 text-[13px] transition-colors ${
                  active ? "bg-ink-800 font-medium text-[#e6e9ef]" : "text-muted hover:bg-ink-850 hover:text-[#e6e9ef]"}`}>
                <Icon size={15} className={active ? "text-accent" : ""} />
                {label}
              </Link>
            );
          })}
        </nav>
        <div className="border-t border-ink-700 px-5 py-3 text-[10px] leading-relaxed text-muted-dim">
          Synthetic merchant · Razorpay Test Mode.
          <br />No production money moves.
        </div>
      </aside>

      <div className="flex min-w-0 flex-1 flex-col">
        <header className="flex h-14 shrink-0 items-center justify-between border-b border-ink-700
          bg-ink-950/80 px-6 backdrop-blur">
          <nav className="flex gap-1 lg:hidden">
            {NAV.slice(0, 4).map(({ href, label }) => (
              <Link key={href} href={href}
                className="rounded px-2 py-1 text-[12px] text-muted hover:text-[#e6e9ef]">{label}</Link>
            ))}
          </nav>
          <div className="hidden lg:block" />
          <HealthPill />
        </header>
        <main className="flex-1 overflow-x-hidden px-6 py-6">{children}</main>
      </div>
    </div>
  );
}

export function PageHeader({ title, subtitle, actions }: {
  title: string; subtitle?: string; actions?: React.ReactNode;
}) {
  return (
    <div className="mb-6 flex items-start justify-between gap-6">
      <div>
        <h1 className="text-[19px] font-semibold tracking-tight text-[#e6e9ef]">{title}</h1>
        {subtitle && <p className="mt-1 max-w-2xl text-[13px] leading-relaxed text-muted">{subtitle}</p>}
      </div>
      {actions && <div className="flex shrink-0 items-center gap-2">{actions}</div>}
    </div>
  );
}