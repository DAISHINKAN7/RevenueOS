/**
 * Typed API client. Every backend call goes through here so the base URL and
 * admin token are configured in exactly one place, and so a failed request
 * surfaces a usable message instead of an unhandled rejection.
 */

import type {
  AgentSummary, DashboardMetrics, EvaluationSummary, HealthStatus,
  MerchantPolicyConfig, OpportunityDetail, OpportunityListItem,
} from "./types";

export const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";
const ADMIN_TOKEN = process.env.NEXT_PUBLIC_ADMIN_TOKEN ?? "";

export class ApiError extends Error {
  constructor(message: string, readonly status: number, readonly code?: string) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE}${path}`, {
      ...init,
      cache: "no-store",
      headers: {
        "Content-Type": "application/json",
        ...(init?.method && init.method !== "GET"
          ? { "X-Admin-Token": ADMIN_TOKEN }
          : {}),
        ...init?.headers,
      },
    });
  } catch {
    throw new ApiError("RevenueOS backend unavailable", 0, "OFFLINE");
  }
  if (!res.ok) {
    let code: string | undefined;
    let message = `Request failed (${res.status})`;
    try {
      const body = await res.json();
      const detail = body?.detail ?? body;
      code = detail?.error_code;
      message = detail?.message || code || message;
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(message, res.status, code);
  }
  return res.json() as Promise<T>;
}

export const api = {
  health: () => request<HealthStatus>("/health"),
  version: () => request<Record<string, string>>("/api/version"),

  opportunities: (state?: string) =>
    request<OpportunityListItem[]>(
      `/api/opportunities${state ? `?state=${encodeURIComponent(state)}` : ""}`,
    ),
  opportunity: (id: string) =>
    request<OpportunityDetail>(`/api/opportunities/${id}`),

  analyze: (id: string) =>
    request<Record<string, unknown>>(`/api/opportunities/${id}/analyze`, {
      method: "POST",
    }),
  approve: (id: string, actor = "demo-operator") =>
    request<Record<string, unknown>>(`/api/opportunities/${id}/approve`, {
      method: "POST", body: JSON.stringify({ actor_id: actor }),
    }),
  reject: (id: string, actor = "demo-operator") =>
    request<Record<string, unknown>>(`/api/opportunities/${id}/reject`, {
      method: "POST", body: JSON.stringify({ actor_id: actor }),
    }),
  execute: (id: string) =>
    request<{
      execution_id: string; status: string; state: string;
      external_order_id: string | null;
      checkout?: {
        razorpay_key_id: string; razorpay_order_id: string;
        amount_paise: number; currency: string; display_name: string;
        payment_environment: string;
      };
    }>(`/api/opportunities/${id}/execute`, { method: "POST" }),

  /** Offline demo only. Clearly distinguished from a verified provider event. */
  simulatePayment: (id: string, outcome: "success" | "failure",
                    failureMode = "card_declined") =>
    request<{
      status: string; state: string; simulated: true;
      net_recovered_gmv?: number | null; realized_contribution?: number | null;
      normalized_reason?: string; category?: string; failure_code?: string;
    }>(`/api/opportunities/${id}/simulate-payment`, {
      method: "POST",
      body: JSON.stringify({ outcome, failure_mode: failureMode }),
    }),
  failureModes: () =>
    request<{ modes: { key: string; description: string; step: string }[] }>(
      "/api/simulation/failure-modes"),

  dashboard: () => request<DashboardMetrics>("/api/dashboard/metrics"),
  evaluation: () => request<EvaluationSummary>("/api/evaluation/summary"),
  agent: () => request<AgentSummary>("/api/agent/summary"),
  agentRun: (id: string) =>
    request<{
      agent_run_id: string; opportunity_id: string; disposition: string;
      planner_source: string; llm_model: string | null;
      events: {
        sequence: number; timestamp: string; event_type: string;
        tool_name: string | null; reasoning: string | null;
        workflow_state: string | null; tool_output: string | null;
      }[];
    }>(`/api/agent/runs/${id}`),
  policy: () => request<MerchantPolicyConfig>("/api/policy"),
};