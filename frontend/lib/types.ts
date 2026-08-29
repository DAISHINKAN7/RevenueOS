/** Strict types mirroring the backend response shapes. */

export type WorkflowState =
  | "DETECTED" | "ANALYZING" | "CANDIDATES_SCORED" | "ECONOMICALLY_RANKED"
  | "POLICY_CHECKED" | "AWAITING_APPROVAL" | "AUTHORIZED" | "EXECUTION_PENDING"
  | "EXECUTING" | "AWAITING_PAYMENT" | "PAYMENT_FAILED_RECOVERABLE"
  | "RECOVERED" | "NOT_RECOVERED" | "STOPPED" | "ESCALATED" | "EXPIRED"
  | "EXECUTION_FAILED";

export type PolicyStatus = "PASS" | "REJECT" | "REQUIRE_APPROVAL" | "STOP";

export interface OpportunityListItem {
  opportunity_id: string;
  opportunity_type: string;
  state: WorkflowState;
  revenue_at_risk: number;
  contribution_margin_at_risk: number;
  selected_action: string | null;
  attempt: number;
  execution_mode: string;
  detected_at: string;
  customer_segment: string | null;
}

export interface CandidateAction {
  action: string;
  rank: number | null;
  probability: number;
  incentive_cost: number;
  fixed_cost: number;
  expected_value: number;
  incremental_expected_value: number;
}

export interface PolicyRule {
  rule_id: string;
  passed: boolean;
  decision: string;
  reason: string;
  input: string | null;
  threshold: string | null;
}

export interface PolicyDecision {
  decision: PolicyStatus;
  reason_code: string;
  policy_version: string;
  maximum_authorized_downside: number;
  rules: PolicyRule[];
}

export interface ExecutionRecord {
  execution_id: string;
  action: string;
  status: string;
  attempt: number;
  provider: string;
  order_id: string | null;
  payment_id: string | null;
  amount: number | null;
  error_code: string | null;
}

export interface AuditEvent {
  sequence: number;
  timestamp: string;
  event_type: string;
  summary: string;
  state_before: string | null;
  state_after: string | null;
  actor: string;
  payload: Record<string, unknown>;
  execution_id: string | null;
}

export interface OpportunityDetail {
  opportunity: {
    opportunity_id: string; type: string; state: WorkflowState;
    revenue_at_risk: number; contribution_margin_at_risk: number;
    attempt: number; execution_mode: string; trace_id: string; detected_at: string;
  };
  customer_summary: Record<string, unknown>;
  checkout_summary: Record<string, number | null>;
  payment_summary: { failure_reason: string | null; payment_method: string | null };
  candidate_actions: CandidateAction[];
  selected_action: string | null;
  policy_decision: PolicyDecision | null;
  execution: ExecutionRecord[];
  outcome: {
    net_recovered_gmv: number; realized_contribution: number;
    discount_amount: number; recovered_at: string;
  } | null;
  audit_timeline: AuditEvent[];
}

export interface DashboardMetrics {
  metric_class: string;
  note: string;
  opportunities: number;
  revenue_at_risk: number;
  contribution_margin_at_risk: number;
  recovered_gmv: number;
  net_contribution_recovered: number;
  intervention_cost: number;
  recovery_rate: number;
  number_of_policy_blocks: number;
  number_of_stops: number;
  number_of_approval_cases: number;
  number_of_do_nothing_decisions: number;
}

export interface EvaluationSummary {
  metric_class: string;
  available: boolean;
  model?: { roc_auc: number; pr_auc: number; brier: number; ece: number };
  divergence?: number;
  do_nothing_rate?: number;
  do_nothing_precision?: number;
  mean_regret?: number;
  headline?: Record<string, string>;
  policies?: {
    policy: string; conversion: number; net_gmv_per_opp: number;
    incentive_cost_per_opp: number; net_contribution_per_opp: number;
  }[];
  data?: {
    train_rows: number; validation_rows: number; test_rows: number;
    simulator_version: string; oracle_access_policy: string;
  };
}

export interface AgentSummary {
  agent_version: string;
  authorizer_version: string;
  planner: { enabled: boolean; provider: string; model: string; active: boolean; max_steps: number };
  metrics: Record<string, number>;
  tools: { tool: string; class: string }[];
  state_matrix: { state: string; terminal: boolean; tools: string[] }[];
  runs: {
    agent_run_id: string; opportunity_id: string; disposition: string;
    planner_source: string; tool_calls: number; blocked: number; replans: number;
    initial_state: string; final_state: string; started_at: string | null;
  }[];
}

export interface HealthStatus {
  backend_status: string; database_status: string; model_loaded: boolean;
  model_version: string; policy_version: string; razorpay_mode: string;
  payment_environment: string; autonomous_execution_enabled: boolean;
}

export interface MerchantPolicyConfig { [key: string]: string | number | boolean; }