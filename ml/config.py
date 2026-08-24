"""Central configuration for the synthetic behavioural environment.

Every magic number in the simulator lives here so that `docs/simulator.md` and
`docs/data-card.md` can be generated from a single authoritative source, and so
that scale can be changed without touching generation logic.

Scale rationale (spec Section 20, revised)
------------------------------------------
The headline evaluation is a doubly robust off-policy estimate computed on the
newest 15% of opportunities, restricted to the randomised exploration cohort.
At 3,000 opportunities that leaves roughly 70-90 exploration events spread over
9 actions, which is far too thin to bootstrap. Targets below are sized so the
held-out exploration cohort supports a usable effective sample size.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict

SIMULATOR_VERSION = "1.1.0"
LOGGING_POLICY_VERSION = "1.1.0"


@dataclass(frozen=True)
class SimulationConfig:
    seed: int = 42

    # ---- scale -----------------------------------------------------------
    n_products: int = 60
    n_customers: int = 8_000
    n_sessions: int = 120_000
    days: int = 180
    start_date: str = "2025-06-01"

    # ---- funnel ----------------------------------------------------------
    # Tuned to land near: 30k checkouts, ~12k recovery opportunities.
    p_checkout_start: float = 0.50
    p_abandon_before_payment: float = 0.42
    p_payment_failure: float = 0.115

    # ---- logging policy --------------------------------------------------
    # ~25%, not the 15-20% originally sketched. Kept deliberately: the headline
    # doubly robust estimate is computed on the held-out fold only, and 15%
    # left per-action support too thin to bootstrap. Documented in the data card.
    exploration_rate: float = 0.25
    min_propensity: float = 0.02

    # ---- splits ----------------------------------------------------------
    train_frac: float = 0.70
    validation_frac: float = 0.15
    # test_frac is the remainder

    # ---- hidden environmental mechanisms (Section 31) --------------------
    n_bank_outages: int = 6
    bank_outage_hours: int = 9
    n_competitor_sales: int = 4
    competitor_sale_days: int = 3
    payday_days: tuple[int, ...] = (1, 2, 3, 28, 29, 30)
    n_courier_disruptions: int = 3
    courier_disruption_days: int = 4

    # ---- outcome noise ---------------------------------------------------
    # Logit-space noise on the true response surface. Without this the model
    # can recover the generating process almost exactly and metrics become
    # unrealistically good (spec Section 31).
    response_logit_noise_sd: float = 0.45

    def as_dict(self) -> dict:
        d = asdict(self)
        d["simulator_version"] = SIMULATOR_VERSION
        d["logging_policy_version"] = LOGGING_POLICY_VERSION
        return d


DEFAULT_CONFIG = SimulationConfig()

# Columns the ML layer must NEVER see (spec Section 30). Enforced by
# tests/test_leakage.py against the generated customer table.
FORBIDDEN_FEATURE_PREFIXES: tuple[str, ...] = ("hidden_",)

FORBIDDEN_FEATURE_COLUMNS: tuple[str, ...] = (
    "true_recovery_probability",
    "oracle_",
    "converted_after_intervention",
    "recovered_revenue",
    "recovered_contribution",
)
