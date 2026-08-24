"""Dataset generator entrypoint.

    python -m ml.simulation.generate --seed 42

Produces, under `data/generated/`:

    products.parquet
    customers.parquet              observable only  (safe for the model)
    customers_hidden.parquet       latent traits    (simulator only)
    sessions.parquet
    checkouts.parquet
    payment_attempts.parquet
    opportunities.parquet          + chronological split label
    interventions.parquet          logged action, propensity, outcome
    oracle.parquet                 counterfactual P(recovery) for EVERY action
    manifest.json                  seed + versions + counts

`oracle.parquet` is quarantined by name and by the leakage tests: it exists for
Evaluation Stream C only and must never enter a feature matrix.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ml.actions import Action, eligible_actions
from ml.config import DEFAULT_CONFIG, LOGGING_POLICY_VERSION, SIMULATOR_VERSION, SimulationConfig
from ml.financial_engine import OpportunityEconomics, incentive_cost, fixed_action_cost
from ml.simulation.behavior import response_surface
from ml.simulation.customers import generate_customers
from ml.simulation.environment import build_environment
from ml.simulation.logging_policy import assign_actions
from ml.simulation.products import generate_products
from ml.simulation.sessions import generate_sessions

OUT_DIR = Path("data/generated")


def build_opportunities(checkouts: pd.DataFrame, payments: pd.DataFrame,
                        rng: np.random.Generator) -> pd.DataFrame:
    """Abandoned checkouts and failed payments both become opportunities."""
    ab = checkouts[checkouts["abandoned"]].copy()
    ab["opportunity_type"] = "CHECKOUT_ABANDONMENT"
    ab["detected_at"] = ab["abandoned_at"] + pd.to_timedelta(rng.integers(1, 15, size=len(ab)), "m")
    ab["failure_reason"] = "NONE"
    ab["payment_method"] = "NONE"
    ab["attempt_number"] = 1

    failed = payments[payments["status"] == "FAILED"].copy()
    fa = failed.merge(checkouts, on=["checkout_id", "customer_id"], how="left")
    fa["opportunity_type"] = "PAYMENT_FAILURE"
    fa["detected_at"] = fa["timestamp"] + pd.to_timedelta(rng.integers(1, 8, size=len(fa)), "m")
    fa["attempt_number"] = fa["retry_number"] + 1

    cols = [
        "checkout_id", "customer_id", "cart_value", "cart_cogs", "shipping_cost",
        "shipping_fee_charged", "base_contribution_margin", "product_return_rate",
        "opportunity_type", "detected_at", "failure_reason", "payment_method",
        "attempt_number", "abandonment_stage", "device_type", "traffic_source",
        "network_context", "hour_of_day", "day_of_week", "coupon_attempted",
        "checkout_started_at",
    ]
    for c in cols:
        if c not in ab.columns:
            ab[c] = None
        if c not in fa.columns:
            fa[c] = None

    opp = pd.concat([ab[cols], fa[cols]], ignore_index=True)
    opp = opp.sort_values("detected_at").reset_index(drop=True)
    opp.insert(0, "opportunity_id", [f"OPP{i + 1:07d}" for i in range(len(opp))])

    opp["revenue_at_risk"] = opp["cart_value"]
    opp["contribution_margin_at_risk"] = opp["base_contribution_margin"]
    opp["minutes_since_event"] = np.round(
        (opp["detected_at"] - pd.to_datetime(opp["checkout_started_at"])).dt.total_seconds() / 60.0, 2
    )
    return opp


def chronological_split(opp: pd.DataFrame, cfg: SimulationConfig) -> pd.Series:
    """Time-aware split (Section 38). Never random."""
    n = len(opp)
    i_train = int(n * cfg.train_frac)
    i_val = int(n * (cfg.train_frac + cfg.validation_frac))
    split = np.array(["TEST"] * n, dtype=object)
    split[:i_train] = "TRAIN"
    split[i_train:i_val] = "VALIDATION"
    return pd.Series(split, index=opp.index)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate the RevenueOS synthetic dataset.")
    ap.add_argument("--seed", type=int, default=DEFAULT_CONFIG.seed)
    ap.add_argument("--customers", type=int, default=DEFAULT_CONFIG.n_customers)
    ap.add_argument("--sessions", type=int, default=DEFAULT_CONFIG.n_sessions)
    ap.add_argument("--out", type=str, default=str(OUT_DIR))
    args = ap.parse_args()

    cfg = SimulationConfig(seed=args.seed, n_customers=args.customers, n_sessions=args.sessions)
    rng = np.random.default_rng(cfg.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[1/7] environment (seed={cfg.seed})")
    env = build_environment(cfg, rng)

    print("[2/7] products + customers")
    products = generate_products(cfg.n_products, rng)
    customers, customers_hidden = generate_customers(cfg.n_customers, rng, cfg.start_date)

    print("[3/7] sessions -> checkouts -> payments")
    sessions, checkouts, payments = generate_sessions(
        cfg, rng, customers, customers_hidden, products, env
    )

    print("[4/7] recovery opportunities")
    opp = build_opportunities(checkouts, payments, rng)
    opp["split"] = chronological_split(opp, cfg)

    print("[5/7] counterfactual response surface")
    hid = customers_hidden.set_index("customer_id").loc[opp["customer_id"]].reset_index()
    ctx = {
        "cart_value": opp["cart_value"].astype(float).values,
        "shipping_fee_charged": opp["shipping_fee_charged"].astype(float).values,
        "failure_reason": opp["failure_reason"].astype(str).values,
        "opportunity_type": opp["opportunity_type"].astype(str).values,
        "minutes_since_event": opp["minutes_since_event"].astype(float).values,
        "attempt_number": opp["attempt_number"].astype(float).values,
        "in_bank_outage": env.in_bank_outage(opp["detected_at"]).astype(float),
        "in_competitor_sale": env.in_competitor_sale(opp["detected_at"]).astype(float),
        "is_payday": env.is_payday(opp["detected_at"]).astype(float),
        **{c: hid[c].values for c in hid.columns if c.startswith("hidden_")},
    }

    all_actions = eligible_actions("PAYMENT_FAILURE")  # superset
    surface = response_surface(ctx, all_actions, rng, cfg.response_logit_noise_sd)

    # Mask actions that are not eligible for abandonment opportunities.
    is_abandon = ctx["opportunity_type"] == "CHECKOUT_ABANDONMENT"
    oracle = pd.DataFrame({"opportunity_id": opp["opportunity_id"]})
    for a in all_actions:
        p = surface[a].copy()
        if a in (Action.IMMEDIATE_RETRY, Action.DELAYED_RETRY):
            p = np.where(is_abandon, np.nan, p)
        oracle[f"p_recovery__{a.value}"] = np.round(p, 6)

    print("[6/7] logging policy + outcomes")
    logged = assign_actions(ctx, rng, cfg.exploration_rate, cfg.min_propensity)

    taken_p = np.array([
        surface[Action(a)][i] for i, a in enumerate(logged["action_taken"])
    ])
    converted = rng.random(len(opp)) < taken_p

    # Realised economics of the logged action, via the canonical engine.
    inc_cost, fix_cost, rec_rev, rec_contrib = [], [], [], []
    for i, a in enumerate(logged["action_taken"]):
        econ = OpportunityEconomics(
            cart_value=float(opp["cart_value"].iloc[i]),
            cogs=float(opp["cart_cogs"].iloc[i]),
            shipping_cost=float(opp["shipping_cost"].iloc[i]),
            shipping_fee_charged=float(opp["shipping_fee_charged"].iloc[i]),
            return_probability=float(hid["hidden_return_propensity"].iloc[i]),
            cancellation_probability=float(hid["hidden_cancellation_propensity"].iloc[i]),
        )
        ic = incentive_cost(econ, a)
        fc = fixed_action_cost(a)
        inc_cost.append(ic)
        fix_cost.append(fc)
        if converted[i]:
            rec_rev.append(econ.cart_value - ic if "DISCOUNT" in a else econ.cart_value)
            rec_contrib.append(econ.base_contribution_margin - ic - fc)
        else:
            rec_rev.append(0.0)
            rec_contrib.append(-fc)

    interventions = pd.DataFrame({
        "intervention_id": [f"IV{i + 1:07d}" for i in range(len(opp))],
        "opportunity_id": opp["opportunity_id"].values,
        "decision_timestamp": opp["detected_at"].values,
        "action_taken": logged["action_taken"].values,
        "action_propensity": np.round(logged["action_propensity"].values, 6),
        "is_exploration": logged["is_exploration"].values,
        "propensity_vector": [json.dumps(v) for v in logged["propensity_vector"]],
        "fixed_action_cost": np.round(fix_cost, 2),
        "incentive_cost_if_recovered": np.round(inc_cost, 2),
        "converted_after_intervention": converted,
        "recovered_revenue": np.round(rec_rev, 2),
        "recovered_contribution": np.round(rec_contrib, 2),
        "historical_policy_version": LOGGING_POLICY_VERSION,
        "split": opp["split"].values,
    })

    print("[7/7] writing")
    frames = {
        "products": products, "customers": customers, "customers_hidden": customers_hidden,
        "sessions": sessions, "checkouts": checkouts, "payment_attempts": payments,
        "opportunities": opp, "interventions": interventions, "oracle": oracle,
    }
    for name, df in frames.items():
        df.to_parquet(out / f"{name}.parquet", index=False)

    manifest = {
        "seed": cfg.seed,
        "simulator_version": SIMULATOR_VERSION,
        "logging_policy_version": LOGGING_POLICY_VERSION,
        "config": cfg.as_dict(),
        "counts": {k: int(len(v)) for k, v in frames.items()},
        "hidden_environment": env.summary(),
        "dataset_period": [str(opp["detected_at"].min()), str(opp["detected_at"].max())],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))

    print("\nGenerated:")
    for k, v in manifest["counts"].items():
        print(f"  {k:<20} {v:>9,}")
    print(f"\nWrote {out}/ (seed={cfg.seed})")


if __name__ == "__main__":
    main()
