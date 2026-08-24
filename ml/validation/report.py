"""Data Validation Report (spec Section 92, Task 12 / Section 93).

    python -m ml.validation.report

This is the review gate before any model training. It answers three questions:

1. Does the dataset have the scale the evaluation design requires?
2. Does it contain learnable structure (do actions differ by segment)?
3. Is it too easy, leaky, or too thin to support off-policy estimation?

Writes `evaluation/results/data_validation_report.md` and prints a verdict.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ml.actions import ALL_ACTIONS, Action, eligible_actions
from ml.config import FORBIDDEN_FEATURE_PREFIXES
from ml.financial_engine import (
    OpportunityEconomics, incentive_cost, fixed_action_cost, rank_actions,
)

DATA = Path("data/generated")
OUT = Path("evaluation/results/data_validation_report.md")


def _fmt(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False, floatfmt=".4f")


def load() -> dict[str, pd.DataFrame]:
    frames = {}
    for name in ["products", "customers", "customers_hidden", "sessions", "checkouts",
                 "payment_attempts", "opportunities", "interventions", "oracle"]:
        frames[name] = pd.read_parquet(DATA / f"{name}.parquet")
    return frames


def effective_sample_size(weights: np.ndarray) -> float:
    """Kish ESS. The number that decides whether OPE is usable at all."""
    w = np.asarray(weights, dtype=float)
    if w.sum() == 0:
        return 0.0
    return float(w.sum() ** 2 / np.square(w).sum())


def _econ_for(row) -> OpportunityEconomics:
    return OpportunityEconomics(
        cart_value=float(row.cart_value),
        cogs=float(row.cart_cogs),
        shipping_cost=float(row.shipping_cost),
        shipping_fee_charged=float(row.shipping_fee_charged),
    )


def oracle_economics(opp: pd.DataFrame, oracle: pd.DataFrame, sample: int = 4000,
                     seed: int = 0) -> pd.DataFrame:
    """Run the canonical financial engine over the oracle response surface.

    Produces, per opportunity: the probability-maximising action, the
    dEV-maximising action, whether all interventions have dEV <= 0, and the
    realised economics of DO_NOTHING / flat 10% / oracle-economic action.
    """
    d = opp.merge(oracle, on="opportunity_id")
    if len(d) > sample:
        d = d.sample(sample, random_state=seed)

    rows = []
    for r in d.itertuples():
        econ = _econ_for(r)
        acts = eligible_actions(r.opportunity_type)
        probs = {}
        for a in acts:
            v = getattr(r, f"p_recovery__{a.value}")
            if v is not None and not np.isnan(v):
                probs[a] = float(v)

        ranked = rank_actions(econ, probs, r.opportunity_type)
        by_prob = max(ranked, key=lambda v: v.recovery_probability)
        interventions = [v for v in ranked if v.action is not Action.DO_NOTHING]
        best_interv = max(interventions, key=lambda v: v.delta_ev)
        all_non_positive = best_interv.delta_ev <= 0
        ev_best = (next(v for v in ranked if v.action is Action.DO_NOTHING)
                   if all_non_positive else best_interv)

        nothing = next(v for v in ranked if v.action is Action.DO_NOTHING)
        flat = next((v for v in ranked if v.action is Action.MEDIUM_DISCOUNT), None)

        rec = {
            "prob_max_action": by_prob.action.value,
            "ev_max_action": ev_best.action.value,
            "all_interventions_non_positive": all_non_positive,
            "differ": by_prob.action is not ev_best.action,
            # strategy-level expected economics
            "nothing_p": nothing.recovery_probability,
            "nothing_gmv": nothing.expected_recovered_gmv,
            "nothing_cost": 0.0,
            "nothing_contrib": nothing.expected_value,
            "ev_p": ev_best.recovery_probability,
            "ev_gmv": ev_best.expected_recovered_gmv,
            "ev_cost": ev_best.recovery_probability * ev_best.incentive_cost + ev_best.fixed_cost,
            "ev_contrib": ev_best.expected_value,
        }
        if flat is not None:
            rec.update({
                "flat_p": flat.recovery_probability,
                "flat_gmv": flat.expected_recovered_gmv,
                "flat_cost": flat.recovery_probability * flat.incentive_cost + flat.fixed_cost,
                "flat_contrib": flat.expected_value,
            })
        rows.append(rec)
    return pd.DataFrame(rows)


def build_report() -> str:
    f = load()
    manifest = json.loads((DATA / "manifest.json").read_text())
    opp, iv, oracle = f["opportunities"], f["interventions"], f["oracle"]
    lines: list[str] = []
    warnings: list[str] = []

    lines.append("# RevenueOS — Data Validation Report\n")
    lines.append(f"Seed `{manifest['seed']}` · simulator `{manifest['simulator_version']}` "
                 f"· logging policy `{manifest['logging_policy_version']}`\n")
    lines.append(f"Period: {manifest['dataset_period'][0]} → {manifest['dataset_period'][1]}\n")

    # ---- 1. record counts ------------------------------------------------
    lines.append("## 1. Record counts\n")
    counts = pd.DataFrame(
        [{"table": k, "rows": v} for k, v in manifest["counts"].items()]
    )
    lines.append(counts.to_markdown(index=False) + "\n")

    # ---- 2. funnel -------------------------------------------------------
    ck = f["checkouts"]
    pa = f["payment_attempts"]
    funnel = pd.DataFrame([
        {"stage": "sessions", "n": len(f["sessions"]), "rate_of_prior": np.nan},
        {"stage": "checkouts started", "n": len(ck), "rate_of_prior": len(ck) / len(f["sessions"])},
        {"stage": "abandoned", "n": int(ck["abandoned"].sum()), "rate_of_prior": ck["abandoned"].mean()},
        {"stage": "payment attempts", "n": len(pa), "rate_of_prior": len(pa) / max(len(ck), 1)},
        {"stage": "payment failures", "n": int((pa["status"] == "FAILED").sum()),
         "rate_of_prior": (pa["status"] == "FAILED").mean()},
        {"stage": "opportunities", "n": len(opp), "rate_of_prior": np.nan},
    ])
    lines.append("## 2. Funnel\n" + _fmt(funnel) + "\n")
    lines.append(
        "\n> **Interpretation.** These are not generic website conversion rates. The "
        "simulator represents commerce sessions that have already demonstrated "
        "substantial purchase intent — a high-intent cohort, not all anonymous "
        "merchant traffic. This oversampling is deliberate: it ensures adequate "
        "recovery-opportunity volume for policy learning and off-policy evaluation. "
        "Do not read `checkouts started / sessions` as a site-wide conversion rate.\n"
    )

    # ---- 3. splits + OPE support ----------------------------------------
    lines.append("## 3. Chronological split and off-policy support\n")
    split_tbl = (
        iv.groupby("split")
        .agg(opportunities=("intervention_id", "size"),
             exploration=("is_exploration", "sum"),
             exploration_rate=("is_exploration", "mean"))
        .reset_index()
    )
    lines.append(_fmt(split_tbl) + "\n")

    test = iv[iv["split"] == "TEST"]
    test_expl = test[test["is_exploration"]]
    per_action = test_expl["action_taken"].value_counts().reindex(
        [a.value for a in ALL_ACTIONS]).fillna(0).astype(int)
    lines.append("Held-out exploration cohort by action:\n\n"
                 + per_action.to_frame("n").to_markdown() + "\n")

    w = 1.0 / test["action_propensity"].values
    ess = effective_sample_size(w)
    lines.append(f"\nHeld-out rows: **{len(test):,}** · exploration rows: **{len(test_expl):,}**\n")
    lines.append(f"Kish ESS of uniform-target importance weights: **{ess:,.0f}** "
                 f"(max weight {w.max():.1f}, min propensity {test['action_propensity'].min():.4f})\n")

    if len(test_expl) < 250:
        warnings.append(
            f"Held-out exploration cohort is only {len(test_expl)} rows. Doubly robust "
            "estimates will have very wide bootstrap intervals. Increase --sessions."
        )
    if per_action.min() < 20:
        thin = per_action[per_action < 20].index.tolist()
        warnings.append(f"Thin held-out action support: {thin}. OPE for these actions is unreliable.")
    if w.max() > 200:
        warnings.append(f"Max importance weight {w.max():.0f} is large; consider weight clipping in SNIPS/DR.")

    # ---- 4. propensity distribution -------------------------------------
    lines.append("## 4. Propensity distribution\n")
    q = iv["action_propensity"].describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99])
    lines.append(q.to_frame("action_propensity").to_markdown(floatfmt=".4f") + "\n")
    if (iv["action_propensity"] <= 0).any():
        warnings.append("CRITICAL: zero propensity logged — importance weights would be infinite.")

    # ---- 5. logged action mix -------------------------------------------
    lines.append("## 5. Logged action distribution\n")
    mix = (iv.groupby("action_taken")
             .agg(n=("intervention_id", "size"),
                  share=("intervention_id", lambda s: len(s) / len(iv)),
                  conversion=("converted_after_intervention", "mean"),
                  mean_contribution=("recovered_contribution", "mean"))
             .reset_index().sort_values("n", ascending=False))
    lines.append(_fmt(mix) + "\n")
    lines.append(
        "\n> Naive conversion rates above are **confounded** with context — the historical "
        "policy chose actions based on cart value and failure reason. They are not causal "
        "estimates and must not be read as action effectiveness.\n"
    )

    # ---- 6. learnable structure -----------------------------------------
    lines.append("## 6. Structure check — oracle uplift by segment\n")
    cust = f["customers"][["customer_id", "customer_segment"]]
    o = oracle.merge(opp[["opportunity_id", "customer_id", "opportunity_type"]], on="opportunity_id")
    o = o.merge(cust, on="customer_id")
    base = o["p_recovery__DO_NOTHING"]
    rows = []
    for seg, g in o.groupby("customer_segment"):
        r = {"segment": seg, "n": len(g)}
        for a in ["FREE_SHIPPING", "SMALL_DISCOUNT", "MEDIUM_DISCOUNT",
                  "PAYMENT_METHOD_SWITCH", "DELAYED_RETRY"]:
            col = f"p_recovery__{a}"
            r[a] = float((g[col] - g["p_recovery__DO_NOTHING"]).mean())
        rows.append(r)
    seg_tbl = pd.DataFrame(rows).sort_values("segment")
    lines.append(_fmt(seg_tbl) + "\n")
    lines.append("\n> True ΔP(recovery) vs DO_NOTHING. Distinct rows here are the signal the "
                 "model is supposed to learn. Flat rows mean the simulator has no structure.\n")

    spread = seg_tbl[["FREE_SHIPPING", "SMALL_DISCOUNT", "PAYMENT_METHOD_SWITCH"]].values
    if np.nanstd(spread) < 0.02:
        warnings.append("Segment ΔP spread is very low — the response surface may be too flat to learn.")

    # ---- 7. dual oracle views -------------------------------------------
    act_cols = [c for c in oracle.columns if c.startswith("p_recovery__")]
    oe = oracle_economics(opp, oracle)

    prob_mix = oe["prob_max_action"].value_counts(normalize=True).to_frame("share")
    ev_mix = oe["ev_max_action"].value_counts(normalize=True).to_frame("share")

    lines.append("## 7a. Oracle best action — PROBABILITY (argmax P(recovery|a))\n")
    lines.append(prob_mix.to_markdown(floatfmt=".4f") + "\n")
    lines.append("\n## 7b. Oracle best action — ECONOMICS (argmax dEV, incl. DO_NOTHING)\n")
    lines.append(ev_mix.to_markdown(floatfmt=".4f") + "\n")
    lines.append(f"\nComputed on {len(oe):,} sampled opportunities via the canonical "
                 "financial engine.\n")

    if prob_mix["share"].max() > 0.80:
        warnings.append(
            f"One action is probability-optimal in {prob_mix['share'].max():.0%} of cases; "
            "the conversion decision problem may be degenerate."
        )
    if ev_mix["share"].max() > 0.70:
        warnings.append(
            f"One action is economics-optimal in {ev_mix['share'].max():.0%} of cases; "
            "check whether intervention costs or uplifts are unrealistic."
        )
    do_nothing_share = float(ev_mix["share"].get("DO_NOTHING", 0.0))
    if do_nothing_share < 0.02:
        warnings.append(
            "DO_NOTHING is almost never economics-optimal. Intervention uplift is probably "
            "too strong or intervention cost too cheap."
        )

    # ---- 8. intelligent restraint ---------------------------------------
    lines.append("## 8. Intelligent restraint\n")
    restraint = pd.DataFrame([{
        "metric": "share_of_opportunities_where_all_interventions_have_delta_ev_lte_zero",
        "value": float(oe["all_interventions_non_positive"].mean()),
    }, {
        "metric": "share_where_probability_max_action_differs_from_ev_max_action",
        "value": float(oe["differ"].mean()),
    }, {
        "metric": "share_where_DO_NOTHING_is_economics_optimal",
        "value": do_nothing_share,
    }])
    lines.append(_fmt(restraint) + "\n")
    lines.append(
        "\n> The second row is the headline RevenueOS metric: on that share of "
        "opportunities, maximising conversion and maximising contribution select "
        "**different actions**.\n"
    )

    # ---- 9. flat-discount sanity comparison ------------------------------
    lines.append("## 9. Flat-10%-discount sanity comparison\n")
    lines.append("Expected values under three fixed strategies, same opportunities:\n")
    strat = pd.DataFrame([
        {"strategy": "DO_NOTHING",
         "conversion": oe["nothing_p"].mean(),
         "recovered_gmv_net": oe["nothing_gmv"].mean(),
         "incentive_cost": oe["nothing_cost"].mean(),
         "net_contribution": oe["nothing_contrib"].mean()},
        {"strategy": "FLAT_10PCT_DISCOUNT",
         "conversion": oe["flat_p"].mean(),
         "recovered_gmv_net": oe["flat_gmv"].mean(),
         "incentive_cost": oe["flat_cost"].mean(),
         "net_contribution": oe["flat_contrib"].mean()},
        {"strategy": "ORACLE_ECONOMIC",
         "conversion": oe["ev_p"].mean(),
         "recovered_gmv_net": oe["ev_gmv"].mean(),
         "incentive_cost": oe["ev_cost"].mean(),
         "net_contribution": oe["ev_contrib"].mean()},
    ])
    lines.append(_fmt(strat) + "\n")

    flat_wins_conv = float((oe["flat_p"] > oe["ev_p"]).mean())
    flat_loses_contrib = float((oe["flat_contrib"] < oe["ev_contrib"]).mean())
    both = float(((oe["flat_p"] > oe["ev_p"]) & (oe["flat_contrib"] < oe["ev_contrib"])).mean())
    lines.append(f"\n- Flat 10% converts better than the economic action in "
                 f"**{flat_wins_conv:.1%}** of opportunities\n")
    lines.append(f"- Flat 10% yields lower net contribution in **{flat_loses_contrib:.1%}**\n")
    lines.append(f"- **Both simultaneously (converts better, earns less): {both:.1%}** "
                 "— these are the cases that demonstrate the thesis\n")

    if both < 0.10:
        warnings.append(
            f"Only {both:.1%} of cases show the 'higher conversion, lower profit' pattern. "
            "The central demonstration may be weak."
        )

    # ---- 8. distributions ------------------------------------------------
    lines.append("## 10. Key distributions\n")
    dist = pd.DataFrame({
        "cart_value": opp["cart_value"].describe(),
        "shipping_fee_charged": opp["shipping_fee_charged"].describe(),
        "shipping_cost": opp["shipping_cost"].describe(),
        "base_contribution_margin": opp["base_contribution_margin"].describe(),
        "minutes_since_event": opp["minutes_since_event"].describe(),
    })
    lines.append(dist.to_markdown(floatfmt=".2f") + "\n")

    margin_pct = 100 * opp["base_contribution_margin"] / opp["cart_value"]
    lines.append(f"\nContribution margin %: mean {margin_pct.mean():.1f}%, "
                 f"p05 {margin_pct.quantile(0.05):.1f}%, p95 {margin_pct.quantile(0.95):.1f}%\n")
    if (margin_pct < 0).mean() > 0.05:
        warnings.append(f"{(margin_pct < 0).mean():.1%} of opportunities have negative base margin.")

    lines.append("\n### Opportunity type mix\n")
    lines.append(opp["opportunity_type"].value_counts().to_frame("n").to_markdown() + "\n")
    lines.append("\n### Failure reason mix (payment failures)\n")
    fr = opp[opp["opportunity_type"] == "PAYMENT_FAILURE"]["failure_reason"].value_counts()
    lines.append(fr.to_frame("n").to_markdown() + "\n")

    # ---- 9. hidden mechanisms -------------------------------------------
    lines.append("## 11. Hidden environmental mechanisms\n")
    lines.append("These windows shift outcomes but are never exposed as features:\n")
    lines.append("```json\n" + json.dumps(manifest["hidden_environment"], indent=2) + "\n```\n")

    # ---- 10. leakage -----------------------------------------------------
    lines.append("## 12. Leakage checks\n")
    leaked = [c for c in f["customers"].columns
              if any(c.startswith(p) for p in FORBIDDEN_FEATURE_PREFIXES)]
    checks = [
        ("no hidden_* columns in observable customers", not leaked),
        ("oracle stored in a separate quarantined file", "oracle" in manifest["counts"]),
        ("opportunities sorted chronologically", opp["detected_at"].is_monotonic_increasing),
        ("all propensities > 0", bool((iv["action_propensity"] > 0).all())),
        ("all propensities <= 1", bool((iv["action_propensity"] <= 1).all())),
        ("train precedes validation precedes test",
         _split_ordered(opp)),
    ]
    for label, ok in checks:
        lines.append(f"- {'PASS' if ok else 'FAIL'} — {label}")
        if not ok:
            warnings.append(f"Leakage check failed: {label}")
    lines.append("")

    # ---- 11. difficulty --------------------------------------------------
    lines.append("## 13. Is the simulator too easy?\n")
    p_all = oracle[act_cols].values
    lines.append(f"- Oracle P(recovery) mean {np.nanmean(p_all):.3f}, sd {np.nanstd(p_all):.3f}\n")
    lines.append(f"- Realised conversion rate: {iv['converted_after_intervention'].mean():.3f}\n")
    lines.append(
        "\n> The response surface carries shared logit noise plus unobserved environment "
        "windows. If a trained model later reaches test ROC-AUC above ~0.85 or near-perfect "
        "calibration, treat that as a simulator defect and raise `response_logit_noise_sd`.\n"
    )

    # ---- verdict ---------------------------------------------------------
    lines.append("## Verdict\n")
    if warnings:
        lines.append(f"**{len(warnings)} warning(s) — review before training.**\n")
        for wmsg in warnings:
            lines.append(f"- {wmsg}")
    else:
        lines.append("**No blocking warnings. Dataset is ready for feature engineering (Phase 3).**")
    lines.append("")
    return "\n".join(lines), warnings


def _split_ordered(opp: pd.DataFrame) -> bool:
    try:
        mx = opp.groupby("split")["detected_at"].max()
        mn = opp.groupby("split")["detected_at"].min()
        return bool(mx["TRAIN"] <= mn["VALIDATION"] and mx["VALIDATION"] <= mn["TEST"])
    except KeyError:
        return False


def main() -> None:
    text, warnings = build_report()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text)
    print(text)
    print(f"\nWrote {OUT}")
    if warnings:
        print(f"\n{len(warnings)} warning(s) raised — see Verdict section.")


if __name__ == "__main__":
    main()
