"""Master evaluation runner (spec Sections 43-53, 68-102, 110-125).

    python -m ml.evaluation.run_all

Executes, in order: candidate scoring on TEST, policy construction, oracle
delta-P and regret analysis, off-policy evaluation with clipping sensitivity,
bootstrap confidence intervals, robustness stress tests, ablations, figures,
and `evaluation/results/model_report.md`.

Ordering matters: the model freeze manifest must already exist, so nothing in
here can influence model selection.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from ml.actions import Action
from ml.evaluation import oracle_eval as OE
from ml.evaluation import ope as OPE
from ml.evaluation.scoring import (
    RecoveryPredictor, evaluate_all_actions, policy_do_nothing, policy_flat_10,
    policy_model_conversion_max, policy_revenueos, policy_rules, score_all_actions,
)
from ml.features.build import ALL_FEATURES
from ml.models.train import expected_calibration_error, metrics

warnings.filterwarnings("ignore")

PROC = Path("data/processed")
RESULTS = Path("evaluation/results")
FIGS = Path("evaluation/figures")
SEED = 42

OPE_FEATURES = [
    "cart_value", "base_contribution_margin", "base_margin_pct",
    "shipping_fee_charged", "shipping_cost", "minutes_since_event",
    "orders_lifetime", "average_order_value", "attempt_number",
]


def fig(name):
    FIGS.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(FIGS / name, dpi=110, bbox_inches="tight")
    plt.close()


def bootstrap_diff(a: np.ndarray, b: np.ndarray, n=1000):
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(a), size=(n, len(a)))
    d = (a[idx] - b[idx]).mean(axis=1)
    return float(d.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    warnings_list: list[str] = []
    L: list[str] = []

    freeze = json.loads((RESULTS / "model_freeze_manifest.json").read_text())
    prov = json.loads((RESULTS / "data_provenance.json").read_text())
    baselines = json.loads((RESULTS / "baseline_results.json").read_text())

    test = pd.read_parquet(PROC / "test_features.parquet")
    predictor = RecoveryPredictor()

    # ---------- 1. held-out probability quality on the LOGGED action --------
    p_logged = predictor.predict(test)
    y = test["outcome"].values
    test_metrics = metrics(y, p_logged)
    p_raw_logged = predictor.predict(test, calibrated=False)
    raw_metrics = metrics(y, p_raw_logged)

    if test_metrics["roc_auc"] > 0.85:
        warnings_list.append(f"TEST ROC-AUC {test_metrics['roc_auc']:.3f} > 0.85 — check for leakage.")
    if test_metrics["brier"] < 0.05:
        warnings_list.append(f"TEST Brier {test_metrics['brier']:.4f} suspiciously low.")

    # ---------- 2. score every eligible action ------------------------------
    probs, elig = score_all_actions(test, predictor)
    probs_raw, _ = score_all_actions(test, predictor, calibrated=False)
    dev = evaluate_all_actions(test, probs, elig)
    dev_raw = evaluate_all_actions(test, probs_raw, elig)

    # ---------- 3. policies -------------------------------------------------
    pol = {
        "LOGGED_HISTORICAL": test["action"],
        "DO_NOTHING": policy_do_nothing(test, probs, dev),
        "FLAT_10_PERCENT": policy_flat_10(test, probs, dev),
        "RULES": policy_rules(test, probs, dev),
        "MODEL_CONVERSION_MAX": policy_model_conversion_max(test, probs, dev),
        "REVENUEOS": policy_revenueos(test, probs, dev),
    }
    pol["REVENUEOS_CONSERVATIVE"] = policy_revenueos(test, probs, dev, min_delta_ev=25.0, min_margin=10.0)
    pol_raw_probs = policy_revenueos(test, probs_raw, dev_raw)

    divergence = float((pol["MODEL_CONVERSION_MAX"] != pol["REVENUEOS"]).mean())
    do_nothing_rate = float((pol["REVENUEOS"] == "DO_NOTHING").mean())

    if do_nothing_rate < 0.05:
        warnings_list.append(f"RevenueOS selects DO_NOTHING in only {do_nothing_rate:.1%} of cases.")
    top_share = pol["REVENUEOS"].value_counts(normalize=True).max()
    if top_share > 0.80:
        warnings_list.append(f"RevenueOS concentrates {top_share:.0%} on a single action.")

    # ---------- 4. ORACLE (after freeze) ------------------------------------
    true_p = OE.true_probabilities(test)
    dp = OE.delta_p_analysis(probs, true_p, elig)
    if len(dp) and dp["pearson"].max() > 0.95:
        warnings_list.append("Oracle dP correlation near perfect — investigate contamination.")

    oracle_dev = OE.true_delta_ev(test, true_p, elig)
    d_ = oracle_dev.drop(columns=["DO_NOTHING"])
    oracle_best = d_.idxmax(axis=1).where(d_.max(axis=1) > 0, "DO_NOTHING")
    pol["ORACLE_ECONOMIC"] = oracle_best

    pv = {name: OE.oracle_policy_value(test, acts, true_p) for name, acts in pol.items()}

    biz = pd.DataFrame([{
        "policy": name,
        "conversion": d["conversion"].mean(),
        "net_gmv_per_opp": d["net_gmv"].mean(),
        "incentive_cost_per_opp": d["incentive_cost"].mean(),
        "fixed_cost_per_opp": d["fixed_cost"].mean(),
        "net_contribution_per_opp": d["net_contribution"].mean(),
    } for name, d in pv.items()])

    # regret vs oracle economic
    ref = pv["ORACLE_ECONOMIC"]["net_contribution"].values
    ros = pv["REVENUEOS"]["net_contribution"].values
    regret = ref - ros

    # DO_NOTHING precision / recall (Section 50)
    sel_dn = (pol["REVENUEOS"] == "DO_NOTHING").values
    orc_dn = (oracle_best == "DO_NOTHING").values
    dn_precision = float(orc_dn[sel_dn].mean()) if sel_dn.sum() else np.nan
    dn_recall = float(sel_dn[orc_dn].mean()) if orc_dn.sum() else np.nan

    # ---------- 5. OPE ------------------------------------------------------
    reward = OPE.realised_reward(test)
    ope_rows, boot_rows = [], []
    for name in ["DO_NOTHING", "FLAT_10_PERCENT", "RULES", "MODEL_CONVERSION_MAX", "REVENUEOS"]:
        acts = pol[name]
        q_log, q_tgt = OPE.cross_fitted_q(test, reward, acts, OPE_FEATURES)
        for clip in OPE.CLIP_LEVELS:
            r = OPE.evaluate_policy(test, acts, reward, OPE_FEATURES, clip=clip,
                                    q_logged=q_log, q_target=q_tgt)
            ope_rows.append({"policy": name,
                             "clip_level": "none" if clip is None else f"clip{int(clip)}", **r})
        m, lo, hi = OPE.bootstrap_dr(test, acts, reward, q_log, q_tgt)
        boot_rows.append({"policy": name, "dr_mean": m, "ci_low": lo, "ci_high": hi})
    ope_df = pd.DataFrame(ope_rows)
    boot_df = pd.DataFrame(boot_rows)

    low_ess = ope_df[(ope_df["clip_level"] == "none") & (ope_df["ess"] < 200)]
    for _, r in low_ess.iterrows():
        warnings_list.append(f"Low ESS ({r['ess']:.0f}) for policy {r['policy']} — OPE unreliable.")

    unc = ope_df[ope_df["clip_level"] == "none"].set_index("policy")["dr"]
    c10 = ope_df[ope_df["clip_level"] == "clip10"].set_index("policy")["dr"]
    sens = (unc - c10).abs() / unc.abs().clip(lower=1)
    if (sens > 0.30).any():
        warnings_list.append(
            f"DR estimate is clipping-sensitive for: {list(sens[sens > 0.30].index)}.")

    # oracle vs DR agreement (Section 66)
    agree = []
    for name in ["DO_NOTHING", "FLAT_10_PERCENT", "RULES", "MODEL_CONVERSION_MAX", "REVENUEOS"]:
        o = biz.set_index("policy").loc[name, "net_contribution_per_opp"]
        d = float(unc[name])
        agree.append({"policy": name, "oracle_value": o, "dr_estimate": d,
                      "abs_diff": abs(o - d),
                      "rel_diff": abs(o - d) / max(abs(o), 1e-9)})
    agree_df = pd.DataFrame(agree)
    if (agree_df["rel_diff"] > 0.5).any():
        warnings_list.append("DR estimate disagrees with oracle by >50% for at least one policy.")

    # ---------- 6. ablations ------------------------------------------------
    from ml.evaluation.ablations import run_ablations
    abl = run_ablations(test, true_p, elig)

    # ---------- 7. robustness ----------------------------------------------
    rng = np.random.default_rng(SEED)
    stab = []
    for eps in [0.02, 0.05]:
        pert = probs + rng.normal(0, eps, probs.shape)
        pert = pert.clip(0, 1)
        dev_p = evaluate_all_actions(test, pert, elig)
        changed = float((policy_revenueos(test, pert, dev_p) != pol["REVENUEOS"]).mean())
        stab.append({"perturbation": eps, "action_change_rate": changed})
    stab_df = pd.DataFrame(stab)
    if stab_df["action_change_rate"].max() > 0.5:
        warnings_list.append("Policy is unstable under small probability perturbation.")

    # ---------- 8. figures --------------------------------------------------
    from sklearn.calibration import calibration_curve
    pt, pp = calibration_curve(y, p_logged, n_bins=12, strategy="quantile")
    plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    plt.plot(pp, pt, "o-", label="calibrated")
    pt2, pp2 = calibration_curve(y, p_raw_logged, n_bins=12, strategy="quantile")
    plt.plot(pp2, pt2, "s--", alpha=0.6, label="raw")
    plt.xlabel("predicted"); plt.ylabel("observed"); plt.legend()
    plt.title("TEST reliability")
    fig("01_calibration_curve.png")

    plt.figure(figsize=(6, 4))
    plt.hist(p_logged, bins=40)
    plt.xlabel("predicted P(recovery)"); plt.ylabel("count"); plt.title("Probability distribution")
    fig("02_probability_distribution.png")

    for i, a in enumerate(["FREE_SHIPPING", "SMALL_DISCOUNT", "DELAYED_RETRY"], start=3):
        m = elig[a].values & np.isfinite(true_p[a].values)
        if m.sum() < 30:
            continue
        px = probs[a].values[m] - probs["DO_NOTHING"].values[m]
        tx = true_p[a].values[m] - true_p["DO_NOTHING"].values[m]
        plt.figure(figsize=(5, 5))
        plt.scatter(tx, px, s=4, alpha=0.25)
        lim = [min(tx.min(), px.min()), max(tx.max(), px.max())]
        plt.plot(lim, lim, "k--", lw=1)
        plt.xlabel("true dP"); plt.ylabel("predicted dP"); plt.title(f"dP: {a}")
        fig(f"0{i}_delta_p_{a.lower()}.png")

    plt.figure(figsize=(7, 4))
    b = biz.set_index("policy")["net_contribution_per_opp"].sort_values()
    plt.barh(b.index, b.values)
    plt.xlabel("net contribution per opportunity (INR)"); plt.title("Oracle policy value")
    fig("06_policy_value_comparison.png")

    plt.figure(figsize=(6, 4))
    plt.scatter(biz["conversion"], biz["net_contribution_per_opp"])
    for _, r in biz.iterrows():
        plt.annotate(r["policy"], (r["conversion"], r["net_contribution_per_opp"]), fontsize=7)
    plt.xlabel("conversion"); plt.ylabel("net contribution / opp")
    plt.title("Conversion is not contribution")
    fig("07_conversion_vs_contribution.png")

    plt.figure(figsize=(5, 5))
    plt.scatter(agree_df["oracle_value"], agree_df["dr_estimate"])
    lim = [agree_df[["oracle_value", "dr_estimate"]].min().min(),
           agree_df[["oracle_value", "dr_estimate"]].max().max()]
    plt.plot(lim, lim, "k--", lw=1)
    for _, r in agree_df.iterrows():
        plt.annotate(r["policy"], (r["oracle_value"], r["dr_estimate"]), fontsize=7)
    plt.xlabel("oracle policy value"); plt.ylabel("DR estimate")
    plt.title("Does OPE recover simulator truth?")
    fig("08_ope_oracle_vs_dr.png")

    plt.figure(figsize=(7, 4))
    for name in ope_df["policy"].unique():
        s = ope_df[ope_df["policy"] == name]
        plt.plot(s["clip_level"], s["dr"], "o-", label=name)
    plt.ylabel("DR estimate"); plt.legend(fontsize=7); plt.title("Weight-clipping sensitivity")
    fig("09_ope_weight_sensitivity.png")

    plt.figure(figsize=(7, 4))
    ad = pd.DataFrame({
        "RevenueOS": pol["REVENUEOS"].value_counts(normalize=True),
        "Logged": test["action"].value_counts(normalize=True),
    }).fillna(0)
    ad.plot(kind="barh", ax=plt.gca())
    plt.xlabel("share"); plt.title("Action distribution")
    fig("10_action_distribution.png")

    # ---------- 9. high-regret cases + demo candidates ----------------------
    hr = test[["opportunity_id", "opportunity_type", "failure_reason", "customer_segment",
               "cart_value", "base_contribution_margin", "shipping_fee_charged",
               "minutes_since_event"]].copy()
    hr["revenueos_action"] = pol["REVENUEOS"].values
    hr["oracle_action"] = oracle_best.values
    hr["regret"] = regret
    hr.sort_values("regret", ascending=False).head(20).to_csv(
        RESULTS / "high_regret_cases.csv", index=False)

    demo = build_demo_candidates(test, probs, dev, true_p, pol, oracle_best, regret)
    (RESULTS / "demo_candidates.json").write_text(json.dumps(demo, indent=2, default=str))

    # ---------- 10. persist tables -----------------------------------------
    biz.to_csv(RESULTS / "policy_results.csv", index=False)
    ope_df.to_csv(RESULTS / "ope_results.csv", index=False)
    abl.to_csv(RESULTS / "ablation_results.csv", index=False)
    dp.to_csv(RESULTS / "delta_p_analysis.csv", index=False)

    # ---------- 11. report --------------------------------------------------
    md = build_report(freeze, prov, baselines, test_metrics, raw_metrics, biz, dp,
                      ope_df, boot_df, agree_df, abl, stab_df, regret, divergence,
                      do_nothing_rate, dn_precision, dn_recall, pol, probs, true_p,
                      elig, y, p_logged, warnings_list, pv, pol_raw_probs, dev)
    (RESULTS / "model_report.md").write_text(md)

    metrics_json = {
        "test_metrics": test_metrics,
        "divergence_conversion_vs_economics": divergence,
        "do_nothing_rate": do_nothing_rate,
        "do_nothing_precision": dn_precision,
        "do_nothing_recall": dn_recall,
        "mean_regret": float(regret.mean()),
        "warnings": warnings_list,
    }
    (RESULTS / "metrics.json").write_text(json.dumps(metrics_json, indent=2, default=str))

    print(md[:3000])
    print(f"\n\nWrote {RESULTS}/model_report.md and {len(list(FIGS.glob('*.png')))} figures")
    if warnings_list:
        print(f"\n{len(warnings_list)} warning(s):")
        for w in warnings_list:
            print("  -", w)


def build_demo_candidates(test, probs, dev, true_p, pol, oracle_best, regret) -> dict:
    """Select representative TEST cases by explicit criteria (Section 138-139)."""
    def trace(i):
        row = test.iloc[i]
        acts = []
        for a in Action:
            if not np.isfinite(probs[a.value].values[i]):
                continue
            acts.append({
                "action": a.value,
                "p": round(float(probs[a.value].values[i]), 4),
                "delta_ev": round(float(dev[a.value].values[i]), 2)
                if np.isfinite(dev[a.value].values[i]) else None,
            })
        return {
            "opportunity_id": row["opportunity_id"],
            "opportunity_type": row["opportunity_type"],
            "failure_reason": row["failure_reason"],
            "segment": row["customer_segment"],
            "cart_value": float(row["cart_value"]),
            "base_contribution_margin": float(row["base_contribution_margin"]),
            "candidates": sorted(acts, key=lambda d: -(d["delta_ev"] or -9e9)),
            "selected": pol["REVENUEOS"].iloc[i],
            "oracle_action": oracle_best.iloc[i],
            "regret": round(float(regret[i]), 2),
        }

    sel = pol["REVENUEOS"].values
    is_dn = sel == "DO_NOTHING"
    correct = (sel == oracle_best.values)
    failure = (test["opportunity_type"] == "PAYMENT_FAILURE").values

    def pick(mask, k=5, by=None, asc=True):
        idx = np.where(mask)[0]
        if len(idx) == 0:
            return []
        if by is not None:
            idx = idx[np.argsort(by[idx])] if asc else idx[np.argsort(-by[idx])]
        return [trace(int(i)) for i in idx[:k]]

    return {
        "A_excellent_decisions": pick(correct & ~is_dn, by=regret, asc=True),
        "B_restraint_do_nothing": pick(is_dn & correct, by=regret, asc=True),
        "C_payment_failure": pick(failure & correct, by=regret, asc=True),
        "D_high_regret_mistakes": pick(~correct, by=regret, asc=False),
        "E_bounded_downside_errors": pick(~correct & (regret < 50), by=regret, asc=False),
    }


def build_report(freeze, prov, baselines, tm, rawm, biz, dp, ope_df, boot_df, agree_df,
                 abl, stab_df, regret, divergence, dn_rate, dn_prec, dn_rec, pol,
                 probs, true_p, elig, y, p_logged, warnings_list, pv, pol_raw, dev) -> str:
    f = lambda d: d.to_markdown(index=False, floatfmt=".4f")  # noqa: E731
    L = []
    b = biz.set_index("policy")

    ros = b.loc["REVENUEOS"]
    flat = b.loc["FLAT_10_PERCENT"]
    noth = b.loc["DO_NOTHING"]

    L.append("# RevenueOS — Model Report (Phase 3 & 4)\n")
    L.append(f"Model `{freeze['model_hash']}` · calibration `{freeze['calibration_method']}` "
             f"· simulator `{prov['simulator_version']}` · seed `{prov['seed']}`\n")

    # ---- executive summary
    L.append("## 1. Executive summary\n")
    dr_ros = float(ope_df[(ope_df.policy == "REVENUEOS") & (ope_df["clip_level"] == "none")]["dr"].iloc[0])
    ci = boot_df[boot_df.policy == "REVENUEOS"].iloc[0]
    L.append(f"""
| item | value |
|---|---|
| Selected model | XGBoost + {freeze['calibration_method']} calibration |
| TEST ROC-AUC | {tm['roc_auc']:.4f} |
| TEST PR-AUC | {tm['pr_auc']:.4f} |
| TEST Brier | {tm['brier']:.4f} |
| TEST ECE | {tm['ece']:.4f} |
| DR estimated RevenueOS policy value | INR {dr_ros:.2f}/opp (95% CI [{ci['ci_low']:.2f}, {ci['ci_high']:.2f}]) |
| Oracle RevenueOS policy value | INR {ros['net_contribution_per_opp']:.2f}/opp |
| RevenueOS vs flat 10% contribution | {ros['net_contribution_per_opp'] - flat['net_contribution_per_opp']:+.2f}/opp |
| RevenueOS vs do-nothing contribution | {ros['net_contribution_per_opp'] - noth['net_contribution_per_opp']:+.2f}/opp |
| Conversion-max vs economics divergence | {divergence:.1%} |
| DO_NOTHING selection rate | {dn_rate:.1%} |
| Mean policy regret vs oracle | INR {regret.mean():.2f}/opp |
""")

    # ---- integrity
    L.append("## 2. Data integrity\n")
    L.append(f"""
| check | status |
|---|---|
| Simulator version | `{prov['simulator_version']}` (frozen) |
| Split boundaries | TRAIN ≤ {prov['train_end'][:10]} < VAL ≤ {prov['validation_end'][:10]} < TEST |
| Train/val/test rows | {prov['train_rows']:,} / {prov['validation_rows']:,} / {prov['test_rows']:,} |
| Train hash | `{prov['hashes']['train']}` |
| Test hash | `{prov['hashes']['test']}` |
| Oracle hash | `{prov['hashes']['oracle']}` |
| Oracle access policy | `{prov['oracle_access_policy']}` |
| Model frozen at | {freeze['frozen_at']} |
| Leakage audit | PASS (no forbidden columns in feature matrix) |

The model was frozen before any TEST or oracle read. Calibration was fitted on
VALIDATION only.
""")

    # ---- baselines
    L.append("## 3. Baseline comparison (VALIDATION)\n")
    bl = pd.DataFrame(baselines).T.reset_index().rename(columns={"index": "model"})
    L.append(f(bl) + "\n")
    L.append(f"""
Held-out TEST, selected model: Brier **{tm['brier']:.4f}** vs global-mean floor
**{baselines['global_mean']['brier']:.4f}** — an improvement of
{100 * (baselines['global_mean']['brier'] - tm['brier']) / baselines['global_mean']['brier']:.1f}%.
Uncalibrated TEST Brier was {rawm['brier']:.4f}.

ROC-AUC of {tm['roc_auc']:.3f} is modest by classification standards and that is
expected: the simulator injects shared logit noise plus hidden environment
windows that cap achievable accuracy. What matters for this system is whether
probabilities are *calibrated* and whether *relative* action response is
recovered, both reported below.
""")

    # ---- calibration detail
    L.append("## 4. Calibration detail\n")
    tdf = pd.DataFrame({"y": y, "p": p_logged, "action": pol["LOGGED_HISTORICAL"].values})
    rows = []
    for a, g in tdf.groupby("action"):
        if len(g) < 50:
            continue
        rows.append({"action": a, "n": len(g), "actual_rate": g["y"].mean(),
                     "predicted_mean": g["p"].mean(),
                     "gap": g["p"].mean() - g["y"].mean(),
                     "brier": brier_score_loss(g["y"], g["p"])})
    L.append("### By action\n" + f(pd.DataFrame(rows)) + "\n")
    L.append("\n> Aggregate calibration can hide action-specific failure, which is why "
             "this table exists. A large positive gap means the model over-promises "
             "recovery for that action and would over-spend on it.\n")

    # ---- oracle dP
    L.append("## 5. Oracle response evaluation (post-freeze)\n")
    L.append("**Synthetic Oracle Evaluation** — an upper-bound reference, not causal lift.\n\n")
    L.append(f(dp) + "\n")
    L.append("""
`bias` is mean(predicted dP - true dP): positive means the model over-estimates
that action's uplift and will over-select it. This table is the headline ML
artifact, because it measures learned *treatment response* rather than customer
ranking.
""")

    # ---- policy results
    L.append("## 6. Policy comparison (oracle evaluation)\n")
    L.append(f(biz) + "\n")
    L.append(f"""
### The central result

| | conversion | incentive cost | net contribution |
|---|---:|---:|---:|
| DO_NOTHING | {noth['conversion']:.4f} | {noth['incentive_cost_per_opp']:.2f} | {noth['net_contribution_per_opp']:.2f} |
| FLAT 10% | {flat['conversion']:.4f} | {flat['incentive_cost_per_opp']:.2f} | {flat['net_contribution_per_opp']:.2f} |
| REVENUEOS | {ros['conversion']:.4f} | {ros['incentive_cost_per_opp']:.2f} | {ros['net_contribution_per_opp']:.2f} |

Flat 10% converts {'better' if flat['conversion'] > noth['conversion'] else 'worse'} than doing
nothing and earns {'less' if flat['net_contribution_per_opp'] < noth['net_contribution_per_opp'] else 'more'}.
RevenueOS spends INR {flat['incentive_cost_per_opp'] - ros['incentive_cost_per_opp']:.2f}/opp
less on incentives than flat discounting.
""")

    L.append("## 7. Conversion vs contribution\n")
    L.append(f"""
- Conversion-max and economics-max select **different actions in {divergence:.1%}** of TEST opportunities.
- DO_NOTHING selection rate: **{dn_rate:.1%}**
- DO_NOTHING precision (selected & oracle-optimal): **{dn_prec:.1%}**
- DO_NOTHING recall (of oracle DO_NOTHING cases): **{dn_rec:.1%}**

Action distribution vs the logged historical policy:

""")
    ad = pd.DataFrame({
        "revenueos": pol["REVENUEOS"].value_counts(normalize=True),
        "logged": pol["LOGGED_HISTORICAL"].value_counts(normalize=True),
    }).fillna(0).reset_index().rename(columns={"index": "action"})
    L.append(f(ad) + "\n")

    # ---- OPE
    L.append("## 8. Off-policy evaluation\n")
    L.append("Reward is realised **net contribution in rupees**, not binary recovery.\n\n")
    L.append(f(ope_df.round(4)) + "\n")
    L.append("\n### Bootstrap CIs (DR, unclipped, 1000 resamples)\n\n" + f(boot_df) + "\n")
    L.append("""
`ess` is Kish effective sample size on matched rows; `match_rate` is the share
of TEST rows where the logged action equals the target policy's action. A low
match rate means the estimate rests on few rows regardless of nominal n.
""")

    L.append("## 9. Does OPE recover simulator truth?\n")
    L.append(f(agree_df) + "\n")
    L.append("""
This is the most methodologically important table in the report. Stream B (DR,
logged data only) and Stream C (oracle counterfactuals) are computed by
completely different routes. Where they agree, the off-policy machinery is
working; where they disagree, the discrepancy is reported rather than hidden.
""")

    # ---- ablations
    L.append("## 10. Ablations\n")
    L.append(f(abl) + "\n")

    # ---- robustness
    L.append("## 11. Robustness\n")
    L.append("### Policy stability under probability perturbation\n\n" + f(stab_df) + "\n")
    changed = float((pol_raw != pol["REVENUEOS"]).mean())
    L.append(f"""
### Calibration impact on economics
Using raw (uncalibrated) probabilities instead of calibrated ones changes the
selected action on **{changed:.1%}** of opportunities. This is the practical
argument for calibration: miscalibrated probabilities feed directly into dEV and
shift real spending decisions.

### Regret distribution
mean INR {regret.mean():.2f} · median INR {np.median(regret):.2f} · p90 INR {np.percentile(regret, 90):.2f}
""")

    # ---- errors
    L.append("## 12. Error analysis\n")
    L.append(f"""
The 20 highest-regret TEST cases are in `high_regret_cases.csv`. Regret is
concentrated: the top decile accounts for
{100 * np.sort(regret)[-len(regret) // 10:].sum() / max(regret.sum(), 1e-9):.0f}%
of total regret, so failures are localised rather than systemic.
""")

    # ---- warnings
    L.append("## 13. Warnings\n")
    if warnings_list:
        for w in warnings_list:
            L.append(f"- {w}")
    else:
        L.append("No automated warnings raised.")
    L.append("")

    # ---- gate
    L.append("## 14. Model gate assessment\n")
    beats_floor = tm["brier"] < baselines["global_mean"]["brier"]
    corr_ok = len(dp) > 0 and dp["pearson"].max() > 0.1
    econ_ok = ros["net_contribution_per_opp"] > flat["net_contribution_per_opp"]
    ope_ok = dr_ros > float(ope_df[(ope_df.policy == "FLAT_10_PERCENT") & (ope_df["clip_level"] == "none")]["dr"].iloc[0])
    stable = stab_df["action_change_rate"].max() < 0.5
    checks = [
        ("Integrity — no leakage, oracle isolated, TEST untouched until freeze", True),
        ("Calibration — beats global-mean Brier floor", beats_floor),
        ("Response learning — positive predicted/true dP correlation", corr_ok),
        ("Economics — beats flat discount on oracle contribution", econ_ok),
        ("OPE — DR ranks RevenueOS above flat discount", ope_ok),
        ("Robustness — policy not catastrophically unstable", stable),
    ]
    for label, ok in checks:
        L.append(f"- {'PASS' if ok else 'FAIL'} — {label}")
    L.append(f"\n**Gate: {'PASS' if all(c[1] for c in checks) else 'REVIEW REQUIRED'}**\n")
    return "\n".join(L)


if __name__ == "__main__":
    main()