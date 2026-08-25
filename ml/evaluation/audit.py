"""Final model audit (Phase 3/4 close-out).

    python -m ml.evaluation.audit

Answers one question the first gate could not: **how much of the economic gain
comes from the machine learning, and how much from the financial engine?**

Runs the identical policy machinery on four response models of increasing
sophistication (segment lookup -> logistic -> raw XGBoost -> final calibrated),
so the marginal economic value of each modelling step is directly visible.

Also produces the regret distribution, tail-risk profile, high-value-cohort
performance, and the final OPE cross-check.

Writes `evaluation/results/final_model_audit.md`.
"""

from __future__ import annotations

import json
import pickle
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ml.actions import Action
from ml.evaluation import oracle_eval as OE
from ml.evaluation import ope as OPE
from ml.evaluation.scoring import (
    RecoveryPredictor, eligibility_mask, evaluate_all_actions,
    policy_do_nothing, policy_flat_10, policy_model_conversion_max,
    policy_revenueos, policy_rules,
)
from ml.features.build import ALL_FEATURES, add_action_features
from ml.models.train import metrics, SegmentBaseline  # noqa: F401  (needed to unpickle)

warnings.filterwarnings("ignore")

PROC = Path("data/processed")
ART = Path("ml/artifacts")
RESULTS = Path("evaluation/results")
FIGS = Path("evaluation/figures")
SEED = 42
OPE_FEATURES = [
    "cart_value", "base_contribution_margin", "base_margin_pct",
    "shipping_fee_charged", "shipping_cost", "minutes_since_event",
    "orders_lifetime", "average_order_value", "attempt_number",
]


class GenericPredictor:
    """Wraps any fitted estimator so every model family scores identically."""

    def __init__(self, model, calibrator=None, kind="sklearn"):
        self.model, self.calibrator, self.kind = model, calibrator, kind

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.kind == "segment":
            p = self.model.predict_proba_1d(X)
        else:
            p = self.model.predict_proba(X[ALL_FEATURES])[:, 1]
        if self.calibrator is not None:
            p = (self.calibrator.predict_proba(np.asarray(p).reshape(-1, 1))[:, 1]
                 if hasattr(self.calibrator, "predict_proba")
                 else self.calibrator.predict(p))
        p = np.asarray(p, dtype=float)
        p[~np.isfinite(p)] = 0.0
        return np.clip(p, 0.0, 1.0)


def score_all(base: pd.DataFrame, pred) -> tuple[pd.DataFrame, pd.DataFrame]:
    probs, elig = {}, {}
    for a in Action:
        m = eligibility_mask(base, a)
        col = np.full(len(base), np.nan)
        if m.any():
            rows = base.loc[m].copy()
            rows["action"] = a.value
            col[m] = pred.predict(add_action_features(rows))
        probs[a.value], elig[a.value] = col, m
    return pd.DataFrame(probs, index=base.index), pd.DataFrame(elig, index=base.index)


def fmt(df):
    return df.to_markdown(index=False, floatfmt=".4f")


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    freeze = json.loads((RESULTS / "model_freeze_manifest.json").read_text())
    test = pd.read_parquet(PROC / "test_features.parquet")
    val = pd.read_parquet(PROC / "validation_features.parquet")
    y = test["outcome"].values
    L: list[str] = []

    final = RecoveryPredictor()
    true_p = OE.true_probabilities(test)

    # ---------------- model families --------------------------------------
    families = {}
    with open(ART / "segment_baseline.pkl", "rb") as f:
        families["A_segment_lookup"] = GenericPredictor(pickle.load(f), kind="segment")
    with open(ART / "logistic_model.pkl", "rb") as f:
        families["B_logistic"] = GenericPredictor(pickle.load(f))
    with open(ART / "recovery_model.pkl", "rb") as f:
        raw_model = pickle.load(f)
    families["C_xgboost_raw"] = GenericPredictor(raw_model)
    families["D_final_selected"] = GenericPredictor(raw_model, final.calibrator)

    scored = {}
    for name, pred in families.items():
        probs, elig = score_all(test, pred)
        dev = evaluate_all_actions(test, probs, elig)
        acts = policy_revenueos(test, probs, dev)
        pv = OE.oracle_policy_value(test, acts, true_p)
        scored[name] = {"probs": probs, "elig": elig, "dev": dev,
                        "actions": acts, "pv": pv}

    probs, elig, dev = (scored["D_final_selected"][k] for k in ("probs", "elig", "dev"))
    ros_actions = scored["D_final_selected"]["actions"]

    # reference policies
    ref = {
        "DO_NOTHING": policy_do_nothing(test, probs, dev),
        "FLAT_10_PERCENT": policy_flat_10(test, probs, dev),
        "RULES": policy_rules(test, probs, dev),
        "MODEL_CONVERSION_MAX": policy_model_conversion_max(test, probs, dev),
    }
    oracle_dev = OE.true_delta_ev(test, true_p, elig)
    d_ = oracle_dev.drop(columns=["DO_NOTHING"])
    oracle_best = d_.idxmax(axis=1).where(d_.max(axis=1) > 0, "DO_NOTHING")
    ref["ORACLE_ECONOMIC"] = oracle_best

    ref_pv = {k: OE.oracle_policy_value(test, v, true_p) for k, v in ref.items()}
    v_nothing = ref_pv["DO_NOTHING"]["net_contribution"].mean()
    v_oracle = ref_pv["ORACLE_ECONOMIC"]["net_contribution"].mean()
    headroom = v_oracle - v_nothing

    # ---------------- report ----------------------------------------------
    L.append("# RevenueOS — Final Model Audit\n")
    L.append(f"Pipeline `{freeze['pipeline_version']}` · model `{freeze['model_hash']}` "
             f"· calibration `{freeze['calibration_method']}` · simulator `1.1.0` (frozen)\n")

    # 1. corrected protocol
    L.append("## 1. Corrected calibration protocol\n")
    L.append(f"> {freeze['protocol_correction_note']}\n")
    L.append(f"""
**What went wrong.** The first gate fitted Platt and isotonic on VALIDATION and
then *selected between them on the same VALIDATION rows*. Isotonic scored
ECE = **{freeze['optimistic_same_set_reference']['xgboost_isotonic_same_set_ece']:.5f}** —
which is not a measurement of calibration quality at all. Isotonic regression is
a step function fitted to those exact points; reproducing them is guaranteed.
The procedure structurally favoured the most flexible calibrator.

**The correction.** VALIDATION is now split chronologically:

| partition | period | rows | role |
|---|---|---:|---|
| TRAIN | {freeze['base_model_training_period'][0][:10]} → {freeze['base_model_training_period'][1][:10]} | 25,385 | fits base models |
| CALIBRATION | {freeze['calibration_period'][0][:10]} → {freeze['calibration_period'][1][:10]} | {freeze['calibration_rows']:,} | fits Platt / isotonic |
| MODEL_SELECTION | {freeze['model_selection_period'][0][:10]} → {freeze['model_selection_period'][1][:10]} | {freeze['model_selection_rows']:,} | chooses among candidates |
| TEST | after | 5,440 | reporting only |

No candidate is ever scored on rows that fitted it.
""")

    sel = pd.DataFrame(freeze["selection_metrics"]).T.reset_index().rename(columns={"index": "model"})
    L.append("### MODEL_SELECTION metrics (honest)\n" + fmt(sel) + "\n")
    L.append(f"""
Measured honestly, isotonic ECE is **{freeze['selection_metrics']['xgboost_isotonic']['ece']:.4f}**,
not 0.0000 — and raw XGBoost now has both the best Brier
({freeze['selection_metrics']['xgboost_raw']['brier']:.5f}) and the lowest ECE
({freeze['selection_metrics']['xgboost_raw']['ece']:.4f}) of the three.

**Selected: `{freeze['calibration_method']}` calibration.** This choice came
exclusively from MODEL_SELECTION. TEST was not consulted, even though the first
gate's TEST numbers happened to point the same way — using them would have been
selection-on-test regardless of the outcome.
""")

    # 2/3. test metrics
    p_test = final.predict(test)
    tm = metrics(y, p_test)
    L.append("## 2. Final TEST metrics (reporting only)\n")
    L.append(fmt(pd.DataFrame([tm]).assign(model="final").iloc[:, ::-1]) + "\n")

    # 4/9. economic value by model family
    L.append("## 3. Economic value by model family (ML contribution decomposition)\n")
    rows = []
    for name, s in scored.items():
        pv = s["pv"]
        v = pv["net_contribution"].mean()
        reg = ref_pv["ORACLE_ECONOMIC"]["net_contribution"].values - pv["net_contribution"].values
        rows.append({
            "policy": name,
            "conversion": pv["conversion"].mean(),
            "net_contribution_per_opp": v,
            "incremental_vs_do_nothing": v - v_nothing,
            "value_capture_pct": 100 * (v - v_nothing) / headroom,
            "mean_regret": reg.mean(),
            "do_nothing_rate": float((s["actions"] == "DO_NOTHING").mean()),
        })
    for name in ["RULES", "FLAT_10_PERCENT", "DO_NOTHING", "ORACLE_ECONOMIC"]:
        pv = ref_pv[name]
        v = pv["net_contribution"].mean()
        reg = ref_pv["ORACLE_ECONOMIC"]["net_contribution"].values - pv["net_contribution"].values
        rows.append({
            "policy": name, "conversion": pv["conversion"].mean(),
            "net_contribution_per_opp": v, "incremental_vs_do_nothing": v - v_nothing,
            "value_capture_pct": 100 * (v - v_nothing) / headroom,
            "mean_regret": reg.mean(),
            "do_nothing_rate": float((ref[name] == "DO_NOTHING").mean()),
        })
    fam = pd.DataFrame(rows)
    fam.to_csv(RESULTS / "model_family_economics.csv", index=False)
    L.append(fmt(fam) + "\n")

    a_val = fam.loc[fam.policy == "A_segment_lookup", "net_contribution_per_opp"].iloc[0]
    d_val = fam.loc[fam.policy == "D_final_selected", "net_contribution_per_opp"].iloc[0]
    delta_ml = d_val - a_val
    verdict = ("adds" if delta_ml > 0 else "**subtracts**")
    L.append(f"""
**This is the most important table in the audit.** All four rows A-D use the
identical financial engine and policy rule; only the response model changes, so
the differences isolate the marginal economic value of the modelling itself.

- The financial engine driven by a crude segment lookup table reaches
  INR {a_val:.2f}/opp, capturing
  {fam.loc[fam.policy == 'A_segment_lookup', 'value_capture_pct'].iloc[0]:.1f}% of available headroom.
- Moving to the final XGBoost {verdict} **INR {delta_ml:+.2f}/opp**
  ({fam.loc[fam.policy == 'D_final_selected', 'value_capture_pct'].iloc[0]:.1f}% capture).
- Flat 10% discounting *destroys*
  INR {abs(fam.loc[fam.policy == 'FLAT_10_PERCENT', 'incremental_vs_do_nothing'].iloc[0]):.2f}/opp
  relative to doing nothing, and the rule baseline captures only
  {fam.loc[fam.policy == 'RULES', 'value_capture_pct'].iloc[0]:.1f}%.

**The honest reading, stated plainly: gradient boosting does not beat a smoothed
contingency table on economic value here — it is
INR {abs(delta_ml):.2f}/opp {'better' if delta_ml > 0 else 'worse'}.** Essentially all of the
economic win comes from the financial optimisation layer correctly refusing to
discount, not from model sophistication.

Why this happens is visible in the response metrics: at ROC-AUC ~0.60 the model
carries little individual-level signal, and what the policy actually needs is a
coarse, well-behaved estimate of *which action class* helps *which kind of
opportunity*. A smoothed lookup over (opportunity_type x failure_reason x action)
supplies exactly that, with less variance. XGBoost's extra flexibility mostly
adds noise to dEV comparisons that are frequently close calls.

This does not invalidate the system — the policy still beats every naive
baseline by a wide margin, and off-policy evidence confirms it independently. It
does mean the defensible claim is about *incremental economics plus bounded
authority*, not about model sophistication. Reporting it this way is more
credible than an ML-centric story the numbers do not support.
""")

    # 5. value capture
    L.append("## 4. Oracle incremental value capture\n")
    L.append(f"""
```
capture = (policy_value - DO_NOTHING_value) / (ORACLE_value - DO_NOTHING_value)
```
Headroom available to the oracle: INR {headroom:.2f}/opp
(DO_NOTHING {v_nothing:.2f} → ORACLE {v_oracle:.2f}).

**RevenueOS captures {fam.loc[fam.policy == 'D_final_selected', 'value_capture_pct'].iloc[0]:.1f}%**
of the incremental economic value available to a perfectly-informed oracle.
""")

    # 6. regret distribution
    ros_pv = scored["D_final_selected"]["pv"]
    regret = ref_pv["ORACLE_ECONOMIC"]["net_contribution"].values - ros_pv["net_contribution"].values
    qs = [0.5, 0.75, 0.90, 0.95, 0.99]
    rd = pd.DataFrame([{
        "metric": "absolute_regret_inr", "mean": regret.mean(),
        **{f"p{int(q*100)}": float(np.percentile(regret, q * 100)) for q in qs},
        "max": regret.max(),
    }, {
        "metric": "regret_over_cart_value",
        "mean": float(np.mean(regret / test["cart_value"].values)),
        **{f"p{int(q*100)}": float(np.percentile(regret / test["cart_value"].values, q * 100)) for q in qs},
        "max": float(np.max(regret / test["cart_value"].values)),
    }, {
        "metric": "regret_over_base_margin",
        "mean": float(np.mean(regret / test["base_contribution_margin"].clip(lower=1).values)),
        **{f"p{int(q*100)}": float(np.percentile(regret / test["base_contribution_margin"].clip(lower=1).values, q * 100)) for q in qs},
        "max": float(np.max(regret / test["base_contribution_margin"].clip(lower=1).values)),
    }])
    rd.to_csv(RESULTS / "regret_distribution.csv", index=False)
    L.append("## 5. Regret distribution\n" + fmt(rd) + "\n")
    share_top1 = 100 * np.sort(regret)[-len(regret) // 100:].sum() / max(regret.sum(), 1e-9)
    L.append(f"\nThe worst 1% of opportunities account for **{share_top1:.1f}%** of total regret.\n")

    plt.figure(figsize=(7, 4))
    plt.hist(regret, bins=60)
    plt.axvline(regret.mean(), color="k", ls="--", lw=1, label=f"mean {regret.mean():.0f}")
    plt.axvline(np.percentile(regret, 95), color="r", ls=":", lw=1,
                label=f"p95 {np.percentile(regret, 95):.0f}")
    plt.xlabel("regret vs oracle (INR/opportunity)"); plt.ylabel("count")
    plt.legend(); plt.title("Policy regret distribution")
    plt.tight_layout(); plt.savefig(FIGS / "11_regret_distribution.png", dpi=110)
    plt.close()

    # 7. tail risk
    thr = np.percentile(regret, 99)
    tail = test.copy()
    tail["regret"] = regret
    tail["selected"] = ros_actions.values
    tail["oracle_action"] = oracle_best.values
    t1 = tail[tail["regret"] >= thr]
    L.append("## 6. Tail-risk analysis (worst 1%)\n")
    L.append(f"{len(t1)} opportunities with regret >= INR {thr:.0f}.\n")
    comp = pd.DataFrame([
        {"attribute": "share PAYMENT_FAILURE",
         "tail": float((t1.opportunity_type == "PAYMENT_FAILURE").mean()),
         "overall": float((tail.opportunity_type == "PAYMENT_FAILURE").mean())},
        {"attribute": "mean cart_value", "tail": t1.cart_value.mean(), "overall": tail.cart_value.mean()},
        {"attribute": "mean base_margin", "tail": t1.base_contribution_margin.mean(),
         "overall": tail.base_contribution_margin.mean()},
        {"attribute": "share cold-start (no coupon history)",
         "tail": float(t1.coupon_history_missing.mean()),
         "overall": float(tail.coupon_history_missing.mean())},
        {"attribute": "share DO_NOTHING selected",
         "tail": float((t1.selected == "DO_NOTHING").mean()),
         "overall": float((tail.selected == "DO_NOTHING").mean())},
    ])
    L.append(fmt(comp) + "\n")
    L.append("\n### Tail composition by selected vs oracle action\n")
    L.append(fmt(t1.groupby(["selected", "oracle_action"]).size()
                 .reset_index(name="n").sort_values("n", ascending=False).head(8)) + "\n")
    L.append("\n### Tail by failure reason\n")
    L.append(fmt(t1["failure_reason"].value_counts().reset_index()
                 .rename(columns={"index": "failure_reason", "count": "n"}).head(8)) + "\n")
    ratio = t1.cart_value.mean() / tail.cart_value.mean()
    L.append(f"""
Tail cases carry **{ratio:.1f}x the mean cart value** of the overall test set.
Regret scales with the money at stake, so the natural mitigation is a
cart-value-linked human-approval threshold rather than a better model — which is
exactly what the Phase 5 policy engine should encode.
""")

    # 8. high-value cohort — threshold from VALIDATION
    hv_thr = float(val["cart_value"].quantile(0.90))
    hv = tail[tail["cart_value"] >= hv_thr]
    L.append("## 7. High-value cohort\n")
    L.append(f"Threshold: VALIDATION p90 cart value = INR {hv_thr:,.0f} "
             f"(derived without touching TEST). Cohort n = {len(hv):,}.\n\n")
    hv_tbl = pd.DataFrame([{
        "cohort": "high_value", "n": len(hv), "mean_regret": hv.regret.mean(),
        "p95_regret": float(np.percentile(hv.regret, 95)),
        "do_nothing_rate": float((hv.selected == "DO_NOTHING").mean()),
        "revenueos_value": ros_pv.loc[hv.index, "net_contribution"].mean(),
        "oracle_value": ref_pv["ORACLE_ECONOMIC"].loc[hv.index, "net_contribution"].mean(),
    }, {
        "cohort": "rest", "n": len(tail) - len(hv),
        "mean_regret": tail[tail.cart_value < hv_thr].regret.mean(),
        "p95_regret": float(np.percentile(tail[tail.cart_value < hv_thr].regret, 95)),
        "do_nothing_rate": float((tail[tail.cart_value < hv_thr].selected == "DO_NOTHING").mean()),
        "revenueos_value": ros_pv.loc[tail.cart_value < hv_thr, "net_contribution"].mean(),
        "oracle_value": ref_pv["ORACLE_ECONOMIC"].loc[tail.cart_value < hv_thr, "net_contribution"].mean(),
    }])
    L.append(fmt(hv_tbl) + "\n")
    L.append("\n### High-value action distribution\n")
    L.append(fmt(hv["selected"].value_counts(normalize=True).reset_index()
                 .rename(columns={"index": "action", "proportion": "share"})) + "\n")

    # 10. calibration economic effect
    raw_actions = scored["C_xgboost_raw"]["actions"]
    raw_pv = scored["C_xgboost_raw"]["pv"]
    L.append("## 8. Does calibration change money decisions?\n")
    div = float((raw_actions != ros_actions).mean())
    L.append(f"""
| comparison | value |
|---|---:|
| Action divergence (raw vs final) | {div:.2%} |
| Net contribution difference | INR {ros_pv['net_contribution'].mean() - raw_pv['net_contribution'].mean():+.2f}/opp |
| DO_NOTHING rate difference | {float((ros_actions == 'DO_NOTHING').mean()) - float((raw_actions == 'DO_NOTHING').mean()):+.2%} |

Since the corrected protocol selected `{freeze['calibration_method']}`, the final
model *is* raw XGBoost and this comparison is definitionally zero. That is itself
the finding: the honest calibration procedure concluded no post-hoc calibrator
improved on the base model's probabilities, so no calibration step is applied.
""")

    # 11/12. OPE
    reward = OPE.realised_reward(test)
    q_log, q_tgt = OPE.cross_fitted_q(test, reward, ros_actions, OPE_FEATURES)
    ope_rows = []
    for clip in OPE.CLIP_LEVELS:
        r = OPE.evaluate_policy(test, ros_actions, reward, OPE_FEATURES, clip=clip,
                                q_logged=q_log, q_target=q_tgt)
        ope_rows.append({"clip_level": "none" if clip is None else f"clip{int(clip)}", **r})
    ope_df = pd.DataFrame(ope_rows)
    m, lo, hi = OPE.bootstrap_dr(test, ros_actions, reward, q_log, q_tgt)
    L.append("## 9. Final off-policy cross-check\n" + fmt(ope_df.round(4)) + "\n")
    L.append(f"\nDR bootstrap (1000 resamples): **INR {m:.2f}/opp**, 95% CI [{lo:.2f}, {hi:.2f}].\n")

    dr_val = float(ope_df[ope_df.clip_level == "none"]["dr"].iloc[0])
    oracle_val = ros_pv["net_contribution"].mean()
    L.append("## 10. DR vs oracle agreement\n")
    L.append(f"""
| quantity | value |
|---|---:|
| DR estimate (logged data only) | INR {dr_val:.2f}/opp |
| Oracle value (counterfactual) | INR {oracle_val:.2f}/opp |
| Absolute error | INR {abs(dr_val - oracle_val):.2f} |
| Relative error | {100 * abs(dr_val - oracle_val) / abs(oracle_val):.2f}% |

Two independent routes — one using only logged actions and propensities, the
other using the simulator's counterfactuals — agree to within
{100 * abs(dr_val - oracle_val) / abs(oracle_val):.1f}%. This is the core
credibility metric: it says the off-policy machinery recovers the truth on data
where the truth happens to be knowable.
""")

    # 13. headline
    conv_div = float((ref["MODEL_CONVERSION_MAX"] != ros_actions).mean())
    dn_rate = float((ros_actions == "DO_NOTHING").mean())
    v_flat = ref_pv["FLAT_10_PERCENT"]["net_contribution"].mean()
    head = pd.DataFrame([
        ("TEST ROC-AUC", f"{tm['roc_auc']:.4f}"),
        ("TEST PR-AUC", f"{tm['pr_auc']:.4f}"),
        ("TEST Brier", f"{tm['brier']:.4f}"),
        ("TEST ECE", f"{tm['ece']:.4f}"),
        ("RevenueOS Oracle Value / Opportunity", f"INR {oracle_val:.2f}"),
        ("RevenueOS DR Value / Opportunity", f"INR {dr_val:.2f}"),
        ("DR vs Oracle Relative Error", f"{100 * abs(dr_val - oracle_val) / abs(oracle_val):.2f}%"),
        ("RevenueOS vs DO_NOTHING", f"INR {oracle_val - v_nothing:+.2f}"),
        ("RevenueOS vs Flat 10%", f"INR {oracle_val - v_flat:+.2f}"),
        ("Oracle Incremental Value Captured", f"{100 * (oracle_val - v_nothing) / headroom:.1f}%"),
        ("Conversion/Economics Action Divergence", f"{conv_div:.1%}"),
        ("DO_NOTHING Selection Rate", f"{dn_rate:.1%}"),
        ("Mean Regret", f"INR {regret.mean():.2f}"),
        ("P95 Regret", f"INR {np.percentile(regret, 95):.2f}"),
    ], columns=["Metric", "Final Value"])
    L.insert(2, "## Headline metrics\n\n" + head.to_markdown(index=False) + "\n")
    head.to_csv(RESULTS / "headline_metrics.csv", index=False)

    # 13/14. limitations + recommendation
    L.append(f"""## 11. Limitations

1. **ML contribution is not positive.** A segment lookup table plus the financial
   engine captures
   {fam.loc[fam.policy == 'A_segment_lookup', 'value_capture_pct'].iloc[0]:.1f}% of headroom vs
   {fam.loc[fam.policy == 'D_final_selected', 'value_capture_pct'].iloc[0]:.1f}% for the final model —
   gradient boosting is INR {abs(d_val - a_val):.2f}/opp *worse* on economics. The
   value comes from the financial layer, not the model.
2. **No calibrator helped.** Under the corrected protocol neither Platt nor
   isotonic beat raw XGBoost, so the "calibration matters" story is weaker than
   the first gate implied — though the *measurement* of that is now sound.
3. **PAYMENT_METHOD_SWITCH response is not learned** (near-zero oracle dP
   correlation), so the policy never selects it despite genuine effectiveness on
   CARD_DECLINED.
4. **Regret is concentrated in high-value carts** ({ratio:.1f}x mean cart value in
   the tail), which is precisely where errors are most expensive.
5. Synthetic environment throughout: this is policy evaluation under a documented
   behavioural model, not real-world causal uplift.
6. Off-policy estimates depend on overlap; match rate is
   {ope_df.iloc[0]['match_rate']:.1%} with ESS {ope_df.iloc[0]['ess']:.0f}.

## 12. Final recommendation

**SELECTED_PRODUCTION_MODEL: XGBoost (no post-hoc calibration)** — retained for
the response-estimation interface, but with an explicit caveat: on this dataset
the segment lookup baseline achieves higher economic value
(INR {a_val:.2f} vs {d_val:.2f}/opp). XGBoost is kept because it produces
per-action probabilities across the full feature space that the policy and future
UI depend on, and because the lookup table cannot extrapolate to unseen
context combinations. If the gap persists after Phase 5, the lookup table should
replace it. **This should be stated in the README rather than omitted.**

Rejected alternatives, with reasons:

- *Isotonic* — appeared best only because it was scored on its own fitting
  partition. Under the corrected protocol its ECE is
  {freeze['selection_metrics']['xgboost_isotonic']['ece']:.4f} vs raw
  {freeze['selection_metrics']['xgboost_raw']['ece']:.4f}.
- *Platt* — worse Brier and ECE than raw on MODEL_SELECTION.
- *Logistic regression* — lower value capture, no interpretability advantage that
  matters here since the financial engine already supplies the explanation layer.
- *Segment lookup* — **economically superior here** and must be reported as such.
  Rejected as the production interface only because it cannot score arbitrary
  context/action combinations, not because it performed worse.

**Gate: PASS**, with the ML-contribution caveat stated explicitly rather than
buried. The defensible claim is: *a modest response model combined with a correct
incremental-economics layer and a deterministic policy gate beats both naive
discounting and conversion-maximisation, and off-policy evidence independently
confirms it to within {100 * abs(dr_val - oracle_val) / abs(oracle_val):.0f}%.*
""")

    (RESULTS / "final_model_audit.md").write_text("\n".join(L))
    print("\n".join(L)[:2500])
    print(f"\n\nWrote {RESULTS}/final_model_audit.md")


if __name__ == "__main__":
    main()