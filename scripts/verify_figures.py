"""Cross-check the numbers the paper's figures and captions state against live cdm computations.

uv run python scripts/verify_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from cdm import Beta, Competitor, Lottery, Policy, expected_improvement, p_best
from cdm.belief import temper
from cdm.decide import branches
from cdm.mechanism import next_decision_value, next_decision_value_bound

failures = 0


def check(label: str, *, ok: bool, detail: str = "") -> None:
    global failures  # noqa: PLW0603
    failures += not ok
    print(f"[{'PASS' if ok else 'FAIL'}] {label}" + (f"  :: {detail}" if detail else ""))


# fig_temper: tempering keeps a distribution, keeps ranks, and flattens toward uniform
p = np.array([0.04, 0.08, 0.18, 0.30, 0.40])
prev = np.inf
for t in (1.0, 2.0, 4.0, 8.0):
    q = temper(p, t)
    spread = float(q.max() - q.min())
    check(
        f"temper T={t}: sums to 1, ranks kept, flatter",
        ok=abs(q.sum() - 1) < 1e-12 and bool(np.all(np.diff(q) > 0)) and spread < prev,
    )
    prev = spread

# fig_cvar: the bold option is not reliable at α = 0.2, τ = 0.05
bold = Lottery.of([0.0, 0.25, 0.5, 0.75, 1.0], [0.08, 0.12, 0.25, 0.30, 0.25])
base = Lottery.of([0.25, 0.5, 0.75], [0.2, 0.6, 0.2])
check("fig_cvar: bold has the higher mean", ok=bold.mean() > base.mean())
check(
    "fig_cvar: bold fails CVaR_0.2 ≥ CVaR_0.2(B) − 0.05",
    ok=bold.cvar(0.2) < base.cvar(0.2) - 0.05,
    detail=f"{bold.cvar(0.2):.3f} vs {base.cvar(0.2) - 0.05:.3f}",
)

# fig_ties: the fair split sums to 1 for every n
one = Lottery.point(1.0)
check(
    "fig_ties: fair split sums to 1, each gets 1/n",
    ok=all(np.allclose(p_best([Competitor(one)] * n), 1 / n) for n in range(1, 9)),
)

# fig_ei: EI of 0.5 ± w/2 over B = 0.5 is (1 − f)·w/4
b = Lottery.point(0.5)
check(
    "fig_ei: EI = (1−f)·w/4",
    ok=all(
        abs(
            expected_improvement(Lottery.of([0.5 - w / 2, 0.5 + w / 2], [0.5, 0.5]), b, f)
            - (1 - f) * w / 4
        )
        < 1e-12
        for w in (0.2, 0.6, 1.0)
        for f in (0.0, 0.3, 0.6)
    ),
)

# fig_verify: residual mass f(1 − s) at H = 0
v = Lottery.of([0.25, 0.5, 0.75, 1.0], [0.1, 0.2, 0.3, 0.4])
for s, want in ((1.0, 0.0), (0.9, 0.05), (0.7, 0.15)):
    u, _, _, res = branches(v, b, 0.5, Policy(check_sensitivity=s), verify=True)
    check(
        f"fig_verify s={s}: P(U=H) = {want}",
        ok=abs(res - want) < 1e-12 and abs(float(u.cdf(0.0)) - want) < 1e-9,
    )

# fig_dependence: 1 − 0.8³ = 0.488 inside the Fréchet band [0.2, 0.6]
f = 1 - 0.8**3
check("fig_dependence: 0.488 within [0.2, 0.6]", ok=abs(f - 0.488) < 1e-12 and 0.2 <= f <= 0.6)


# Every number quoted from the benchmarks ---------------------------------------------------------
def near(x: float, want: float, dp: int) -> bool:
    return round(x, dp) == round(want, dp)


exp = json.loads(Path("paper_figs/experiments.json").read_text())
bm = exp["benchmark"]
check("synthetic: tau* = 0.2 on the tuning seed", ok=bm["tau_star"] == "0.2")
quoted = {  # name: (regret, harm, explores, gain)
    "raw judge": (0.144, 0.233, 0.68, -0.054),
    "accept uncertain": (0.106, 0.167, 0.71, -0.015),
    "veto uncertain": (0.091, 0.0, 0.0, 0.0),
    "cdm, perfect-check model": (0.068, 0.009, 0.38, 0.023),
    "cdm, check model, default tau": (0.106, 0.001, 0.04, -0.016),
    "cdm, check model, tuned tau": (0.045, 0.012, 0.48, 0.045),
    "expected utility + checks (no CVaR gate)": (0.045, 0.012, 0.48, 0.045),
}
for name, (reg, harm, nov, gain) in quoted.items():
    d = bm["table"][name]
    check(
        f"synthetic table: {name}",
        ok=near(d["regret"], reg, 3)
        and near(d["harm"], harm, 3)
        and near(d["novel"], nov, 2)
        and near(d["gain_vs_ref"], gain, 3),
        detail=f"{d['regret']:.4f} {d['harm']:.4f} {d['novel']:.3f} {d['gain_vs_ref']:+.4f}",
    )
fr = bm["frontier"]
check(
    "synthetic: tau=0.1 regret 0.073, harm 0.6%",
    ok=near(fr["0.1"]["regret"], 0.073, 3) and near(fr["0.1"]["harm"], 0.006, 3),
)
check(
    "synthetic: tau >= 0.2 is the no-gate policy",
    ok=fr["0.2"]["regret"] == fr["inf"]["regret"] == fr["0.3"]["regret"],
)
hs = bm["harm_by_sens"]
check(
    "synthetic: perfect-check harm 0.9% at s=0.9, 4.6% at s=0.5",
    ok=near(hs["cdm, perfect-check model"][4], 0.009, 3)
    and near(hs["cdm, perfect-check model"][0], 0.046, 3),
)
check(
    "synthetic: check-model harm within 1.2-2.2% for s<1",
    ok=all(0.0115 <= x <= 0.0225 for x in hs["cdm, check model, tuned tau"][:-1]),
)
rb = bm["robustness"]
for key, (c_reg, v_reg, gain) in {
    "confidently wrong judge (20%)": (0.066, 0.116, 0.051),
    "specificity 0.8, cost 0.01": (0.056, 0.091, 0.035),
    "specificity 0.95, cost 0.05": (0.069, 0.091, 0.022),
    "specificity 0.8, cost 0.05": (0.076, 0.091, 0.014),
    "assumed T = 1.0": (0.045, 0.091, 0.046),
    "assumed T = 2.0": (0.048, 0.091, 0.043),
}.items():
    r = rb[key]
    check(
        f"robustness: {key}",
        ok=near(r["cdm"]["regret"], c_reg, 3)
        and near(r["veto"]["regret"], v_reg, 3)
        and near(r["cdm"]["gain_vs_ref"], gain, 3),
    )
b4 = exp["b4"]
check(
    "B4: median 0.017, p95 0.061 at eta=0.03",
    ok=near(b4["median_err"], 0.017, 3) and near(b4["p95_err"], 0.061, 3),
)
mc = {r["option"]: r for r in exp["exact_vs_mc"]}
check(
    "MC table: baseline E exact 0.5550, MC 0.5552",
    ok=near(mc["base"]["mean_exact"], 0.555, 4) and near(mc["base"]["mean_mc"], 0.5552, 4),
)
check(
    "MC: every mean within 3 se",
    ok=all(abs(r["mean_exact"] - r["mean_mc"]) < 3 * r["mean_se"] for r in exp["exact_vs_mc"]),
)

jev = json.loads(Path("paper_figs/jev_benchmark.json").read_text())
h = jev["harm"]
check(
    "Jev: mean q 0.93 / 0.15 / 0.34, ambiguous truth 0.37",
    ok=near(h["harm"]["mean_q"], 0.93, 2)
    and near(h["safe"]["mean_q"], 0.15, 2)
    and near(h["ambiguous"]["mean_q"], 0.34, 2)
    and near(h["ambiguous"]["true_rate"], 0.37, 2),
)
check(
    "Jev: AUROC 0.9998 / 0.92, Brier 0.104, ECE 0.087 -> 0.074",
    ok=near(h["auroc_explicit"], 0.9998, 4)
    and near(h["auroc_all"], 0.92, 2)
    and near(h["brier_raw"], 0.104, 3)
    and near(h["ece_raw"], 0.087, 3)
    and near(h["ece_default_cal"], 0.074, 3),
)
safe_q = np.array(jev["q_by_class"]["safe"])
check("Jev: 21% of safe options above delta", ok=near(float((safe_q > 0.15).mean()), 0.21, 2))
v = jev["value"]
check(
    "Jev: top-1 50% (63% with evidence), within one 88%",
    ok=near(v["top1_accuracy"], 0.50, 2)
    and near(v["top1_accuracy_with_evidence"], 0.63, 2)
    and near(v["within_one_level"], 0.88, 2),
)
check(
    "Jev: held-out NLL 1.88 / 1.57 / 1.31, fitted T 2.2-2.5",
    ok=near(v["heldout_nll"]["T=1"], 1.88, 2)
    and near(v["heldout_nll"]["T=1.3 (audit prior)"], 1.57, 2)
    and near(v["heldout_nll"]["fitted on other fold"], 1.31, 2)
    and all(2.2 <= t <= 2.55 for t in v["fitted_T_per_fold"]),
)
check("Jev: direction gap 0.03 levels", ok=near(v["mean_abs_gap_levels"], 0.03, 2))
check(
    "Jev: cdm variants identical; gate excludes 3",
    ok=jev["cdm_variants_identical"] and jev["gate_excluded_options"] == 3,
)
for name, (reg, harm, amb, gain) in {
    "raw judge": (0.146, 0.188, 0.29, -0.010),
    "accept uncertain": (0.099, 0.104, 0.35, 0.036),
    "veto uncertain": (0.135, 0.042, 0.17, 0.0),
    "cdm, check model, tuned tau": (0.084, 0.031, 0.31, 0.051),
}.items():
    d = jev["policies"][name]
    check(
        f"Jev table: {name}",
        ok=near(d["regret"], reg, 3)
        and near(d["harm"], harm, 3)
        and near(d["ambiguous_commit"], amb, 2)
        and near(d["gain_vs_veto"], gain, 3),
    )
ci = jev["policies"]["cdm, check model, tuned tau"]["gain_vs_veto_ci"]
check("Jev: gain CI [0.009, 0.102]", ok=near(ci[0], 0.009, 3) and near(ci[1], 0.102, 3))
bs = jev["baseline_truth_sensitivity"]
g1 = bs["1"]["cdm, check model, tuned tau"]
g3 = bs["3"]["cdm, check model, tuned tau"]
check(
    "Jev: baseline level 1 gain 0.083 [0.025, 0.148]; level 3 gain 0.020 [-0.019, 0.067]",
    ok=near(g1["gain_vs_veto"], 0.083, 3)
    and near(g1["gain_vs_veto_ci"][0], 0.025, 3)
    and near(g1["gain_vs_veto_ci"][1], 0.148, 3)
    and near(g3["gain_vs_veto"], 0.020, 3)
    and near(g3["gain_vs_veto_ci"][0], -0.019, 3)
    and near(g3["gain_vs_veto_ci"][1], 0.067, 3),
)

# --- mechanism layer -----------------------------------------------------------------------
ratio = next_decision_value_bound(Beta(1, 1)) / next_decision_value_bound(Beta(50, 50))
check("M5: ignorance vs a known coin = 101/3 (about 34)", ok=near(ratio, 101 / 3, 9))
check(
    "fig_mech_voi: peak at t = mean equals the bound",
    ok=all(
        near(next_decision_value(Beta(a, a), 0.5), next_decision_value_bound(Beta(a, a)), 12)
        for a in (1, 5, 50)
    ),
)
mj = json.loads(Path("paper_figs/mechanism_jev.json").read_text())
check("mech Jev: 56 options, 28 mechanisms", ok=mj["n_options"] == 56 and mj["n_mechanisms"] == 28)
for key, (mean, lo, hi) in {
    "judged": (0.058, 0.038, 0.080),
    "nu": (0.005, -0.006, 0.017),
    "kappa": (0.007, -0.008, 0.025),
    "q_mech": (0.035, 0.018, 0.054),
}.items():
    e = mj[f"hype_effect_{key}"]
    check(
        f"mech Jev: hype effect on {key} {mean} [{lo}, {hi}]",
        ok=near(e["mean"], mean, 3) and near(e["ci"][0], lo, 3) and near(e["ci"][1], hi, 3),
    )
check(
    "mech Jev: MAE 0.122 / 0.044, Pearson 0.90 / 0.98",
    ok=near(mj["track_judged"]["mae"], 0.122, 3)
    and near(mj["track_nu"]["mae"], 0.044, 3)
    and near(mj["track_judged"]["pearson"], 0.90, 2)
    and near(mj["track_nu"]["pearson"], 0.98, 2),
)
cl = mj["classification"]
check(
    "mech Jev: 111/112 cells, 15/16 NEW, mean P(true) 0.87",
    ok=round(cl["accuracy"] * cl["n_cells"]) == 111
    and cl["n_cells"] == 112
    and near(cl["accuracy_new"] * 16, 15, 9)
    and near(cl["mean_p_true"], 0.87, 2),
)
bk = mj["by_kind"]["precedent"]
check(
    "mech Jev: repeated mechanisms 0.08 (judged) and 0.05 (nu)",
    ok=near((bk["plain"]["judged"] + bk["hype"]["judged"]) / 2, 0.08, 2)
    and near((bk["plain"]["nu"] + bk["hype"]["nu"]) / 2, 0.05, 2),
)
gj, gn = mj["gate_judged"], mj["gate_nu"]
check(
    "mech Jev: AUROC 1 for both; gate 0.5 for nu passes 50% / 65%; gate 0.25 separates",
    ok=gj["auroc"] == 1.0
    and gn["auroc"] == 1.0
    and near(gn["novel_plain_ge_0.5"], 0.50, 2)
    and near(gn["novel_hype_ge_0.5"], 0.65, 2)
    and all(
        gn[f"novel_{w}_ge_0.25"] == 1.0 and gn[f"known_{w}_ge_0.25"] == 0.0
        for w in ("plain", "hype")
    ),
)
nu_true = [r["nu_true"] for r in mj["rows"] if r["kind"] != "precedent" and not r["hype"]]
check(
    "mech Jev: 18 of 20 novel mechanisms have nu* = 0.5",
    ok=sum(abs(x - 0.5) < 1e-9 for x in nu_true) == 18 and len(nu_true) == 20,
)

mb = json.loads(Path("paper_figs/mechanism_benchmark.json").read_text())


def cell(b: float, eps: float, horizon: int) -> dict:
    return next(g for g in mb["grid"] if (g["base_value"], g["eps"], g["T"]) == (b, eps, horizon))


check("mech bench: 300 runs", ok=mb["runs"] == 300)
table = {
    (0.5, 0.05, 24): {
        "veto untested principles": (0.0542, 0.0019, 0.041, 0.01, 0.01),
        "commit only, learning": (0.0260, 0.0007, 0.058, 0.12, 0.27),
        "judged novelty, no learning": (0.0334, 0.0009, 0.064, 0.26, 0.22),
        "mechanism novelty, no learning": (0.0331, 0.0009, 0.064, 0.26, 0.22),
        "mechanism novelty + learning": (0.0250, 0.0007, 0.056, 0.07, 0.27),
        "+ learning bonus (w = 1)": (0.0246, 0.0007, 0.055, 0.07, 0.27),
    },
    (0.65, 0.05, 48): {
        "veto untested principles": (0.0083, 0.0006, 0.006, 0.00, 0.00),
        "commit only, learning": (0.0074, 0.0004, 0.012, 0.02, 0.08),
        "judged novelty, no learning": (0.0124, 0.0005, 0.017, 0.06, 0.05),
        "mechanism novelty, no learning": (0.0124, 0.0005, 0.017, 0.06, 0.05),
        "mechanism novelty + learning": (0.0085, 0.0003, 0.013, 0.01, 0.11),
        "+ learning bonus (w = 1)": (0.0089, 0.0004, 0.013, 0.01, 0.11),
    },
}
for cfg, rows in table.items():
    for name, (reg, se, harm, bad, good) in rows.items():
        d = cell(*cfg)["policies"][name]
        check(
            f"mech bench table {cfg}: {name}",
            ok=near(d["regret"], reg, 4)
            and near(d["regret_se"], se, 4)
            and near(d["harm"], harm, 3)
            and near(d["bad_last8"], bad, 2)
            and near(d["good_last8"], good, 2),
        )
learning = "mechanism novelty + learning"
learn_gain = [g["policies"][learning]["gain_vs_no_learning"] for g in mb["grid"]]
check(
    "mech bench: learning gains 0.0025..0.0081, all CIs above zero",
    ok=near(min(learn_gain), 0.0025, 4)
    and near(max(learn_gain), 0.0081, 4)
    and all(g["policies"][learning]["gain_ci"][0] > 0 for g in mb["grid"]),
)
a = cell(0.5, 0.05, 24)["policies"]
long_run = cell(0.65, 0.05, 48)["policies"]
check(
    "mech bench: learning gain 0.0081 [0.0068, 0.0094]",
    ok=near(a[learning]["gain_ci"][0], 0.0068, 4) and near(a[learning]["gain_ci"][1], 0.0094, 4),
)
check(
    "mech bench: bad late commitments 26% -> 7% (B=0.5) and 6% -> 1% (B=0.65)",
    ok=near(a["mechanism novelty, no learning"]["bad_last8"], 0.26, 2)
    and near(a[learning]["bad_last8"], 0.07, 2)
    and near(long_run["mechanism novelty, no learning"]["bad_last8"], 0.06, 2)
    and near(long_run[learning]["bad_last8"], 0.01, 2),
)
check(
    "mech bench: judged vs nu differ by at most 0.0009",
    ok=max(
        abs(
            g["policies"]["judged novelty, no learning"]["regret"]
            - g["policies"]["mechanism novelty, no learning"]["regret"]
        )
        for g in mb["grid"]
    )
    < 0.00095,
)


def vs_commit(cfg: tuple[float, float, int], name: str) -> list[float]:
    d = cell(*cfg)["policies"][name]
    return [d["gain_vs_commit_only"], *d["gain_vs_commit_only_ci"]]


for cfg, name, want in (
    ((0.5, 0.05, 24), learning, (0.0009, 0.0001, 0.0018)),
    ((0.5, 0.05, 24), "+ learning bonus (w = 1)", (0.0014, 0.0006, 0.0022)),
    ((0.5, 0.2, 24), learning, (-0.0034, -0.0043, -0.0025)),
    ((0.65, 0.05, 48), learning, (-0.0011, -0.0015, -0.0007)),
):
    got = vs_commit(cfg, name)
    check(
        f"mech bench vs commit-only {cfg} {name}: {want}",
        ok=all(near(x, y, 4) for x, y in zip(got, want, strict=True)),
    )
strong = [g for g in mb["grid"] if g["base_value"] == 0.65]
check(
    "mech bench: innovation slot loses to commit-only whenever B = 0.65",
    ok=all(g["policies"][learning]["gain_vs_commit_only_ci"][1] < 0 for g in strong),
)
check(
    "mech bench: veto within 0.0008 of commit-only and about half the harm at B = 0.65",
    ok=all(
        -0.00085 < g["policies"]["veto untested principles"]["gain_vs_commit_only"] <= 0
        and g["policies"]["veto untested principles"]["harm"]
        < 0.55 * g["policies"]["commit only, learning"]["harm"]
        for g in strong
    ),
)
check(
    "mech bench: w = 4 never beats w = 1",
    ok=all(
        g["policies"]["+ learning bonus (w = 4)"]["regret"]
        >= g["policies"]["+ learning bonus (w = 1)"]["regret"] - 1e-4
        for g in mb["grid"]
    ),
)
check(
    "mech bench: late good-principle share 0.08 -> 0.11 at B = 0.65, T = 48",
    ok=near(long_run["commit only, learning"]["good_last8"], 0.08, 2)
    and near(long_run[learning]["good_last8"], 0.11, 2),
)

md = json.loads(Path("paper_figs/mechanism_disguise.json").read_text())
check("disguise: 104 options", ok=md["n_options"] == 104)
for key, n, vals in (
    (
        "hype_all",
        28,
        {
            "judged": (0.055, 0.034, 0.078),
            "nu": (0.003, -0.005, 0.013),
            "q_mech": (0.036, 0.021, 0.051),
        },
    ),
    (
        "jargon_repeated",
        8,
        {
            "judged": (0.096, 0.053, 0.136),
            "nu": (0.043, 0.011, 0.083),
            "q_mech": (0.028, 0.004, 0.058),
        },
    ),
    (
        "jargon_novel",
        20,
        {
            "judged": (0.049, 0.036, 0.062),
            "nu": (0.031, -0.002, 0.064),
            "q_mech": (0.047, 0.010, 0.085),
        },
    ),
    (
        "mundane_novel",
        20,
        {
            "judged": (0.007, -0.005, 0.018),
            "nu": (0.016, 0.001, 0.039),
            "q_mech": (0.044, 0.011, 0.084),
        },
    ),
):
    for measure, (mean, lo, hi) in vals.items():
        e = md[f"{measure}_{key}"]
        check(
            f"disguise table: {key} {measure} {mean} [{lo}, {hi}]",
            ok=e["n"] == n
            and near(e["mean"], mean, 3)
            and near(e["ci"][0], lo, 3)
            and near(e["ci"][1], hi, 3),
        )
lv = md["levels"]
check(
    "disguise: repeated judged 0.08 -> 0.18, nu 0.05 -> 0.10 under jargon",
    ok=near(lv["repeated/plain"]["judged"], 0.08, 2)
    and near(lv["repeated/jargon"]["judged"], 0.18, 2)
    and near(lv["repeated/plain"]["nu"], 0.05, 2)
    and near(lv["repeated/jargon"]["nu"], 0.10, 2),
)
acc = md["accuracy"]
check(
    "disguise: accuracy 95% jargon, 98-100% otherwise",
    ok=near(acc["jargon"], 0.95, 2)
    and all(0.98 <= acc[w] <= 1.0 for w in ("plain", "hype", "mundane")),
)
for measure in ("judged", "nu"):
    g = md[f"gate_{measure}"]
    check(
        f"disguise: {measure} at 0.25 admits no repeated and all new under every wording; AUROC 1",
        ok=g["precision@0.25"] == 1.0 and g["recall@0.25"] == 1.0 and g["auroc"] == 1.0,
    )
check(
    "disguise: MAE 0.128 / 0.054; recall at 0.5: 86% judged, 56% nu",
    ok=near(md["gate_judged"]["mae"], 0.128, 3)
    and near(md["gate_nu"]["mae"], 0.054, 3)
    and near(md["gate_judged"]["recall@0.5"], 0.86, 2)
    and near(md["gate_nu"]["recall@0.5"], 0.56, 2),
)

mr = json.loads(Path("paper_figs/mechanism_robustness.json").read_text())


def rob(knob: str, value: object) -> dict:
    return next(r for r in mr["results"] if r["knob"] == knob and r["value"] == value)


check("robustness: 200 runs", ok=mr["runs"] == 200)
for (knob, value), (mean, lo, hi) in {
    ("p_correct", 1.0): (0.0117, 0.0101, 0.0133),
    ("default", "-"): (0.0076, 0.0060, 0.0091),
    ("p_correct", 0.7): (0.0033, 0.0016, 0.0050),
    ("p_correct", 0.5): (0.0010, -0.0006, 0.0025),
    ("interact", 0.5): (0.0064, 0.0051, 0.0077),
}.items():
    r = rob(knob, value)
    check(
        f"robustness: learning gain at {knob}={value} is {mean} [{lo}, {hi}]",
        ok=near(r["learning_gain"], mean, 4)
        and near(r["learning_gain_ci"][0], lo, 4)
        and near(r["learning_gain_ci"][1], hi, 4),
    )
check(
    "robustness: learning gain 0.0053 with 1 untested principle, 0.0058 with 12",
    ok=near(rob("untested", 1)["learning_gain"], 0.0053, 4)
    and near(rob("untested", 12)["learning_gain"], 0.0058, 4),
)
r7 = rob("p_correct", 0.7)
check(
    "robustness: pilot vs commit 0.0018 [0.0004, 0.0033] at accuracy 0.7",
    ok=near(r7["pilot_vs_commit"], 0.0018, 4)
    and near(r7["pilot_vs_commit_ci"][0], 0.0004, 4)
    and near(r7["pilot_vs_commit_ci"][1], 0.0033, 4),
)
check(
    "robustness: pilot gain within [0, 0.002] everywhere",
    ok=all(0 <= r["pilot_vs_commit"] <= 0.0021 for r in mr["results"]),
)
sig = {(r["knob"], r["value"]) for r in mr["results"] if r["pilot_vs_commit_ci"][0] > 0}
check(
    "robustness: pilot interval excludes 0 exactly at accuracy 0.5, 0.7, 0.85, 1",
    ok=sig == {("p_correct", 0.5), ("p_correct", 0.7), ("p_correct", 0.85), ("p_correct", 1.0)},
    detail=str(sorted(sig, key=str)),
)

jl = json.loads(Path("paper_figs/mechanism_jevloop.json").read_text())
check("Jev loop: 4 problems x 300 worlds", ok=jl["problems"] == 4 and jl["runs_per_problem"] == 300)
for name, (reg, se, harm, bad, gain, lo, hi) in {
    "v0.2": (0.0742, 0.0019, 0.033, 0.18, 0.0, 0.0, 0.0),
    "veto untested": (0.1175, 0.0022, 0.038, 0.02, -0.0433, -0.0472, -0.0393),
    "v0.3 without settled rule, no learning": (
        0.0745,
        0.0019,
        0.033,
        0.18,
        -0.0003,
        -0.0003,
        -0.0002,
    ),
    "v0.3 without settled rule, learning": (0.0607, 0.0018, 0.026, 0.10, 0.0135, 0.0112, 0.0158),
    "v0.3, no learning": (0.0489, 0.0013, 0.053, 0.24, 0.0253, 0.0228, 0.0279),
    "v0.3, learning": (0.0370, 0.0011, 0.050, 0.13, 0.0372, 0.0340, 0.0405),
    "v0.3, commit only": (0.0374, 0.0012, 0.050, 0.15, 0.0369, 0.0336, 0.0401),
}.items():
    d = jl["policies"][name]
    check(
        f"Jev loop table: {name}",
        ok=near(d["regret"], reg, 4)
        and near(d["regret_se"], se, 4)
        and near(d["harm"], harm, 3)
        and near(d["bad_last8"], bad, 2)
        and near(d["gain_vs_v02"], gain, 4)
        and near(d["gain_vs_v02_ci"][0], lo, 4)
        and near(d["gain_vs_v02_ci"][1], hi, 4),
    )
for key, (mean, lo, hi) in {
    "settled_rule": (0.026, 0.023, 0.028),
    "learning_vs_no_learning": (0.012, 0.010, 0.014),
    "learning_without_settled": (0.014, 0.011, 0.016),
    "pilot_vs_commit_only": (0.0004, -0.0006, 0.0013),
}.items():
    e = jl[key]
    dp = 4 if key == "pilot_vs_commit_only" else 3
    check(
        f"Jev loop: {key} {mean} [{lo}, {hi}]",
        ok=near(e["mean"], mean, dp) and near(e["ci"][0], lo, dp) and near(e["ci"][1], hi, dp),
    )
pol = jl["policies"]
check(
    "Jev loop: full layer halves v0.2 regret and wins in every problem",
    ok=near(pol["v0.3, learning"]["regret"] / pol["v0.2"]["regret"], 0.5, 1)
    and all(
        a < b
        for a, b in zip(pol["v0.3, learning"]["by_problem"], pol["v0.2"]["by_problem"], strict=True)
    ),
)
check(
    "Jev loop: veto untested is the worst policy",
    ok=pol["veto untested"]["regret"] == max(r["regret"] for r in pol.values()),
)

sys.exit(1 if failures else 0)
