"""Cross-check the numbers the paper's figures and captions state against live cdm computations.

uv run python scripts/verify_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from cdm import Competitor, Lottery, Policy, expected_improvement, p_best
from cdm.belief import temper
from cdm.decide import branches

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

sys.exit(1 if failures else 0)
