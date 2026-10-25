"""Jev in the loop: the sequential benchmark on Jev's real mechanism judgments.

Uses the 104 options of scripts/mechanism_disguise.py (4 problems; 6 mechanisms besides the
status quo, each in up to four wordings). Everything the policies see about mechanisms is what Jev
answered for that exact text: the classification π, the failure estimates r and the judged
novelty. What Jev cannot know is simulated, as in scripts/mechanism_benchmark.py: whether each
principle works (θ ~ Beta(1, 1) for principles no precedent used, including each NEW one; 0.95 for
principles precedents used), each mechanism's value if it works (U(0.4, 0.6) for repeated
mechanisms, U(0.3, 0.9) for new ones, the same for every wording), and noisy, overconfident value
beliefs. Each decision offers the status quo and five of the six mechanisms, each in a random
wording, so the same principle reappears under different words.

Policies (status quo 0.5, pilot share 0.05, 24 decisions, 300 worlds per problem):

    v0.2           judged novelty; the mechanism check uses only Jev's r (q = 1 − Π(1 − r_f));
                   no shared beliefs
    veto untested  only options whose modal principles are all settled by precedent
    v0.3, no learning      ν, settled principles, Jev's r for the rest; beliefs never updated
    v0.3, learning         as above, updating per principle after every test
    v0.3, commit only      learning, no innovation pilot

    uv run python scripts/mechanism_jevloop.py     # needs paper_figs/disguise_raw.json
"""

from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

import mechanism_benchmark as mb
import numpy as np
from mechanism_disguise import rows
from mechanism_jev import DOMAINS, KIND, morphology

from cdm import NEW, Dimension, Judgment, Morphology, Policy, Problem, decide
from cdm.belief import temper
from cdm.decide import MECHANISM
from cdm.simplex import closure

OUT = Path("paper_figs/mechanism_jevloop.json")
SEED = 20261027
RUNS = int(sys.argv[1]) if __name__ == "__main__" and len(sys.argv) > 1 else 300
T, B, EPS = 24, 0.5, 0.05
POLICIES = (
    "v0.2",
    "veto untested",
    "v0.3, no learning",
    "v0.3, learning",
    "v0.3, commit only",
    "v0.3 without settled rule, no learning",
    "v0.3 without settled rule, learning",
)


class Unsettled(Morphology):
    """Ablation: precedents count for novelty but not as proof that a principle works."""

    def settled(self, fid: str, p: str) -> bool:  # noqa: ARG002
        return False


def morph_for(name: str, di: int) -> Morphology:
    d = DOMAINS[di]
    cls = Unsettled if "without settled" in name else Morphology
    return cls(list(d["functions"]), [dict(r) for r in d["record"]], settle="precedent")


def truth_key(di: int, mi: int, f: str) -> str:
    p = DOMAINS[di]["mechanisms"][mi][1][f]
    return f"new:{mi}" if p == NEW else p


def world(rng: np.random.Generator, di: int, m: Morphology) -> mb.World:
    theta, mu = {}, {}
    for f in m.functions:
        for p in f.principles:
            theta[(f.id, p)] = 0.95 if m.uses(f.id, p) else float(rng.beta(1, 1))
        for mi in range(7):
            if DOMAINS[di]["mechanisms"][mi][1][f.id] == NEW:
                theta[(f.id, f"new:{mi}")] = float(rng.beta(1, 1))
    for mi in range(7):
        lo, hi = (0.4, 0.6) if KIND[mi] == "precedent" else (0.3, 0.9)
        mu[("mech", str(mi))] = float(rng.uniform(lo, hi))
    return mb.World(theta, mu)


def cands_for(
    rng: np.random.Generator,
    di: int,
    w: mb.World,
    m: Morphology,
    variants: dict[int, list[dict[str, Any]]],
) -> list[mb.Cand]:
    base_choice = {f.id: truth_key(di, 0, f.id) for f in m.functions}
    out = [mb.Cand(base_choice, B, mb.level_dist(B), {}, {}, 0.0, baseline=True)]
    for mi in rng.choice(np.arange(1, 7), size=5, replace=False):
        v = variants[int(mi)][int(rng.integers(len(variants[int(mi)])))]
        value = float(np.clip(w.mu[("mech", str(mi))] + rng.normal(0, 0.05), 0, 1))
        judged = temper(closure(mb.level_dist(value) * np.exp(rng.normal(0, 0.3, 5))), 1 / 1.3)
        choice = {f.id: truth_key(di, int(mi), f.id) for f in m.functions}
        pi = {f: np.array(x) for f, x in v["pi"].items()}
        out.append(mb.Cand(choice, value, judged, pi, dict(v["risk"]), float(v["judged"])))
    return out


def judgments_v02(cands: list[mb.Cand], m: Morphology) -> list[Judgment]:
    out = []
    for j, c in zip(mb.judgments(cands, m), cands, strict=True):
        q = 0.0 if c.baseline else 1 - float(np.prod([1 - r for r in c.risk.values()]))
        out.append(replace(j, mechanism={}, mechanism_risk={}, violations={MECHANISM.id: q}))
    return out


def choose(name: str, cands: list[mb.Cand], m: Morphology) -> tuple[int, bool, int | None]:
    pol = Policy(
        downside_tolerance=mb.TAU,
        check_sensitivity=mb.SENS,
        check_specificity=mb.SPEC,
        check_cost=mb.COST,
        harm_value=mb.HARM,
    )
    dims = [Dimension("value", "q", mb.LEVELS, 1.0)]
    if name == "v0.2":
        pr = Problem("g", {}, [], dims, constraints=[MECHANISM])
        d = decide(pr, judgments_v02(cands, m), pol)
        ci = int(d.commit[1:])
        ii = int(d.innovation[1:]) if d.innovation else None
        return ci, d.by_id(d.commit).status == "verify", ii
    spec = {
        "veto untested": mb.Spec("veto", learn=False, innovate=False, veto_untested=True),
        "v0.3, no learning": mb.Spec("nl", learn=False),
        "v0.3, learning": mb.Spec("l"),
        "v0.3, commit only": mb.Spec("c", innovate=False),
        "v0.3 without settled rule, no learning": mb.Spec("unl", learn=False),
        "v0.3 without settled rule, learning": mb.Spec("ul"),
    }[name]
    return mb.choose(spec, cands, m)


def run_domain(di: int) -> dict[str, Any]:
    rng = np.random.default_rng(SEED + di)
    variants: dict[int, list[dict[str, Any]]] = {}
    for r in rows():
        if r["domain"] == di:
            variants.setdefault(r["mech"], []).append(r)
    funcs_m = morphology(DOMAINS[di])
    keys = ("regret", "harm", "bad")
    per = {p: {k: np.zeros((RUNS, T)) for k in keys} for p in POLICIES}
    for run in range(RUNS):
        w = world(rng, di, funcs_m)
        decisions = [cands_for(rng, di, w, funcs_m, variants) for _ in range(T)]
        zs = [
            [{k: bool(rng.random() < th) for k, th in w.theta.items()} for _ in range(6)]
            for _ in range(T)
        ]
        us = [
            [{f.id: float(rng.random()) for f in funcs_m.functions} for _ in range(6)]
            for _ in range(T)
        ]
        bad = {k for k, th in w.theta.items() if th < 0.3}
        for name in POLICIES:
            m = morph_for(name, di)
            learn = name.endswith(", learning") or name == "v0.3, commit only"
            for t, cands in enumerate(decisions):
                ci, checked, ii = choose(name, cands, m)
                if name in ("veto untested", "v0.3, commit only"):
                    ii = None
                ev, harm = mb.expected(w, cands[ci], B, checked=checked)
                if ii is not None:
                    ei, hi = mb.expected(w, cands[ii], B, checked=True)
                    ev, harm = (1 - EPS) * ev + EPS * ei, (1 - EPS) * harm + EPS * hi
                per[name]["regret"][run, t] = mb.oracle(w, cands, B) - ev
                per[name]["harm"][run, t] = harm
                if not cands[ci].baseline:
                    per[name]["bad"][run, t] = bool(set(cands[ci].choice.items()) & bad)
                if learn:
                    if not cands[ci].baseline:
                        mb.observe(m, cands[ci], zs[t][ci], us[t][ci], checked=checked)
                    if ii is not None:
                        mb.observe(m, cands[ii], zs[t][ii], us[t][ii], checked=True)
        if (run + 1) % 100 == 0:
            print(f"  problem {di}: {run + 1}/{RUNS}", file=sys.stderr)
    return {p: {k: v.tolist() for k, v in per[p].items()} for p in POLICIES}


def summarise(results: list[dict[str, Any]]) -> dict[str, Any]:
    reg = {p: np.concatenate([np.array(r[p]["regret"]) for r in results]) for p in POLICIES}
    out: dict[str, Any] = {"runs_per_problem": RUNS, "problems": len(results), "policies": {}}
    ref = reg["v0.2"].mean(axis=1)
    for p in POLICIES:
        per_run = reg[p].mean(axis=1)
        diff = ref - per_run
        half = 1.96 * diff.std(ddof=1) / np.sqrt(diff.size) if p != "v0.2" else 0.0
        harm = np.concatenate([np.array(r[p]["harm"]) for r in results])
        bad = np.concatenate([np.array(r[p]["bad"]) for r in results])
        out["policies"][p] = {
            "regret": float(per_run.mean()),
            "regret_se": float(per_run.std(ddof=1) / np.sqrt(per_run.size)),
            "harm": float(harm.mean()),
            "bad_last8": float(bad[:, -8:].mean()),
            "gain_vs_v02": float(diff.mean()),
            "gain_vs_v02_ci": [float(diff.mean() - half), float(diff.mean() + half)],
            "by_problem": [float(np.array(r[p]["regret"]).mean()) for r in results],
        }
    a = reg["v0.3, learning"].mean(axis=1)
    b = reg["v0.3, no learning"].mean(axis=1)
    c = reg["v0.3, commit only"].mean(axis=1)
    u_nl = reg["v0.3 without settled rule, no learning"].mean(axis=1)
    u_l = reg["v0.3 without settled rule, learning"].mean(axis=1)
    v02 = reg["v0.2"].mean(axis=1)
    for label, x, y in (
        ("learning_vs_no_learning", a, b),
        ("pilot_vs_commit_only", a, c),
        ("settled_rule", b, u_nl),
        ("nu_and_record_without_settled_vs_v02", u_nl, v02),
        ("learning_without_settled", u_l, u_nl),
    ):
        d = y - x
        h = 1.96 * d.std(ddof=1) / np.sqrt(d.size)
        out[label] = {"mean": float(d.mean()), "ci": [float(d.mean() - h), float(d.mean() + h)]}
    return out


def main() -> None:
    with ProcessPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(run_domain, range(len(DOMAINS))))
    res = summarise(results)
    OUT.write_text(json.dumps(res, indent=1))
    for p, r in res["policies"].items():
        print(
            f"{p:20s} regret {r['regret']:.4f}±{r['regret_se']:.4f} harm {r['harm']:.3f} "
            f"bad {r['bad_last8']:.2f} gain vs v0.2 {r['gain_vs_v02']:+.4f} "
            f"[{r['gain_vs_v02_ci'][0]:+.4f},{r['gain_vs_v02_ci'][1]:+.4f}]"
        )
    for k in (
        "learning_vs_no_learning",
        "pilot_vs_commit_only",
        "settled_rule",
        "nu_and_record_without_settled_vs_v02",
        "learning_without_settled",
    ):
        print(k, res[k])


if __name__ == "__main__":
    main()
