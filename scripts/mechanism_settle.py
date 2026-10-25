"""When should a precedent's working principle count as proven? (Jev in the loop)

A settled principle is deployed without a check. Version 0.3.0 settled every principle that a
precedent had used; the evidence rule settles one only if its recorded history puts its
reliability above 0.95 with 95% confidence (Cantelli). This compares the rules on Jev's real
mechanism judgments (the 104 options of scripts/mechanism_disguise.py), in a world where precedents
are not all equally reliable: each principle a precedent used works with θ = 0.995 (half of them) or
θ ~ U(0.7, 0.9). Each has a history of n past uses with Binomial(n, θ) successes, available as
evidence. Untested principles, values and value beliefs are as in scripts/mechanism_jevloop.py.
Every policy learns per principle after each test.

    precedent use (v0.3.0)   settle="precedent"; the history is not recorded
    evidence (default)       settle="evidence"; the history is recorded as evidence
    evidence, no history     settle="evidence"; nothing recorded, so nothing is settled at first
    never                    settle="never"; the history is recorded as evidence

    PYTHONPATH=scripts uv run python scripts/mechanism_settle.py
"""

from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import mechanism_benchmark as mb
import numpy as np
from mechanism_disguise import rows
from mechanism_jev import DOMAINS
from mechanism_jevloop import EPS, B, T, cands_for

from cdm import NEW, Beta, Morphology

OUT = Path("paper_figs/mechanism_settle.json")
SEED = 20261028
RUNS = int(sys.argv[1]) if __name__ == "__main__" and len(sys.argv) > 1 else 300
HISTORY = (30, 300)  # past uses of each precedent principle
RULES = ("precedent use (v0.3.0)", "evidence (default)", "evidence, no history", "never")


def world(rng: np.random.Generator, di: int, n_hist: int) -> tuple[mb.World, dict[Any, Beta]]:
    d = DOMAINS[di]
    m = Morphology(list(d["functions"]), [dict(r) for r in d["record"]])
    theta, mu, history = {}, {}, {}
    for f in m.functions:
        for p in f.principles:
            if m.uses(f.id, p):
                th = 0.995 if rng.random() < 0.5 else float(rng.uniform(0.7, 0.9))
                ok = int(rng.binomial(n_hist, th))
                history[(f.id, p)] = Beta(1.0 + ok, 1.0 + n_hist - ok)
            else:
                th = float(rng.beta(1, 1))
            theta[(f.id, p)] = th
        for mi in range(7):
            if d["mechanisms"][mi][1][f.id] == NEW:
                theta[(f.id, f"new:{mi}")] = float(rng.beta(1, 1))
    for mi in range(7):
        lo, hi = (0.4, 0.6) if mi < 2 else (0.3, 0.9)
        mu[("mech", str(mi))] = float(rng.uniform(lo, hi))
    return mb.World(theta, mu), history


def morph_for(rule: str, di: int, history: dict[Any, Beta]) -> Morphology:
    d = DOMAINS[di]
    settle = {"precedent use (v0.3.0)": "precedent", "never": "never"}.get(rule, "evidence")
    recorded = rule in ("evidence (default)", "never")
    return Morphology(
        list(d["functions"]),
        [dict(r) for r in d["record"]],
        evidence=dict(history) if recorded else {},
        settle=settle,
    )


def run_domain(job: tuple[int, int]) -> dict[str, Any]:
    di, n_hist = job
    rng = np.random.default_rng(SEED + 10 * di + n_hist)
    variants: dict[int, list[dict[str, Any]]] = {}
    for r in rows():
        if r["domain"] == di:
            variants.setdefault(r["mech"], []).append(r)
    keys = ("regret", "harm", "checked", "settled_bad", "settled_any")
    per = {p: {k: np.zeros((RUNS, T)) for k in keys} for p in RULES}
    spec = mb.Spec("learn")
    for run in range(RUNS):
        w, history = world(rng, di, n_hist)
        base_m = morph_for("never", di, history)
        decisions = [cands_for(rng, di, w, base_m, variants) for _ in range(T)]
        zs = [
            [{k: bool(rng.random() < th) for k, th in w.theta.items()} for _ in range(6)]
            for _ in range(T)
        ]
        us = [
            [{f.id: float(rng.random()) for f in base_m.functions} for _ in range(6)]
            for _ in range(T)
        ]
        for rule in RULES:
            m = morph_for(rule, di, history)
            rec = per[rule]
            for t, cands in enumerate(decisions):
                ci, checked, ii = mb.choose(spec, cands, m)
                ev, harm = mb.expected(w, cands[ci], B, checked=checked)
                if ii is not None:
                    ei, hi = mb.expected(w, cands[ii], B, checked=True)
                    ev, harm = (1 - EPS) * ev + EPS * ei, (1 - EPS) * harm + EPS * hi
                rec["regret"][run, t] = mb.oracle(w, cands, B) - ev
                rec["harm"][run, t] = harm
                rec["checked"][run, t] = checked
                c = cands[ci]
                rec["settled_bad"][run, t] = any(
                    m.settled(f, p) and w.theta[(f, p)] < 0.95 for f, p in c.choice.items()
                )
                rec["settled_any"][run, t] = any(m.settled(f, p) for f, p in c.choice.items())
                if not c.baseline:
                    mb.observe(m, c, zs[t][ci], us[t][ci], checked=checked)
                if ii is not None:
                    mb.observe(m, cands[ii], zs[t][ii], us[t][ii], checked=True)
        if (run + 1) % 100 == 0:
            print(f"  problem {di}, history {n_hist}: {run + 1}/{RUNS}", file=sys.stderr)
    return {
        "domain": di,
        "history": n_hist,
        **{p: {k: v.tolist() for k, v in per[p].items()} for p in RULES},
    }


def summarise(results: list[dict[str, Any]], n_hist: int) -> dict[str, Any]:
    sub = [r for r in results if r["history"] == n_hist]

    def cat(p: str, k: str) -> np.ndarray:
        return np.concatenate([np.array(r[p][k]) for r in sub])

    ref = cat("precedent use (v0.3.0)", "regret").mean(axis=1)
    ref_h = cat("precedent use (v0.3.0)", "harm").mean(axis=1)
    out: dict[str, Any] = {}
    for p in RULES:
        reg = cat(p, "regret").mean(axis=1)
        harm = cat(p, "harm").mean(axis=1)
        d, dh = ref - reg, ref_h - harm
        h = 1.96 * d.std(ddof=1) / np.sqrt(d.size)
        hh = 1.96 * dh.std(ddof=1) / np.sqrt(dh.size)
        out[p] = {
            "regret": float(reg.mean()),
            "regret_se": float(reg.std(ddof=1) / np.sqrt(reg.size)),
            "harm": float(harm.mean()),
            "checked": float(cat(p, "checked").mean()),
            "settled_unreliable": float(cat(p, "settled_bad").mean()),
            "settled_any": float(cat(p, "settled_any").mean()),
            "regret_gain_vs_precedent": float(d.mean()),
            "regret_gain_ci": [float(d.mean() - h), float(d.mean() + h)],
            "harm_reduction_vs_precedent": float(dh.mean()),
            "harm_reduction_ci": [float(dh.mean() - hh), float(dh.mean() + hh)],
        }
    return out


def main() -> None:
    jobs = [(di, n) for n in HISTORY for di in range(len(DOMAINS))]
    with ProcessPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(run_domain, jobs))
    res = {
        "runs_per_problem": RUNS,
        "problems": len(DOMAINS),
        "by_history": {str(n): summarise(results, n) for n in HISTORY},
    }
    OUT.write_text(json.dumps(res, indent=1))
    for n, block in res["by_history"].items():
        print(f"\n== history of {n} past uses")
        for p, r in block.items():
            print(
                f"{p:24s} regret {r['regret']:.4f}±{r['regret_se']:.4f} harm {r['harm']:.4f} "
                f"checked {r['checked']:.2f} settled {r['settled_any']:.2f} "
                f"settled-unreliable {r['settled_unreliable']:.3f} "
                f"regret gain {r['regret_gain_vs_precedent']:+.4f} "
                f"[{r['regret_gain_ci'][0]:+.4f},{r['regret_gain_ci'][1]:+.4f}] "
                f"harm cut {r['harm_reduction_vs_precedent']:+.4f} "
                f"[{r['harm_reduction_ci'][0]:+.4f},{r['harm_reduction_ci'][1]:+.4f}]"
            )


if __name__ == "__main__":
    main()
