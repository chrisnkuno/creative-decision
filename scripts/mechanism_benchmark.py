"""Sequential benchmark: does learning per working principle pay, and what does innovation do?

A programme of T related decisions (T = 24, or 48 in one configuration) in one domain,
simulated. The design has two functions. Each has two familiar principles, used by
precedents and truly reliable (θ = 0.95), and three
untested principles whose true reliability θ_p ~ Beta(1, 1) is unknown to everyone. Untested
principles carry more upside: their contribution to value is U(0.3, 0.9) against U(0.4, 0.6) for
familiar ones. Every decision offers the status quo (a familiar combination, value 0.5) and five
candidates that combine one principle per function. Whether a principle works in a given decision
is a fresh draw Z ~ Bernoulli(θ_p), so tests in one decision inform, but do not settle, the next.

The judge classifies principles correctly with probability 0.87 (the mean probability Jev put on
the true principle in scripts/mechanism_jev.py), gives value beliefs that are noisy and
overconfident, and does not know whether untested principles work (r ~ U(0.3, 0.6)). Its ordinal
novelty is the true ν plus 0.06 for hyped wording plus noise (the size of the hype effect measured
on Jev).

Each decision commits to `commit` and pilots the innovation option on a fraction ε of the stake.
A checked option's untested principles each get a test report (sensitivity 0.9, specificity
0.95); an option deployed unchecked reveals whether its principles worked. Payoffs are scored in
expectation given the true θ; learning uses the sampled reports. All policies see the same worlds,
candidates, judge outputs and random reports (common random numbers).

    uv run python scripts/mechanism_benchmark.py      # writes paper_figs/mechanism_benchmark.json
"""

from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from cdm import (
    NEW,
    Dimension,
    Function,
    Judgment,
    Morphology,
    Option,
    Policy,
    Problem,
    apply_mechanism,
    decide,
)
from cdm.belief import temper
from cdm.simplex import closure

OUT = Path("paper_figs/mechanism_benchmark.json")
SEED = 20261026
RUNS = int(sys.argv[1]) if len(sys.argv) > 1 else 300
SENS, SPEC, COST, HARM = 0.9, 0.95, 0.01, 0.0
# (status-quo value, pilot share ε, number of decisions T)
GRID = [(0.5, 0.2, 24), (0.5, 0.05, 24), (0.65, 0.2, 24), (0.65, 0.05, 24), (0.65, 0.05, 48)]
P_TRUE_CLASS = 0.87
HYPE_EFFECT, NOV_NOISE = 0.06, 0.10
LEVELS = ("1", "2", "3", "4", "5")
VALUES = np.linspace(0, 1, 5)
FAMILIAR, UNTESTED = ("f1", "f2"), ("u1", "u2", "u3")
TAU = 0.2


def functions() -> list[Function]:
    names = (*FAMILIAR, *UNTESTED)
    return [Function(fid, fid, {p: p for p in names}) for fid in ("a", "b")]


RECORD = [{"a": "f1", "b": "f1"}, {"a": "f2", "b": "f2"}, {"a": "f1", "b": "f2"}]


@dataclass
class World:
    theta: dict[tuple[str, str], float]
    mu: dict[tuple[str, str], float]


@dataclass
class Cand:
    choice: dict[str, str]  # true principle per function
    value: float  # true value if every principle works
    judged: np.ndarray  # judge's value distribution
    pi: dict[str, np.ndarray]  # judge's classification
    risk: dict[str, float]  # judge's r per function
    judged_novelty: float
    baseline: bool = False


def level_dist(v: float, sd: float = 0.12) -> np.ndarray:
    return closure(np.exp(-0.5 * ((VALUES - v) / sd) ** 2) + 1e-9)


def draw_world(rng: np.random.Generator) -> World:
    theta, mu = {}, {}
    for f in ("a", "b"):
        for p in FAMILIAR:
            theta[(f, p)], mu[(f, p)] = 0.95, float(rng.uniform(0.4, 0.6))
        for p in UNTESTED:
            theta[(f, p)], mu[(f, p)] = float(rng.beta(1, 1)), float(rng.uniform(0.3, 0.9))
    return World(theta, mu)


def classify(rng: np.random.Generator, true: str, labels: tuple[str, ...]) -> np.ndarray:
    v = np.full(len(labels), (1 - P_TRUE_CLASS) / (len(labels) - 1))
    shown = true if rng.random() < 0.95 else str(rng.choice([x for x in labels if x != true]))
    v[labels.index(shown)] = P_TRUE_CLASS
    return v


def draw_decision(
    rng: np.random.Generator, w: World, funcs: list[Function], base_value: float
) -> list[Cand]:
    base = Cand(
        {"a": "f1", "b": "f1"}, base_value, level_dist(base_value), {}, {}, 0.0, baseline=True
    )
    out = [base]
    for _ in range(5):
        choice = {
            f: (str(rng.choice(FAMILIAR)) if rng.random() < 0.4 else str(rng.choice(UNTESTED)))
            for f in ("a", "b")
        }
        value = float(np.clip(np.mean([w.mu[(f, p)] for f, p in choice.items()]), 0, 1))
        value = float(np.clip(value + rng.normal(0, 0.05), 0, 1))
        noisy = closure(level_dist(value) * np.exp(rng.normal(0, 0.3, 5)))
        judged = temper(noisy, 1 / 1.3)
        pi = {f.id: classify(rng, choice[f.id], f.labels()) for f in funcs}
        risk = {f: float(rng.uniform(0.3, 0.6)) for f in ("a", "b")}
        nu_true = float(np.mean([p in UNTESTED for p in choice.values()]))
        hype = rng.random() < 0.5
        nov = float(np.clip(nu_true + HYPE_EFFECT * hype + rng.normal(0, NOV_NOISE), 0, 1))
        out.append(Cand(choice, value, judged, pi, risk, nov))
    return out


def judgments(cands: list[Cand], m: Morphology) -> list[Judgment]:
    js = []
    for i, c in enumerate(cands):
        nov = closure(np.array([1 - c.judged_novelty, 0, 0, 0, c.judged_novelty]) + 1e-9)
        pi = c.pi or m.classification(c.choice)
        js.append(
            Judgment(
                Option(f"o{i}", "", baseline=c.baseline),
                {"value": c.judged},
                {},
                nov,
                mechanism=pi,
                mechanism_risk=c.risk,
            )
        )
    return js


@dataclass(frozen=True)
class Spec:
    name: str
    novelty: str = "mechanism"  # or "judged"
    learn: bool = True
    bonus: float = 0.0
    innovate: bool = True
    veto_untested: bool = False


SPECS = [
    Spec("veto untested principles", learn=False, innovate=False, veto_untested=True),
    Spec("commit only, learning", innovate=False),
    Spec("judged novelty, no learning", novelty="judged", learn=False),
    Spec("mechanism novelty, no learning", learn=False),
    Spec("mechanism novelty + learning"),
    Spec("+ learning bonus (w = 1)", bonus=1.0),
    Spec("+ learning bonus (w = 4)", bonus=4.0),
]


def works(w: World, choice: dict[str, str]) -> float:
    return float(np.prod([w.theta[(f, p)] for f, p in choice.items()]))


def expected(w: World, c: Cand, b: float, *, checked: bool) -> tuple[float, float]:
    """(expected utility, P(harm)) of acting on c, given the true θ and status-quo value b."""
    if c.baseline:
        return b, 0.0
    ok = works(w, c.choice)
    if not checked:
        return ok * c.value + (1 - ok) * HARM, 1 - ok
    u = ok * (SPEC * c.value + (1 - SPEC) * b) + (1 - ok) * (SENS * b + (1 - SENS) * HARM)
    return u - COST, (1 - ok) * (1 - SENS)


def oracle(w: World, cands: list[Cand], b: float) -> float:
    return max(max(expected(w, c, b, checked=k)[0] for k in (False, True)) for c in cands)


def choose(spec: Spec, cands: list[Cand], m: Morphology) -> tuple[int, bool, int | None]:
    """(commit index, commit checked, innovation index) for one decision."""
    pr = Problem("g", {}, [], [Dimension("value", "q", LEVELS, 1.0)], morphology=m)
    js = judgments(cands, m)
    pol = Policy(
        downside_tolerance=TAU,
        check_sensitivity=SENS,
        check_specificity=SPEC,
        check_cost=COST,
        harm_value=HARM,
        learning_weight=spec.bonus,
    )
    if spec.veto_untested:
        modal = [
            {f.id: f.labels()[int(np.argmax(j.mechanism[f.id]))] for f in m.functions} for j in js
        ]
        keep = [
            j
            for j, ch in zip(js, modal, strict=True)
            if j.option.baseline or all(m.settled(f, p) for f, p in ch.items())
        ]
        d = decide(pr, keep, pol)
    elif spec.novelty == "judged":
        p2, j2 = apply_mechanism(pr, js)
        d = decide(
            replace(p2, morphology=None), [replace(j, mechanism_novelty=None) for j in j2], pol
        )
    else:
        d = decide(pr, js, pol)
    ci = int(d.commit[1:])
    ii = int(d.innovation[1:]) if (spec.innovate and d.innovation) else None
    return ci, d.by_id(d.commit).status == "verify", ii


def observe(
    m: Morphology, c: Cand, z: dict[tuple[str, str], bool], u: dict[str, float], *, checked: bool
) -> None:
    """Update beliefs about the principles the judge says c uses (modal classification)."""
    for f in m.functions:
        p = f.labels()[int(np.argmax(c.pi[f.id]))]
        if p == NEW or m.settled(f.id, p):
            continue  # NEW is option-specific; settled principles are in fact reliable
        works_here = z[(f.id, c.choice[f.id])]
        if checked:
            passed = (u[f.id] < SPEC) if works_here else (u[f.id] >= SENS)
            m.update(f.id, p, passed=passed, sens=SENS, spec=SPEC)
        else:
            m.update(f.id, p, passed=works_here)


def run(base_value: float, eps: float, T: int) -> dict[str, object]:
    rng = np.random.default_rng(SEED)  # the same worlds for every configuration
    funcs = functions()
    keys = ("regret", "harm", "good", "bad", "innovates")
    per = {s.name: {k: np.zeros((RUNS, T)) for k in keys} for s in SPECS}
    for r in range(RUNS):
        w = draw_world(rng)
        decisions = [draw_decision(rng, w, funcs, base_value) for _ in range(T)]
        zs = [
            [{k: bool(rng.random() < th) for k, th in w.theta.items()} for _ in range(6)]
            for _ in range(T)
        ]
        us = [[{f: float(rng.random()) for f in ("a", "b")} for _ in range(6)] for _ in range(T)]
        good = {k for k, th in w.theta.items() if k[1] in UNTESTED and th >= 0.7 and w.mu[k] >= 0.6}
        bad = {k for k, th in w.theta.items() if k[1] in UNTESTED and th < 0.3}
        for spec in SPECS:
            m = Morphology(funcs, [dict(x) for x in RECORD])
            rec = per[spec.name]
            for t, cands in enumerate(decisions):
                ci, checked, ii = choose(spec, cands, m)
                ev, harm = expected(w, cands[ci], base_value, checked=checked)
                if ii is not None:
                    ei, hi = expected(w, cands[ii], base_value, checked=True)
                    ev, harm = (1 - eps) * ev + eps * ei, (1 - eps) * harm + eps * hi
                rec["regret"][r, t] = oracle(w, cands, base_value) - ev
                rec["harm"][r, t] = harm
                rec["innovates"][r, t] = ii is not None
                if not cands[ci].baseline:
                    used = set(cands[ci].choice.items())
                    rec["good"][r, t] = bool(used & good)
                    rec["bad"][r, t] = bool(used & bad)
                if spec.learn:
                    if not cands[ci].baseline:
                        observe(m, cands[ci], zs[t][ci], us[t][ci], checked=checked)
                    if ii is not None:
                        observe(m, cands[ii], zs[t][ii], us[t][ii], checked=True)
        if (r + 1) % 100 == 0:
            print(f"  B={base_value} ε={eps} T={T}: {r + 1}/{RUNS} runs", file=sys.stderr)

    ref = per["mechanism novelty, no learning"]["regret"].mean(axis=1)
    ref_commit = per["commit only, learning"]["regret"].mean(axis=1)
    rows = {}
    for s in SPECS:
        rec = per[s.name]
        mean_run = rec["regret"].mean(axis=1)
        diff = ref - mean_run
        half = 1.96 * diff.std(ddof=1) / np.sqrt(RUNS)
        rows[s.name] = {
            "regret": float(mean_run.mean()),
            "regret_se": float(mean_run.std(ddof=1) / np.sqrt(RUNS)),
            "regret_first8": float(rec["regret"][:, :8].mean()),
            "regret_last8": float(rec["regret"][:, -8:].mean()),
            "harm": float(rec["harm"].mean()),
            "good_last8": float(rec["good"][:, -8:].mean()),
            "bad_last8": float(rec["bad"][:, -8:].mean()),
            "innovates": float(rec["innovates"].mean()),
            "gain_vs_no_learning": float(diff.mean()),
            "gain_ci": [float(diff.mean() - half), float(diff.mean() + half)],
            "gain_vs_commit_only": float((ref_commit - mean_run).mean()),
            "gain_vs_commit_only_ci": [
                float((ref_commit - mean_run).mean() - _half(ref_commit - mean_run)),
                float((ref_commit - mean_run).mean() + _half(ref_commit - mean_run)),
            ],
            "curve": [float(x) for x in rec["regret"].mean(axis=0)],
        }
    return {"base_value": base_value, "eps": eps, "T": T, "policies": rows}


def _half(x: np.ndarray) -> float:
    return float(1.96 * x.std(ddof=1) / np.sqrt(x.size))


def _run(cfg: tuple[float, float, int]) -> dict[str, object]:
    return run(*cfg)


def main() -> None:
    with ProcessPoolExecutor(max_workers=len(GRID)) as pool:
        results = list(pool.map(_run, GRID))
    OUT.write_text(json.dumps({"runs": RUNS, "seed": SEED, "grid": results}, indent=1))
    for res in results:
        print(f"\n== status quo {res['base_value']}, pilot share {res['eps']}, T = {res['T']}")
        for name, r in res["policies"].items():  # type: ignore[union-attr]
            print(
                f"{name:32s} regret {r['regret']:.4f}±{r['regret_se']:.4f} "
                f"first8 {r['regret_first8']:.4f} last8 {r['regret_last8']:.4f} "
                f"harm {r['harm']:.3f} good {r['good_last8']:.2f} bad {r['bad_last8']:.2f} "
                f"innov {r['innovates']:.2f} gain {r['gain_vs_no_learning']:+.4f} "
                f"[{r['gain_ci'][0]:+.4f},{r['gain_ci'][1]:+.4f}] vs commit-only "
                f"{r['gain_vs_commit_only']:+.4f} [{r['gain_vs_commit_only_ci'][0]:+.4f},"
                f"{r['gain_vs_commit_only_ci'][1]:+.4f}]"
            )


if __name__ == "__main__":
    main()
