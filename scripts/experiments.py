"""Synthetic experiments for the paper: a policy benchmark and sensitivity sweeps.

Everything here is simulated. The benchmark isolates the *policy* layer: every policy sees the
same judge beliefs, and outcomes are scored in exact expectation against a hidden ground truth.
It is evidence about how the gates behave under stated assumptions, not about real decisions.

    uv run python scripts/experiments.py       # writes paper_figs/experiments.json
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cdm import Constraint, Dimension, Judgment, Option, Policy, Problem, decide
from cdm.belief import orbit_mean, temper
from cdm.simplex import closure, cyclic_orbit, perturb, present, relabel

OUT = Path("paper_figs")
LEVELS = ("1", "2", "3", "4", "5")
VALUES = np.linspace(0, 1, 5)
HARM = 0.0  # utility of an undetected violation; the baseline sits near 0.5
COST = 0.01  # utility cost of running a check
SPEC = 0.95  # check specificity
T_JUDGE = 1.3  # the judge is overconfident by this temperature; cdm is assumed to have fit it

PROBLEM = Problem(
    goal="synthetic",
    context={},
    options=[],
    dimensions=[Dimension("value", "q", LEVELS, 1.0)],
    constraints=[Constraint("harm", "harm", hard=True)],
)


@dataclass
class Candidate:
    true_dist: np.ndarray  # true distribution over levels
    judged: np.ndarray  # judge's (overconfident, noisy) distribution
    harmful: bool
    familiar: bool
    q_hat: float  # judge's P(violation)
    baseline: bool = False

    @property
    def true_mean(self) -> float:
        return float(VALUES @ self.true_dist)


def draw_problem(rng: np.random.Generator, k: int = 5) -> list[Candidate]:
    base = closure(np.array([0.05, 0.2, 0.5, 0.2, 0.05]))
    out = [Candidate(base, base, harmful=False, familiar=True, q_hat=0.0, baseline=True)]
    for _ in range(k):
        familiar = bool(rng.random() < 0.5)
        # unfamiliar ideas have more upside and more spread
        conc = np.array([1, 1, 1, 1, 1.0]) if familiar else np.array([1.2, 0.6, 0.5, 0.8, 1.6])
        true = closure(rng.dirichlet(conc * 2) + 1e-6)
        noisy = closure(true * np.exp(rng.normal(0, 0.3, 5)))
        judged = temper(noisy, 1 / T_JUDGE)  # sharpened: overconfident
        harmful = bool(rng.random() < 0.25)
        if familiar:  # the judge knows: harm is visible
            q = (0.85 if harmful else 0.05) + rng.normal(0, 0.03)
        else:  # the judge does not know: q ≈ 0.5 whatever the truth
            q = rng.uniform(0.35, 0.6)
        out.append(Candidate(true, judged, harmful, familiar, float(np.clip(q, 0.001, 0.999))))
    return out


def judgments(cands: list[Candidate], *, t: float = T_JUDGE) -> list[Judgment]:
    """Judgments with the value beliefs tempered by t (t = 1 leaves the judge's raw output)."""
    js = []
    for i, c in enumerate(cands):
        dist = c.judged if c.baseline else temper(c.judged, t)
        nov = (
            np.array([0.1, 0.1, 0.1, 0.1, 0.6])
            if not c.familiar
            else np.array([0.6, 0.1, 0.1, 0.1, 0.1])
        )
        js.append(
            Judgment(
                Option(f"o{i}", "", baseline=c.baseline),
                {"value": dist},
                {"harm": 0.0 if c.baseline else c.q_hat},
                closure(nov),
            )
        )
    return js


@dataclass(frozen=True)
class Env:
    """How the world's checks behave, and what temperature cdm believes the judge has."""

    sens: float = 0.9
    spec: float = SPEC
    cost: float = COST
    t_assumed: float = T_JUDGE

    def policy(self, tau: float) -> Policy:
        return Policy(
            downside_tolerance=tau,
            check_sensitivity=self.sens,
            check_specificity=self.spec,
            check_cost=self.cost,
            harm_value=HARM,
        )


def outcome(c: Candidate, base: Candidate, *, checked: bool, env: Env) -> tuple[float, float]:
    """(expected true utility, P(realised harm)) of acting on c under the check protocol."""
    if c.baseline:
        return base.true_mean, 0.0
    if not checked:
        return (HARM, 1.0) if c.harmful else (c.true_mean, 0.0)
    if c.harmful:
        return env.sens * base.true_mean + (1 - env.sens) * HARM - env.cost, 1 - env.sens
    return env.spec * c.true_mean + (1 - env.spec) * base.true_mean - env.cost, 0.0


Chooser = Callable[[list[Candidate], Env], tuple[int, bool]]  # (option index, was it checked)


def by_mean(cands: list[Candidate], t: float, allowed: Callable[[Candidate], bool]) -> int:
    mean = [float(VALUES @ j.dims["value"]) for j in judgments(cands, t=t)]
    return max((i for i, c in enumerate(cands) if c.baseline or allowed(c)), key=lambda i: mean[i])


def cdm(tau: float, *, perfect: bool = False) -> Chooser:
    def choose(cands: list[Candidate], env: Env) -> tuple[int, bool]:
        pol = Policy(downside_tolerance=tau) if perfect else env.policy(tau)
        d = decide(PROBLEM, judgments(cands, t=env.t_assumed), pol)
        return int(d.commit[1:]), d.by_id(d.commit).status == "verify"

    return choose


def baselines() -> dict[str, Chooser]:
    pol = Policy()
    return {
        "raw judge": lambda cs, _env: (by_mean(cs, 1.0, lambda _c: True), False),
        "accept uncertain": lambda cs, env: (
            by_mean(cs, env.t_assumed, lambda c: c.q_hat < pol.reject_above),
            False,
        ),
        "veto uncertain": lambda cs, env: (
            by_mean(cs, env.t_assumed, lambda c: c.q_hat <= pol.clear_below),
            False,
        ),
    }


def score(problems: list[list[Candidate]], choose: Chooser, env: Env) -> dict[str, np.ndarray]:
    reg, harm, nov = [], [], []
    for cands in problems:
        i, checked = choose(cands, env)
        u, h = outcome(cands[i], cands[0], checked=checked, env=env)
        best = max(outcome(c, cands[0], checked=False, env=env)[0] for c in cands if not c.harmful)
        reg.append(best - u)
        harm.append(h)
        nov.append(not cands[i].familiar and not cands[i].baseline)
    return {"regret": np.array(reg), "harm": np.array(harm), "novel": np.array(nov, dtype=float)}


def summary(r: dict[str, np.ndarray], ref: dict[str, np.ndarray] | None = None) -> dict:
    n = r["regret"].size
    out = {
        "regret": float(r["regret"].mean()),
        "regret_se": float(r["regret"].std(ddof=1) / np.sqrt(n)),
        "harm": float(r["harm"].mean()),
        "novel": float(r["novel"].mean()),
    }
    if ref is not None:  # paired against the same problems: positive = better than ref
        d = ref["regret"] - r["regret"]
        se = float(d.std(ddof=1) / np.sqrt(n))
        out["gain_vs_ref"] = float(d.mean())
        out["gain_ci"] = [float(d.mean() - 1.96 * se), float(d.mean() + 1.96 * se)]
    return out


def problems_for(seed: int, n: int, flip: float = 0.0) -> list[list[Candidate]]:
    """`flip`: share of familiar harmful options the judge confidently misreads as safe."""
    rng = np.random.default_rng(seed)
    out = [draw_problem(rng) for _ in range(n)]
    if flip:
        r2 = np.random.default_rng(seed + 10_000)
        for cands in out:
            for c in cands[1:]:
                if c.familiar and c.harmful and r2.random() < flip:
                    c.q_hat = float(np.clip(0.05 + r2.normal(0, 0.03), 0.001, 0.999))
    return out


INF = float("inf")  # no CVaR gate at all
TAUS = (0.0, 0.025, 0.05, 0.1, 0.15, 0.2, 0.3, INF)
TUNE_SEED, TEST_SEED = 1, 2


def benchmark(n_tune: int = 1000, n_test: int = 1500) -> dict:
    env = Env()
    tune, test = problems_for(TUNE_SEED, n_tune), problems_for(TEST_SEED, n_test)

    # 1. pick τ on the tuning problems only (ties go to the smaller τ)
    tuned = {t: summary(score(tune, cdm(t), env)) for t in TAUS}
    tau_star = min(TAUS, key=lambda t: (round(tuned[t]["regret"], 4), t))

    # 2. the main comparison on fresh problems, paired against veto
    choosers = {
        **baselines(),
        "cdm, perfect-check model": cdm(0.05, perfect=True),
        "cdm, check model, default tau": cdm(0.05),
        "cdm, check model, tuned tau": cdm(tau_star),
        "expected utility + checks (no CVaR gate)": cdm(INF),
    }
    runs = {name: score(test, ch, env) for name, ch in choosers.items()}
    veto = runs["veto uncertain"]
    table = {name: summary(r, veto) for name, r in runs.items()}
    frontier = {str(t): summary(score(test, cdm(t), env), veto) for t in TAUS}

    # 3. harm as check sensitivity varies
    sens_grid = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
    keep = ("raw judge", "accept uncertain", "veto uncertain", "cdm, perfect-check model")
    by_sens: dict[str, list[float]] = {}
    for s in sens_grid:
        e = Env(sens=s)
        for name in keep:
            by_sens.setdefault(name, []).append(
                float(score(test, choosers[name], e)["harm"].mean())
            )
        by_sens.setdefault("cdm, check model, tuned tau", []).append(
            float(score(test, cdm(tau_star), e)["harm"].mean())
        )

    # 4. robustness of the tuned policy against veto
    def versus(probs: list[list[Candidate]], e: Env) -> dict:
        v = score(probs, choosers["veto uncertain"], e)
        return {"cdm": summary(score(probs, cdm(tau_star), e), v), "veto": summary(v)}

    flipped = problems_for(TEST_SEED, n_test, flip=0.2)
    robust = {
        "confidently wrong judge (20%)": versus(flipped, env),
        **{
            f"specificity {sp}, cost {co}": versus(test, Env(spec=sp, cost=co))
            for sp, co in ((0.8, 0.01), (0.95, 0.05), (0.8, 0.05))
        },
        **{f"assumed T = {t}": versus(test, Env(t_assumed=t)) for t in (1.0, 2.0)},
    }
    return {
        "n_tune": n_tune,
        "n_test": n_test,
        "taus": [str(t) for t in TAUS],
        "tuning": {str(t): v for t, v in tuned.items()},
        "tau_star": str(tau_star),
        "table": table,
        "frontier": frontier,
        "sens": list(sens_grid),
        "harm_by_sens": by_sens,
        "robustness": robust,
    }


def exact_vs_mc(n: int = 400_000) -> list[dict]:
    """The exact engine against an independent Monte Carlo, on tests/test_functional's problem."""
    sys.path.insert(0, ".")
    from tests.judgments import judgment, problem  # noqa: PLC0415
    from tests.test_functional import simulate  # noqa: PLC0415

    rng = np.random.default_rng(3)
    pr = problem()

    def dims() -> list[np.ndarray]:
        return [closure(rng.dirichlet(np.ones(5))) for _ in range(3)]

    js = [
        judgment("base", pr, dims(), baseline=True),
        judgment("x", pr, dims(), harm=0.3, effort=0.4),
        judgment("y", pr, dims(), harm=0.1, effort=0.1),
    ]
    pol = Policy(reject_above=0.99)
    d = decide(pr, js, pol)
    u, race = simulate(pr, js, pol, n, rng)
    top = race.max(axis=0)
    wins = top == race
    shares = (wins / wins.sum(axis=0)).mean(axis=1)
    k = int(pol.alpha * n)
    return [
        {
            "option": st.id,
            "mean_exact": st.mean,
            "mean_mc": float(u[i].mean()),
            "mean_se": float(u[i].std() / np.sqrt(n)),
            "cvar_exact": st.cvar,
            "cvar_mc": float(np.sort(u[i])[:k].mean()),
            "pbest_exact": st.p_best,
            "pbest_mc": float(shares[i]),
        }
        for i, st in enumerate(d.stats)
    ]


def b4_smoothing_error(n: int = 2000, eta: float = 0.03) -> dict:
    """How far the cyclic orbit is from exact once raw answers are smoothed into the interior."""
    rng = np.random.default_rng(TEST_SEED)
    err0, err = [], []
    for _ in range(n):
        p, b = closure(rng.dirichlet(np.ones(5))), closure(rng.dirichlet(np.ones(5)))
        ans = [relabel(perturb(present(p, o), b), o) for o in cyclic_orbit(5)]
        err0.append(float(np.abs(orbit_mean(ans, 0.0) - p).max()))
        err.append(float(np.abs(orbit_mean(ans, eta) - p).max()))
    return {
        "eta": eta,
        "max_err_eta0": float(np.max(err0)),
        "median_err": float(np.median(err)),
        "p95_err": float(np.quantile(err, 0.95)),
    }


def innovation_sweep() -> dict:
    """Which option takes the innovation slot as spread, tolerance τ and hard risk vary."""
    base = closure(np.array([0, 0, 0.98, 0.02, 0]) + 1e-12)  # a hair better, so B stays commit
    spreads = np.linspace(0, 0.5, 26)  # mass moved from the middle level to each extreme
    taus = np.linspace(0, 0.3, 31)
    qs = np.linspace(0, 0.69, 24)
    narrow = closure(np.array([0, 0.1, 0.8, 0.1, 0]) + 1e-12)

    def pick(spread: float, tau: float, q: float, pol: Policy | None = None) -> int:
        wide = closure(np.array([spread, 0, 1 - 2 * spread, 0, spread]) + 1e-12)
        js = [
            Judgment(Option("b", "", baseline=True), {"value": base}, {}, closure(np.ones(5))),
            Judgment(
                Option("narrow", ""),
                {"value": narrow},
                {"harm": 0.0},
                closure(np.array([0, 0, 0, 0, 1.0]) + 1e-12),
            ),
            Judgment(
                Option("wide", ""),
                {"value": wide},
                {"harm": q},
                closure(np.array([0, 0, 0, 0, 1.0]) + 1e-12),
            ),
        ]
        d = decide(PROBLEM, js, pol or Policy(downside_tolerance=tau))
        if d.commit != "b":
            return 3
        return {None: 0, "narrow": 1, "wide": 2}[d.innovation]

    tau_grid = [[pick(s, t, 0.0) for s in spreads] for t in taus]
    q_grid = [[pick(s, 0.3, q) for s in spreads] for q in qs]
    imperfect = Policy(downside_tolerance=0.3, check_sensitivity=0.9, harm_value=HARM)
    q_check = [[pick(s, 0.3, q, imperfect) for s in spreads] for q in qs]
    return {
        "spreads": spreads.tolist(),
        "taus": taus.tolist(),
        "qs": qs.tolist(),
        "by_tau": tau_grid,
        "by_q": q_grid,
        "by_q_check": q_check,
    }


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    if sys.argv[1:] in (["innovation"], ["mc"]):  # refresh one cheap part
        result = json.loads((OUT / "experiments.json").read_text())
        if sys.argv[1] == "innovation":
            result["innovation"] = innovation_sweep()
        else:
            result["exact_vs_mc"] = exact_vs_mc()
        (OUT / "experiments.json").write_text(json.dumps(result, indent=1))
        sys.exit()
    result = {
        "benchmark": benchmark(),
        "innovation": innovation_sweep(),
        "b4": b4_smoothing_error(),
        "exact_vs_mc": exact_vs_mc(),
    }
    (OUT / "experiments.json").write_text(json.dumps(result, indent=1))
    b = result["benchmark"]
    print(f"tau* = {b['tau_star']} (tuning seed); test n = {b['n_test']}")
    for name, d in b["table"].items():
        print(
            f"  {name:42s} regret {d['regret']:.4f} ± {d['regret_se']:.4f}  harm {d['harm']:.4f}"
            f"  explores {d['novel']:.2f}  vs veto {d['gain_vs_ref']:+.4f} {d['gain_ci']}"
        )
    for name, v in b["robustness"].items():
        c = v["cdm"]
        print(
            f"  {name:32s} cdm {c['regret']:.4f} (harm {c['harm']:.4f})"
            f"  veto {v['veto']['regret']:.4f} (harm {v['veto']['harm']:.4f})"
            f"  gain {c['gain_vs_ref']:+.4f} {c['gain_ci']}"
        )
    print(result["b4"])
