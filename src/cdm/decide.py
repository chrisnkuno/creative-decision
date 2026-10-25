"""Policy layer: exact creative decisions from calibrated beliefs.

For option i with value-dimension beliefs p_{i,d}, level utilities v_d and weights w_d:

    V_i = m + (Σ_d (w_d / Σw) · X_{i,d} − m) · Π_{soft c} M_{i,c}    X_{i,d} ~ p_{i,d} on v_d
                                                              M_{i,c} = 1−λ w.p. q_{i,c}, else 1
    m   = Σ_d (w_d / Σw) · min v_d                              (the utility floor)

Soft penalties shrink utility toward the floor m, not toward 0, so decisions do not depend on
where zero sits on the utility scale (P8).
    f_i = 1 − Π_{hard c} (1 − q_{i,c})                          P(fails a hard check)
    U_i = V_i  w.p. 1−f_i,   B  w.p. f_i                       (verify, else fall back)

All sources of randomness are treated as independent. Every quantity is computed exactly on
finite lotteries (no sampling).

Imperfect checks. With sensitivity s = P(caught | violation), specificity c = P(pass | no
violation), harm value H and check cost k (charged to verify options):

    U_i = V_i w.p. (1−f_i)c,   B w.p. f_i s + (1−f_i)(1−c),   H w.p. f_i(1−s)      (− k)

so the residual risk f_i(1−s) is explicit. The defaults s = c = 1, k = 0 recover the line above.
Without independence, f_i can be anything in the Fréchet interval [max_c q, min(1, Σ_c q)].
`hard_dependence="robust"` evaluates both ends and keeps the one with the lower E[U_i]; E[U_i] is
affine in f_i, so that is the minimum over every dependence between the checks.

A hard check gets one of three states: clear (q ≤ δ), verify (δ < q < ρ) or reject (q ≥ ρ).
A decision model's q ≈ 0.5 on an unfamiliar option means *unknown*, not *risky*. "Verify" turns
unknown into a check that has a fallback, instead of a veto.

Selection:

    reliable(i)  ⇔ status ≠ reject  ∧  CVaR_α(U_i) ≥ CVaR_α(B) − τ
    commit       = argmax_{reliable} E[U_i]
    innovation   = argmax_{reliable, novelty ≥ ν, ≠ commit} EI_i
                   ν = min_novelty for judged novelty; for mechanism novelty, the gate
                   ½·min_f w̄_f (midway to the smallest ν of one unused principle)
                   EI_i = (1−f)c·E[(V_i−k−B)⁺] + f(1−s)·E[(H−k−B)⁺]   (a fallback adds nothing)
    portfolio    = P(best) over {baseline} ∪ {best reliable option per niche}
    escalate     ⇔ EVPI = E[max_{reliable} U] − max_{reliable} E[U] > κ

Axioms the policy satisfies (tests/test_behaviour.py):

  P1  Baseline is always reliable, so `commit` always exists.
  P2  Order invariance: permuting options permutes the outputs. Renaming them does too, except
      that an exact tie in E[U] between two non-baseline options goes to the smaller id.
  P3  Weight-scale invariance: scaling every w_d by c > 0 changes nothing.
  P4  Independence of irrelevant alternatives for `commit`: adding an option that is rejected,
      or whose E[U] is strictly below the commit's, does not change `commit`. Ties go to the
      status quo, then to the smaller id.
  P5  Novelty neutrality: novelty never changes E[U], CVaR, P(best) or `commit`; it only gates
      eligibility for the innovation slot.
  P6  Gate monotonicity: raising a hard-violation probability never improves an option's status
      or lowers f. With perfect checks E[U] moves toward E[B]; with H ≤ min B, EI never rises.
  P7  Dominance (perfect checks): if V_i ⪰_FSD V_j and f_i ≤ f_j, then P(best_i) ≥ P(best_j);
      and E[U_i] ≥ E[U_j] whenever E[V_j] ≥ E[B]. Without that condition E[U] can reverse:
      a riskier option falls back to a better status quo more often.
  P8  Utility-origin invariance: mapping every level value, H and the null value by x ↦ ax + b
      (a > 0), and τ, κ and k by x ↦ ax, maps every E, CVaR and EI the same way and leaves every
      status, P(best), commit, innovation and escalation unchanged.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace

import numpy as np

from .lottery import TIE_TOL, Competitor, Lottery, expected_improvement, expected_max, p_best
from .mechanism import mechanism_summary
from .model import Constraint, Judgment, Problem

NULL_BASELINE = "∅"
MECHANISM = Constraint(
    "mechanism",
    "fail to work as described, because one of its working principles does not perform its "
    "function in this setting",
    hard=True,
)


@dataclass(frozen=True)
class Policy:
    alpha: float = 0.2  # CVaR tail mass
    clear_below: float = 0.15  # δ
    reject_above: float = 0.70  # ρ
    soft_penalty: float = 0.3  # λ
    downside_tolerance: float = 0.05  # τ
    min_novelty: float = 0.5  # judged novelty on the normalised 0..1 scale
    # gate on mechanism novelty ν; None = half the smallest function weight share, the midpoint
    # between 0 and the smallest ν of an option with one unused principle
    min_mechanism_novelty: float | None = None
    voi_cost: float = 0.02  # κ
    null_value: float = 0.0  # utility of doing nothing, used when no baseline is given
    check_sensitivity: float = 1.0  # s = P(check catches a violation)
    check_specificity: float = 1.0  # c = P(check passes a non-violating option)
    check_cost: float = 0.0  # k, utility charged to every verify option
    harm_value: float = 0.0  # H, utility of an undetected hard violation
    hard_dependence: str = "independent"  # or "robust": worst E[U] over the Fréchet interval
    learning_weight: float = 0.0  # utility per unit change in θ, summed over later uses (M5)

    def __post_init__(self) -> None:
        if not 0 <= self.clear_below <= self.reject_above <= 1:
            raise ValueError("need 0 ≤ clear_below ≤ reject_above ≤ 1")
        if not 0 < self.alpha <= 1 or not 0 <= self.soft_penalty <= 1:
            raise ValueError("alpha ∈ (0,1], soft_penalty ∈ [0,1]")
        if not 0 <= self.check_sensitivity <= 1 or not 0 <= self.check_specificity <= 1:
            raise ValueError("check sensitivity and specificity must be in [0,1]")
        if self.check_cost < 0 or self.hard_dependence not in {"independent", "robust"}:
            raise ValueError("check_cost ≥ 0; hard_dependence is 'independent' or 'robust'")
        if self.learning_weight < 0:
            raise ValueError("learning_weight ≥ 0")


@dataclass
class OptionStats:
    id: str
    value: Lottery  # V_i
    utility: Lottery  # U_i
    fail: float
    mean: float
    std: float
    cvar: float
    novelty: float
    niche: str
    status: str  # clear | verify | reject
    checks: list[str] = field(default_factory=list)
    residual: float = 0.0  # P(undetected hard violation) = f(1−s)
    race: Competitor | None = None  # how the option enters the P(best) race
    p_best: float = 0.0
    ei: float = 0.0
    reliable: bool = False
    baseline: bool = False
    combination: float = 0.0  # κ: P(the combination of principles is in no precedent)
    learning: float = 0.0


@dataclass
class Decision:
    stats: list[OptionStats]
    commit: str
    innovation: str | None
    portfolio: dict[str, float]
    evpi: float
    escalate: list[str]

    def by_id(self, oid: str) -> OptionStats:
        return next(s for s in self.stats if s.id == oid)


def value_lottery(problem: Problem, j: Judgment, policy: Policy) -> Lottery:
    wsum = sum(d.weight for d in problem.dimensions)
    if wsum <= 0:
        raise ValueError("dimension weights must have a positive sum")
    v = Lottery.point(0.0)
    for d in problem.dimensions:
        v = v + Lottery.of(d.level_values(), j.dims[d.id]).scale(d.weight / wsum)
    floor = sum(d.weight / wsum * min(d.level_values()) for d in problem.dimensions)
    hard = {c.id for c in problem.constraints if c.hard}
    soft = [q for cid, q in sorted(j.violations.items()) if cid not in hard and q > 0]
    if soft and floor:
        v = v.shift(-floor)
    for q in soft:
        v = v * Lottery.bernoulli(1.0, 1.0 - policy.soft_penalty, q)
    return v.shift(floor) if soft and floor else v


def hard_risks(problem: Problem, j: Judgment) -> dict[str, float]:
    return {c.id: j.violations.get(c.id, 0.0) for c in problem.constraints if c.hard}


def frechet(qs: dict[str, float]) -> tuple[float, float]:
    """The range of P(any violation) over every joint distribution with these marginals."""
    return (max(qs.values()), min(1.0, sum(qs.values()))) if qs else (0.0, 0.0)


def gate(problem: Problem, j: Judgment, policy: Policy) -> tuple[str, list[str], float]:
    """Gate state, the checks to run, and f under independence."""
    qs = hard_risks(problem, j)
    fail = 1.0 - float(np.prod([1 - q for q in qs.values()])) if qs else 0.0
    if any(q >= policy.reject_above for q in qs.values()):
        return "reject", [], fail
    checks = sorted(c for c, q in qs.items() if q > policy.clear_below)
    return ("verify" if checks else "clear"), checks, fail


def branches(
    v: Lottery, b: Lottery, fail: float, policy: Policy, *, verify: bool
) -> tuple[Lottery, Competitor, float, float]:
    """U, the race entry, EI(U; B) and the residual risk for an option with hard-fail mass f.

    Branches: proceed with V w.p. (1−f)c, fall back to B w.p. fs + (1−f)(1−c), undetected harm
    H w.p. f(1−s). A fallback is the status-quo action itself, so it adds nothing to EI and is
    removed from the race rather than tied with B.
    """
    s, c = policy.check_sensitivity, policy.check_specificity
    k = policy.check_cost if verify else 0.0
    ok, harm = (1 - fail) * c, fail * (1 - s)
    back = max(0.0, 1.0 - ok - harm)
    h = Lottery.point(policy.harm_value)
    go = v if harm <= 0 else (h if ok <= 0 else v.mix(h, harm / (ok + harm)))
    u = b if back >= 1 else (go if back <= 0 else go.mix(b, back))
    if k:
        u, go = u.shift(-k), go.shift(-k)
    ei = ok * expected_improvement(v.shift(-k), b) + harm * expected_improvement(h.shift(-k), b)
    return u, Competitor(go, back), ei, harm


def _novelty(dist: np.ndarray) -> float:
    return float(np.arange(dist.size) @ dist / max(dist.size - 1, 1))


def _niche(j: Judgment) -> str:
    return min(j.niche, key=lambda n: (-j.niche[n], n)) if j.niche else "-"


def _argmax(idx: list[int], key: Callable[[int], float], ids: list[str], base: int) -> int | None:
    """Deterministic argmax. Values within float noise of the best are ties; ties go to the
    status quo (switching needs a strict improvement, which also gives P4), then to the smallest
    id (order invariance, P2)."""
    if not idx:
        return None
    vals = {i: key(i) for i in idx}
    top = max(vals.values())
    near = [i for i in idx if vals[i] >= top - TIE_TOL * max(1.0, abs(top))]
    return min(near, key=lambda i: (i != base, ids[i]))


def apply_mechanism(problem: Problem, judgments: list[Judgment]) -> tuple[Problem, list[Judgment]]:
    """Mechanism layer → policy: ν replaces the judged novelty, q^mech becomes a hard check.

    The status quo's mechanism is settled (q^mech = 0, ν = 0). Idempotent.
    """
    morph = problem.morphology
    if morph is None:
        return problem, judgments
    cons = list(problem.constraints)
    if all(c.id != MECHANISM.id for c in cons):
        cons.append(MECHANISM)
    out = []
    for j in judgments:
        if j.option.baseline:
            settled = {**j.violations, MECHANISM.id: 0.0}
            out.append(replace(j, violations=settled, mechanism_novelty=0.0, learning=0.0))
            continue
        if set(j.mechanism) != {f.id for f in morph.functions}:
            raise ValueError(f"{j.option.id}: needs a classification for every function")
        nov, comb, q, learn = mechanism_summary(morph, j.mechanism, j.mechanism_risk)
        out.append(
            replace(
                j,
                violations={**j.violations, MECHANISM.id: q},
                mechanism_novelty=nov,
                combination=comb,
                learning=learn,
            )
        )
    return replace(problem, constraints=cons), out


def mechanism_gate(problem: Problem, policy: Policy) -> float:
    """The innovation-slot threshold on ν (see `Policy.min_mechanism_novelty`)."""
    if policy.min_mechanism_novelty is not None:
        return policy.min_mechanism_novelty
    morph = problem.morphology
    if morph is None or not morph.functions:
        return policy.min_novelty
    wsum = sum(f.weight for f in morph.functions)
    return 0.5 * min(f.weight for f in morph.functions if f.weight > 0) / wsum


def decide(problem: Problem, judgments: list[Judgment], policy: Policy | None = None) -> Decision:
    pol = policy or Policy()
    problem, judgments = apply_mechanism(problem, judgments)
    mech_nov = {j.option.id: j.mechanism_novelty is not None for j in judgments}
    mech_nov[NULL_BASELINE] = False
    if len({j.option.id for j in judgments}) != len(judgments):
        raise ValueError("option ids must be unique")
    if sum(j.option.baseline for j in judgments) > 1:
        raise ValueError("at most one baseline")

    stats: list[OptionStats] = []
    base_j = next((j for j in judgments if j.option.baseline), None)
    if base_j is None:
        b = Lottery.point(pol.null_value)
        stats.append(
            OptionStats(
                NULL_BASELINE,
                b,
                b,
                0.0,
                b.mean(),
                0.0,
                b.cvar(pol.alpha),
                0.0,
                "-",
                "clear",
                baseline=True,
            )
        )
    else:
        b = value_lottery(problem, base_j, pol)  # the status quo's own checks are settled

    for j in judgments:
        v = b if j.option.baseline else value_lottery(problem, j, pol)
        status, checks, fail = ("clear", [], 0.0) if j.option.baseline else gate(problem, j, pol)
        if j.option.baseline:
            u, entry, ei, residual = v, Competitor(v), 0.0, 0.0
        else:
            ends = frechet(hard_risks(problem, j)) if pol.hard_dependence == "robust" else (fail,)
            fail, (u, entry, ei, residual) = min(
                ((f, branches(v, b, f, pol, verify=status == "verify")) for f in ends),
                key=lambda fb: fb[1][0].mean(),
            )
        stats.append(
            OptionStats(
                id=j.option.id,
                value=v,
                utility=u,
                fail=fail,
                mean=u.mean(),
                std=u.std(),
                cvar=u.cvar(pol.alpha),
                novelty=_novelty(j.novelty) if j.mechanism_novelty is None else j.mechanism_novelty,
                combination=j.combination,
                learning=j.learning,
                niche=_niche(j),
                status=status,
                checks=checks,
                baseline=j.option.baseline,
                ei=ei,
                residual=residual,
                race=entry,
            )
        )

    base_i = next(i for i, s in enumerate(stats) if s.baseline)
    floor = stats[base_i].cvar - pol.downside_tolerance
    for s in stats:
        s.reliable = s.status != "reject" and s.cvar >= floor - 1e-12

    def race(idx: list[int]) -> list[Competitor]:
        return [stats[i].race or Competitor(stats[i].value) for i in idx]

    live = [i for i, s in enumerate(stats) if s.status != "reject"]
    for i, pb in zip(live, p_best(race(live)), strict=True):
        stats[i].p_best = float(pb)

    ids = [s.id for s in stats]
    rel = [i for i, s in enumerate(stats) if s.reliable]
    commit = _argmax(rel, lambda i: stats[i].mean, ids, base_i)
    if commit is None:  # unreachable by P1: the baseline is always reliable
        raise RuntimeError("no reliable option; the baseline invariant is broken")
    gate_nu = mechanism_gate(problem, pol)
    novel = [
        i
        for i in rel
        if i != commit
        and stats[i].novelty >= (gate_nu if mech_nov[stats[i].id] else pol.min_novelty)
    ]
    innovation = _argmax(
        novel, lambda i: stats[i].ei + pol.learning_weight * stats[i].learning, ids, base_i
    )

    elites: dict[str, int] = {}
    for i in rel:
        if stats[i].baseline:
            continue
        n = stats[i].niche
        cur = elites.get(n)
        if cur is None or _argmax([cur, i], lambda k: stats[k].mean, ids, base_i) == i:
            elites[n] = i
    pidx = sorted({base_i, *elites.values()})
    portfolio = {stats[i].id: float(p) for i, p in zip(pidx, p_best(race(pidx)), strict=True)}

    evpi = max(0.0, expected_max(race(rel)) - max(stats[i].mean for i in rel))
    escalate = []
    if evpi > pol.voi_cost:
        top = sorted(rel, key=lambda i: (-stats[i].p_best, stats[i].id))[:2]
        escalate = [stats[i].id for i in top]

    return Decision(
        stats=stats,
        commit=stats[commit].id,
        innovation=None if innovation is None else stats[innovation].id,
        portfolio=portfolio,
        evpi=evpi,
        escalate=escalate,
    )
