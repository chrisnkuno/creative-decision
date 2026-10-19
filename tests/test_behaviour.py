"""Policy axioms P1–P7 (property-based) and the named creative behaviours (scenarios)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from cdm import Decision, Judgment, Policy, Problem, decide
from cdm.decide import NULL_BASELINE
from cdm.simplex import closure

from .judgments import judgment, point, problem, scenario, with_option
from .strategies import close, simplex

Scenario = tuple[Problem, list[Judgment]]
RANK = {"clear": 0, "verify": 1, "reject": 2}


def numbers(d: Decision) -> dict[str, tuple[float, float, float, float, str, bool]]:
    return {s.id: (s.mean, s.cvar, s.p_best, s.ei, s.status, s.reliable) for s in d.stats}


def same(a: Decision, b: Decision) -> bool:
    na, nb = numbers(a), numbers(b)
    return na.keys() == nb.keys() and all(
        close(np.array(na[k][:4]), np.array(nb[k][:4]), 1e-9) and na[k][4:] == nb[k][4:] for k in na
    )


# P1 ----------------------------------------------------------------------------------------
@settings(max_examples=80, deadline=None)
@given(scenario())
def test_p1_baseline_is_reliable_and_a_commit_always_exists(sc: Scenario) -> None:
    d = decide(*sc)
    base = next(s for s in d.stats if s.baseline)
    assert base.reliable
    assert base.status == "clear"
    assert d.by_id(d.commit).reliable
    assert abs(sum(s.p_best for s in d.stats) - 1) < 1e-9
    assert abs(sum(d.portfolio.values()) - 1) < 1e-9
    assert d.evpi >= 0


# P2 ----------------------------------------------------------------------------------------
@settings(max_examples=60, deadline=None)
@given(scenario(), st.randoms())
def test_p2_option_order_and_names_do_not_matter(sc: Scenario, rnd: object) -> None:
    pr, js = sc
    shuffled = list(js)
    rnd.shuffle(shuffled)  # type: ignore[attr-defined]
    a, b = decide(pr, js), decide(pr, shuffled)
    assert same(a, b)
    assert (a.commit, a.innovation, a.escalate) == (b.commit, b.innovation, b.escalate)
    assert a.portfolio.keys() == b.portfolio.keys()

    renamed = [with_option(j, "x_" + j.option.id) for j in js]
    c = decide(pr, renamed)
    assert c.commit in {"x_" + a.commit, a.commit}  # ∅ keeps its name
    assert close(np.array([s.mean for s in c.stats]), np.array([s.mean for s in a.stats]))


# P3 ----------------------------------------------------------------------------------------
@settings(max_examples=60, deadline=None)
@given(scenario(), st.floats(0.01, 100))
def test_p3_scaling_all_weights_changes_nothing(sc: Scenario, c: float) -> None:
    pr, js = sc
    scaled = replace(pr, dimensions=[replace(d, weight=d.weight * c) for d in pr.dimensions])
    a, b = decide(pr, js), decide(scaled, js)
    assert same(a, b)
    assert a.commit == b.commit
    assert a.innovation == b.innovation


# P4 ----------------------------------------------------------------------------------------
@settings(max_examples=60, deadline=None)
@given(scenario())
def test_p4_irrelevant_alternatives_do_not_change_the_commit(sc: Scenario) -> None:
    pr, js = sc
    before = decide(pr, js)
    rejected = judgment("zz_rejected", pr, point(4), harm=0.99, novelty=1.0)
    worst = np.array([1.0, 0.0, 0.0, 0.0, 0.0])  # exact point mass, no floor
    dominated = judgment("zz_dominated", pr, worst, harm=0.0, effort=1.0)
    assert decide(pr, [*js, rejected]).commit == before.commit
    after = decide(pr, [*js, dominated])
    assert after.by_id(before.commit).utility.dominates(after.by_id("zz_dominated").utility)
    assert after.commit == before.commit


# P5 ----------------------------------------------------------------------------------------
@settings(max_examples=60, deadline=None)
@given(scenario(), st.data())
def test_p5_novelty_only_gates_the_innovation_slot(sc: Scenario, data: st.DataObject) -> None:
    pr, js = sc
    remixed = [replace(j, novelty=data.draw(simplex(5))) for j in js]
    a, b = decide(pr, js), decide(pr, remixed)
    assert same(a, b)
    assert a.commit == b.commit
    assert a.portfolio == b.portfolio
    assert a.evpi == b.evpi


# P6 ----------------------------------------------------------------------------------------
@settings(max_examples=60, deadline=None)
@given(scenario(), st.floats(0, 1), st.floats(0, 1))
def test_p6_more_risk_never_helps(sc: Scenario, q1: float, q2: float) -> None:
    pr, js = sc
    lo, hi = sorted((q1, q2))
    target = next((j for j in js if not j.option.baseline), None)
    if target is None:
        return
    others = [j for j in js if j is not target]

    def run(q: float) -> Decision:
        return decide(pr, [*others, replace(target, violations={**target.violations, "harm": q})])

    a, b = run(lo).by_id(target.option.id), run(hi).by_id(target.option.id)
    base = run(lo).stats[0] if all(not j.option.baseline for j in js) else None
    assert RANK[b.status] >= RANK[a.status]
    assert b.fail >= a.fail - 1e-12
    bmean = base.mean if base else next(s for s in run(lo).stats if s.baseline).mean
    assert abs(b.mean - bmean) <= abs(a.mean - bmean) + 1e-9  # U slides toward the fallback
    assert b.ei <= a.ei + 1e-9


@settings(max_examples=60, deadline=None)
@given(scenario(), st.floats(0, 1), st.floats(0, 1), st.floats(0, 1), st.floats(0, 1))
def test_p6_with_imperfect_checks_status_and_ei_still_never_improve(
    sc: Scenario, q1: float, q2: float, s: float, c: float
) -> None:
    pr, js = sc
    lo, hi = sorted((q1, q2))
    target = next((j for j in js if not j.option.baseline), None)
    if target is None:
        return
    others = [j for j in js if j is not target]
    pol = Policy(check_sensitivity=s, check_specificity=c, harm_value=-1.0)  # H below every B

    def run(q: float) -> Decision:
        return decide(
            pr, [*others, replace(target, violations={**target.violations, "harm": q})], pol
        )

    a, b = run(lo).by_id(target.option.id), run(hi).by_id(target.option.id)
    assert RANK[b.status] >= RANK[a.status]
    assert b.fail >= a.fail - 1e-12
    assert b.ei <= a.ei + 1e-9


# P7 ----------------------------------------------------------------------------------------
@settings(max_examples=60, deadline=None)
@given(scenario(), st.integers(0, 4))
def test_p7_dominance_is_respected(sc: Scenario, lvl: int) -> None:
    pr, js = sc
    weak = judgment("weak", pr, point(lvl), harm=0.3, effort=0.5)
    strong = judgment("strong", pr, point(min(lvl + 1, 4)), harm=0.1, effort=0.2)
    d = decide(pr, [*js, weak, strong])
    w, s = d.by_id("weak"), d.by_id("strong")
    base = next(x for x in d.stats if x.baseline)
    assert s.value.dominates(w.value)
    assert s.p_best >= w.p_best - 1e-9
    if w.value.mean() >= base.mean - 1e-12:  # the condition P7 needs for E[U]
        assert s.mean >= w.mean - 1e-9


def test_p7_without_its_condition_a_riskier_option_can_have_higher_expected_utility() -> None:
    # identical V, worse than the status quo: the option that fails more often falls back more
    pr = problem()
    base = judgment("status_quo", pr, point(4), baseline=True)
    safe = judgment("safe", pr, point(0), harm=0.0)
    risky = judgment("risky", pr, point(0), harm=0.6)
    d = decide(pr, [base, safe, risky], Policy(reject_above=0.9))
    assert d.by_id("risky").mean > d.by_id("safe").mean
    assert d.by_id("safe").p_best >= d.by_id("risky").p_best


# P8 ----------------------------------------------------------------------------------------
@settings(max_examples=60, deadline=None)
@given(scenario(), st.floats(0.25, 4), st.floats(-2, 2), st.floats(0, 1), st.floats(-1, 0.5))
def test_p8_decisions_do_not_depend_on_the_utility_origin_or_unit(
    sc: Scenario, a: float, b: float, s: float, h: float
) -> None:
    pr, js = sc
    base_vals = tuple(i / 4 for i in range(5))
    p1 = replace(pr, dimensions=[replace(d, values=base_vals) for d in pr.dimensions])
    p2 = replace(
        pr,
        dimensions=[replace(d, values=tuple(a * v + b for v in base_vals)) for d in pr.dimensions],
    )
    pol1 = Policy(check_sensitivity=s, check_cost=0.01, harm_value=h, null_value=0.1)
    pol2 = replace(
        pol1,
        harm_value=a * h + b,
        null_value=a * 0.1 + b,
        check_cost=a * 0.01,
        downside_tolerance=a * pol1.downside_tolerance,
        voi_cost=a * pol1.voi_cost,
    )
    d1, d2 = decide(p1, js, pol1), decide(p2, js, pol2)
    for x, y in zip(d1.stats, d2.stats, strict=True):
        assert abs(y.mean - (a * x.mean + b)) < 1e-7
        assert abs(y.cvar - (a * x.cvar + b)) < 1e-7
        assert abs(y.ei - a * x.ei) < 1e-7
        assert abs(y.p_best - x.p_best) < 1e-7
        assert (x.status, x.reliable) == (y.status, y.reliable)
    assert (d1.commit, d1.innovation, d1.escalate) == (d2.commit, d2.innovation, d2.escalate)


def test_soft_penalties_never_raise_utility_even_below_zero() -> None:
    neg = (-1.0, -0.75, -0.5, -0.25, 0.0)
    pr = problem(weights=(1.0,))
    pr = replace(pr, dimensions=[replace(d, values=neg) for d in pr.dimensions])
    clean = judgment("clean", pr, point(3))
    penalised = judgment("penalised", pr, point(3), effort=0.9)
    d = decide(pr, [clean, penalised])
    assert d.by_id("penalised").mean < d.by_id("clean").mean


# V1–V3: imperfect verification and dependence ---------------------------------------------
checks = st.tuples(st.floats(0, 1), st.floats(0, 1), st.floats(0, 0.2), st.floats(-1, 0.5))


@settings(max_examples=60, deadline=None)
@given(scenario(), checks)
def test_v1_imperfect_checks_keep_the_invariants_and_expose_residual_risk(
    sc: Scenario, chk: tuple[float, float, float, float]
) -> None:
    s, c, k, h = chk
    pol = Policy(check_sensitivity=s, check_specificity=c, check_cost=k, harm_value=h)
    d = decide(*sc, pol)
    assert abs(sum(st_.p_best for st_ in d.stats) - 1) < 1e-9
    assert abs(sum(d.portfolio.values()) - 1) < 1e-9
    assert d.evpi >= 0
    assert d.by_id(d.commit).reliable
    for o in d.stats:
        assert abs(o.utility.p.sum() - 1) < 1e-9
        assert abs(o.residual - o.fail * (1 - s)) < 1e-12
        if o.residual > 1e-9:  # undetected harm is an atom of U at H − cost
            at = h - (k if o.status == "verify" else 0.0)  # atoms are rounded, so allow 1e-9
            mass = float(o.utility.cdf(at + 1e-9) - o.utility.cdf_left(at - 1e-9))
            assert mass >= o.residual - 1e-9


@settings(max_examples=60, deadline=None)
@given(scenario(), st.floats(0, 1), st.floats(0, 1))
def test_v2_better_checks_never_hurt_when_harm_is_below_the_status_quo(
    sc: Scenario, s1: float, s2: float
) -> None:
    lo, hi = sorted((s1, s2))
    a = decide(*sc, Policy(check_sensitivity=lo, harm_value=-1.0))
    b = decide(*sc, Policy(check_sensitivity=hi, harm_value=-1.0))
    for x, y in zip(a.stats, b.stats, strict=True):
        assert y.mean >= x.mean - 1e-9
        assert y.cvar >= x.cvar - 1e-9
        assert y.residual <= x.residual + 1e-12


@settings(max_examples=60, deadline=None)
@given(scenario(), st.lists(st.floats(0, 1), min_size=1, max_size=4), st.floats(0, 1))
def test_v3_robust_dependence_takes_the_worst_expected_utility_over_the_frechet_interval(
    sc: Scenario, qs: list[float], s: float
) -> None:
    pr, js = sc
    hard = [f"h{i}" for i in range(len(qs))]
    pr = problem(weights=tuple(d.weight for d in pr.dimensions), hard=tuple(hard))
    js = [replace(j, violations=dict(zip(hard, qs, strict=True))) for j in js]
    lo, hi = max(qs), min(1.0, sum(qs))
    pol = Policy(check_sensitivity=s, harm_value=-0.5)
    ind = decide(pr, js, pol)
    rob = decide(pr, js, replace(pol, hard_dependence="robust"))
    for x, y in zip(ind.stats, rob.stats, strict=True):
        if x.baseline:
            continue
        assert lo - 1e-12 <= x.fail <= hi + 1e-12  # independence lies inside the interval
        assert min(abs(y.fail - lo), abs(y.fail - hi)) < 1e-12  # robust picks an end
        assert y.mean <= x.mean + 1e-9  # E[U] is affine in f, so an end is the worst case
        assert x.status == y.status


def test_a_check_that_misses_harm_breaks_the_fallback_bound() -> None:
    pr = problem()
    base = judgment("status_quo", pr, point(2), baseline=True)
    bold = judgment("bold", pr, point(4), harm=0.5, novelty=0.9, niche="b")
    s = decide(pr, [base, bold], Policy(check_sensitivity=0.8)).by_id("bold")
    assert abs(s.residual - 0.1) < 1e-12  # 0.5 × (1 − 0.8) reaches H = 0
    assert abs(float(s.utility.cdf_left(0.5)) - 0.1) < 1e-9


# Named creative behaviours ------------------------------------------------------------------
def test_unknown_is_not_risky_a_coin_flip_check_means_verify_not_veto() -> None:
    pr = problem()
    base = judgment("status_quo", pr, point(2), baseline=True)
    bold = judgment("bold", pr, point(4), harm=0.5, novelty=0.9, niche="b")
    d = decide(pr, [base, bold])
    s = d.by_id("bold")
    assert s.status == "verify"
    assert s.checks == ["harm"]
    assert s.reliable
    assert d.commit == "bold"  # better even after pricing in a 50% chance of falling back


def test_fallback_bounds_the_downside_of_a_bold_bet() -> None:
    pr = problem()
    base = judgment("status_quo", pr, point(2), baseline=True)
    bold = judgment("bold", pr, closure(np.array([0.0, 0.0, 0.0, 0.2, 0.8]) + 1e-9), harm=0.6)
    s = decide(pr, [base, bold]).by_id("bold")
    # the fixture floors every level at 1e-9 mass, so "never" means "only floor-sized mass"
    assert float(s.utility.cdf_left(0.5)) < 1e-6  # P(U below the status quo, level 2 = 0.5)
    assert s.cvar >= 0.5 - 1e-6


def test_clear_evidence_of_harm_is_rejected_everywhere() -> None:
    pr = problem()
    base = judgment("status_quo", pr, point(2), baseline=True)
    fake = judgment("fake", pr, point(4), harm=0.95, novelty=1.0, niche="b")
    d = decide(pr, [base, fake])
    s = d.by_id("fake")
    assert s.status == "reject"
    assert not s.reliable
    assert s.p_best == 0
    assert "fake" not in d.portfolio
    assert d.commit != "fake"
    assert d.innovation != "fake"


def test_with_equal_means_the_wider_novel_bet_takes_the_innovation_slot() -> None:
    pr = problem(weights=(1.0,))
    base = judgment("status_quo", pr, point(2), baseline=True)
    narrow = judgment("narrow", pr, point(2), novelty=0.9)
    wide = judgment("wide", pr, closure(np.array([0.5, 0, 0, 0, 0.5]) + 1e-12), novelty=0.9)
    d = decide(pr, [base, narrow, wide], Policy(downside_tolerance=1.0))
    assert abs(d.by_id("narrow").mean - d.by_id("wide").mean) < 1e-6
    assert d.innovation == "wide"
    assert d.by_id("wide").ei > d.by_id("narrow").ei


def test_no_novelty_means_no_innovation_bet() -> None:
    pr = problem()
    js = [judgment("status_quo", pr, point(2), baseline=True), judgment("x", pr, point(3))]
    assert decide(pr, js).innovation is None


def test_close_contest_escalates_clear_winner_does_not() -> None:
    pr = problem(weights=(1.0,))
    flat = closure(np.ones(5))
    tie = [judgment("a", pr, flat), judgment("b", pr, flat)]
    assert decide(pr, tie).escalate
    clear = [judgment("a", pr, point(4)), judgment("b", pr, point(0))]
    d = decide(pr, clear)
    assert d.commit == "a"
    assert not d.escalate
    assert d.evpi < 1e-9


def test_missing_baseline_uses_doing_nothing() -> None:
    pr = problem()
    d = decide(pr, [judgment("x", pr, point(3))])
    assert d.stats[0].id == NULL_BASELINE
    assert d.stats[0].mean == 0.0


def test_portfolio_keeps_one_elite_per_niche() -> None:
    pr = problem()
    js = [
        judgment("status_quo", pr, point(1), baseline=True),
        judgment("a1", pr, point(3), niche="a"),
        judgment("a2", pr, point(2), niche="a"),
        judgment("b1", pr, point(2), niche="b"),
    ]
    assert set(decide(pr, js).portfolio) == {"status_quo", "a1", "b1"}


def test_invalid_inputs_are_refused() -> None:
    pr = problem()
    j = judgment("x", pr, point(1))
    for bad in (
        lambda: decide(pr, [j, j]),
        lambda: Policy(clear_below=0.8, reject_above=0.2),
        lambda: Policy(alpha=0.0),
        lambda: Policy(check_sensitivity=1.5),
        lambda: Policy(hard_dependence="worst"),
    ):
        try:
            bad()
        except ValueError:
            continue
        raise AssertionError("expected ValueError")
