"""Mechanism layer: M1–M5 (property-based), its hand-off to the policy, and the judge questions."""

from __future__ import annotations

import asyncio
from dataclasses import replace

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from cdm import (
    NEW,
    Beta,
    Function,
    Judgment,
    MockJudge,
    Morphology,
    Option,
    Policy,
    decide,
    judge_problem,
)
from cdm.decide import MECHANISM, apply_mechanism
from cdm.judges import build_questions
from cdm.mechanism import next_decision_value, next_decision_value_bound, shared_principles
from cdm.simplex import Vec

from .judgments import judgment, point, problem
from .strategies import simplex

FUNCS = [
    Function("reach", "reach merchants", {"door": "visit in person", "ads": "paid ads"}, 2.0),
    Function("pay", "collect payment", {"cash": "cash", "card": "card", "momo": "mobile money"}),
]
RECORD = [
    {"reach": "door", "pay": "cash"},
    {"reach": "door", "pay": "cash"},
    {"reach": "ads", "pay": "card"},
    {"reach": "door"},
]

beta = st.builds(Beta, st.floats(0.05, 50), st.floats(0.05, 50))
prob = st.floats(0, 1)


def morph(**kw: object) -> Morphology:
    return Morphology(list(FUNCS), [dict(r) for r in RECORD], **kw)  # type: ignore[arg-type]


@st.composite
def classification(draw: st.DrawFn) -> dict[str, Vec]:
    return {f.id: draw(simplex(len(f.labels()))) for f in FUNCS}


# M1 ----------------------------------------------------------------------------------------
@settings(max_examples=100, deadline=None)
@given(classification())
def test_m1_novelty_and_combination_novelty_are_probabilities(pi: dict[str, Vec]) -> None:
    m = morph()
    assert -1e-12 <= m.novelty(pi) <= 1 + 1e-12
    assert 0 <= m.combination_novelty(pi) <= 1


@settings(max_examples=100, deadline=None)
@given(classification(), classification(), st.floats(0, 1))
def test_m1_novelty_is_affine_in_the_classification(
    a: dict[str, Vec], b: dict[str, Vec], t: float
) -> None:
    m = morph()
    mix = {f: (1 - t) * a[f] + t * b[f] for f in a}
    assert abs(m.novelty(mix) - ((1 - t) * m.novelty(a) + t * m.novelty(b))) < 1e-9


@settings(max_examples=100, deadline=None)
@given(classification(), st.sampled_from(FUNCS), st.data())
def test_m1_mass_moved_to_a_rarer_principle_never_lowers_novelty(
    pi: dict[str, Vec], f: Function, data: st.DataObject
) -> None:
    m = morph()
    labels = f.labels()
    src, dst = data.draw(st.permutations(range(len(labels))))[:2]
    assume(m.principle_novelty(f.id, labels[dst]) >= m.principle_novelty(f.id, labels[src]))
    moved = dict(pi)
    v = pi[f.id].copy()
    amount = data.draw(st.floats(0, 1)) * v[src]
    v[src] -= amount
    v[dst] += amount
    moved[f.id] = v
    assert m.novelty(moved) >= m.novelty(pi) - 1e-12


def test_m1_shah_novelty_values() -> None:
    m = morph()
    assert m.principle_novelty("reach", "door") == 0.0
    assert m.principle_novelty("pay", "card") == 0.0
    shah = morph(familiar_after=None)
    assert shah.principle_novelty("reach", "door") == pytest.approx(1 - 3 / 4)
    assert shah.principle_novelty("pay", "cash") == pytest.approx(1 - 2 / 3)
    assert morph(familiar_after=2).principle_novelty("pay", "card") == pytest.approx(0.5)
    assert m.principle_novelty("pay", "momo") == 1.0
    assert m.principle_novelty("pay", NEW) == 1.0
    assert Morphology(list(FUNCS)).principle_novelty("reach", "door") == 1.0  # no record
    known = m.classification({"reach": "door", "pay": "cash"})
    assert m.combination_novelty(known) == pytest.approx(0.0)
    assert m.combination_novelty(m.classification({"reach": "door", "pay": "card"})) == 1.0


# M2 ----------------------------------------------------------------------------------------
@settings(max_examples=60, deadline=None)
@given(st.sampled_from(FUNCS), st.data())
def test_m2_a_new_precedent_lowers_only_its_own_principle(f: Function, data: st.DataObject) -> None:
    familiar = data.draw(st.sampled_from([None, 1, 2, 3]))
    m = morph(familiar_after=familiar)
    p = data.draw(st.sampled_from(list(f.principles)))
    more = Morphology(list(FUNCS), [*m.record, {f.id: p}], familiar_after=familiar)
    assert more.principle_novelty(f.id, p) <= m.principle_novelty(f.id, p)
    for g in FUNCS:
        for other in g.labels():
            if (g.id, other) != (f.id, p):
                assert more.principle_novelty(g.id, other) >= m.principle_novelty(g.id, other)


# M3 ----------------------------------------------------------------------------------------
@settings(max_examples=100, deadline=None)
@given(classification(), prob, prob, st.sampled_from(FUNCS))
def test_m3_mechanism_risk_is_monotone(
    pi: dict[str, Vec], r1: float, r2: float, f: Function
) -> None:
    m = morph()
    lo, hi = sorted((r1, r2))
    q = [1 - m.works(pi, {f.id: r}) for r in (lo, hi)]
    assert 0 <= q[0] <= q[1] + 1e-12 <= 1 + 1e-12
    p = "momo" if f.id == "pay" else "ads"
    worse = morph(evidence={(f.id, p): Beta(1, 3)})
    better = morph(evidence={(f.id, p): Beta(3, 1)})
    assert worse.works(pi, {}) <= better.works(pi, {}) + 1e-12


# M4 ----------------------------------------------------------------------------------------
@settings(max_examples=200, deadline=None)
@given(beta, st.booleans(), st.floats(0.5, 1), st.floats(0.5, 1))
def test_m4_moment_matched_test_update(b: Beta, passed: bool, s: float, c: float) -> None:
    post = b.observe_test(passed=passed, sens=s, spec=c)
    # exact posterior from the two-component mixture
    like1, like0 = (c, 1 - s) if passed else (1 - c, s)
    w1, w0 = b.mean * like1, (1 - b.mean) * like0
    hi, lo = b.observe(works=True), b.observe(works=False)
    mean = (w1 * hi.mean + w0 * lo.mean) / (w1 + w0)
    m2 = w1 * hi.a * (hi.a + 1) / (hi.n * (hi.n + 1)) + w0 * lo.a * (lo.a + 1) / (lo.n * (lo.n + 1))
    var = m2 / (w1 + w0) - mean**2
    assert post.mean == pytest.approx(mean, rel=1e-9, abs=1e-12)
    assert post.var == pytest.approx(var, rel=1e-6, abs=1e-12)
    assert b.p_pass(s, c) == pytest.approx(b.mean * c + (1 - b.mean) * (1 - s))


@settings(max_examples=100, deadline=None)
@given(beta, st.booleans())
def test_m4_perfect_tests_are_conjugate_and_useless_tests_change_nothing(
    b: Beta, passed: bool
) -> None:
    exact = b.observe(works=passed)
    post = b.observe_test(passed=passed)
    assert (post.a, post.b) == pytest.approx((exact.a, exact.b), rel=1e-9)
    assert b.observe_test(passed=passed, sens=0.7, spec=0.3) == b  # s + c = 1


def test_m4_tests_update_the_shared_belief() -> None:
    m = morph()
    risk = {"pay": 0.3}
    before = m.works(m.classification({"reach": "ads", "pay": "momo"}), risk)
    m.update("pay", "momo", passed=False, sens=0.9, spec=0.95)
    after = m.works(m.classification({"reach": "door", "pay": "momo"}), risk)
    assert m.evidence[("pay", "momo")].mean < 0.5
    assert after < before  # a different option that shares the principle is now riskier
    assert m.settled("pay", "cash")
    m.update("pay", "cash", passed=False)
    assert not m.settled("pay", "cash")
    assert m.evidence[("pay", "cash")] == Beta(3, 2)  # Beta(1 + 2 uses, 1), then one failure


# M5 ----------------------------------------------------------------------------------------
@settings(max_examples=300, deadline=None)
@given(beta, st.floats(0, 1), st.floats(0.01, 10))
def test_m5_value_of_one_test_for_the_next_decision(b: Beta, t: float, alpha: float) -> None:
    v = next_decision_value(b, t, alpha)
    bound = next_decision_value_bound(b, alpha)
    assert -1e-12 <= v <= bound + 1e-12
    lo, hi = b.observe(works=False).mean, b.observe(works=True).mean
    assert hi - lo == pytest.approx(1 / (b.n + 1))
    if not lo < t < hi:
        assert v == pytest.approx(0, abs=1e-12)


@settings(max_examples=100, deadline=None)
@given(beta, st.floats(0.01, 10))
def test_m5_bound_is_attained_at_the_mean(b: Beta, alpha: float) -> None:
    assert next_decision_value(b, b.mean, alpha) == pytest.approx(
        next_decision_value_bound(b, alpha), rel=1e-9, abs=1e-15
    )


def test_m5_ignorance_teaches_more_than_a_known_coin() -> None:
    ratio = next_decision_value_bound(Beta(1, 1)) / next_decision_value_bound(Beta(50, 50))
    assert ratio == pytest.approx(101 / 3)


# Policy hand-off ---------------------------------------------------------------------------
def scenario_with_mechanisms() -> tuple[object, list[Judgment]]:
    pr = replace(problem(hard=("harm",), soft=()), morphology=morph())
    m = pr.morphology
    assert m is not None

    def j(oid: str, level: int, choice: dict[str, str], *, base: bool = False) -> Judgment:
        out = judgment(oid, pr, point(level), novelty=0.0 if base else 0.9, baseline=base)
        return replace(out, mechanism=m.classification(choice), mechanism_risk={})

    js = [
        j("base", 2, {"reach": "door", "pay": "cash"}, base=True),
        j("reword", 3, {"reach": "door", "pay": "cash"}),  # judged novel, known mechanism
        j("momo", 3, {"reach": "door", "pay": "momo"}),  # novel principle, never tested
    ]
    return pr, js


def test_unknown_mechanism_is_verified_not_vetoed() -> None:
    pr, js = scenario_with_mechanisms()
    d = decide(pr, js)  # type: ignore[arg-type]
    momo = d.by_id("momo")
    assert momo.status == "verify"
    assert MECHANISM.id in momo.checks
    assert momo.novelty > d.by_id("reword").novelty
    assert d.by_id("reword").novelty == 0.0
    assert d.by_id("reword").status == "clear"  # its principles are settled by precedent


def test_rewording_does_not_take_the_innovation_slot() -> None:
    pr, js = scenario_with_mechanisms()
    d = decide(pr, js, Policy(min_novelty=0.4, downside_tolerance=1.0))  # type: ignore[arg-type]
    assert d.innovation in {None, "momo"}
    assert d.by_id("reword").novelty < 0.4


def test_apply_mechanism_is_idempotent_and_settles_the_baseline() -> None:
    pr, js = scenario_with_mechanisms()
    p1, j1 = apply_mechanism(pr, js)  # type: ignore[arg-type]
    p2, j2 = apply_mechanism(p1, j1)
    assert [c.id for c in p1.constraints] == [c.id for c in p2.constraints]
    assert [x.violations for x in j1] == [x.violations for x in j2]
    assert j1[0].violations[MECHANISM.id] == 0.0


def test_missing_classification_is_an_error() -> None:
    pr, js = scenario_with_mechanisms()
    with pytest.raises(ValueError, match="classification"):
        decide(pr, [js[0], replace(js[2], mechanism={})])  # type: ignore[arg-type]


@settings(max_examples=40, deadline=None)
@given(st.floats(0, 5))
def test_learning_weight_only_moves_the_innovation_slot(w: float) -> None:
    pr, js = scenario_with_mechanisms()
    a = decide(pr, js)  # type: ignore[arg-type]
    b = decide(pr, js, Policy(learning_weight=w))  # type: ignore[arg-type]
    assert a.commit == b.commit
    assert a.portfolio == b.portfolio
    assert a.evpi == b.evpi
    assert [s.mean for s in a.stats] == [s.mean for s in b.stats]


def test_contracts() -> None:
    with pytest.raises(ValueError, match="reserved"):
        Function("f", "x", {NEW: "reserved"}).labels()
    with pytest.raises(ValueError, match="unique"):
        Morphology([*FUNCS, FUNCS[0]])
    with pytest.raises(ValueError, match="unknown function"):
        Morphology(list(FUNCS), [{"reach": "telepathy"}])
    with pytest.raises(ValueError, match="positive sum"):
        Morphology([Function("f", "x", {"a": "a"}, weight=0.0)])
    with pytest.raises(ValueError, match="positive"):
        Beta(0, 1)
    with pytest.raises(ValueError, match="probability zero"):
        Beta(1, 1).observe_test(passed=True, sens=1.0, spec=0.0)
    with pytest.raises(ValueError, match="untested"):
        morph().belief("pay", NEW)
    with pytest.raises(ValueError, match="catalogue"):
        morph().update("pay", NEW, passed=True)
    with pytest.raises(ValueError, match="familiar_after"):
        morph(familiar_after=0)
    with pytest.raises(ValueError, match="distribution"):
        morph().novelty({"reach": np.array([1.0, 0.0, 0.0]), "pay": np.array([0.5, 0.5])})
    with pytest.raises(ValueError, match="learning_weight"):
        Policy(learning_weight=-1)
    assert Morphology([]).combination_novelty({}) == 0.0


def test_shared_principles_counts_the_reach_of_a_test() -> None:
    m = morph()
    reach = shared_principles(m, [{"reach": "door", "pay": "momo"}, {"reach": NEW, "pay": "momo"}])
    assert reach == {("reach", "door"): 1, ("pay", "momo"): 2}


def test_learning_value_prefers_untested_principles() -> None:
    m = morph()
    tested = m.classification({"reach": "door", "pay": "cash"})
    untested = m.classification({"reach": "door", "pay": "momo"})
    assert m.learning(untested, {}) > m.learning(tested, {}) == 0.0
    assert m.learning(m.classification({"reach": "door", "pay": NEW}), {"pay": 0.5}) > 0


# Judge questions ---------------------------------------------------------------------------
def test_questions_ask_for_the_principle_of_each_function() -> None:
    pr = replace(problem(), morphology=morph())
    qs, orbits = build_questions(pr)
    assert len(orbits["mech:pay"]) == len(FUNCS[1].labels())  # one cyclic orbit member per label
    assert set(qs["mech:pay@0"]["criteria"]) == set(FUNCS[1].labels())
    assert qs["risk:pay"]["type"] == "noul"


def test_mock_pipeline_with_mechanisms_runs_end_to_end() -> None:
    pr = replace(
        problem(),
        morphology=morph(),
        options=[Option("base", "door to door, cash", baseline=True), Option("x", "x")],
    )
    js = asyncio.run(judge_problem(pr, [MockJudge()]))
    for j in js:
        assert set(j.mechanism) == {"reach", "pay"}
        assert all(abs(v.sum() - 1) < 1e-9 for v in j.mechanism.values())
        assert all(0 <= r <= 1 for r in j.mechanism_risk.values())
    d = decide(pr, js)
    assert d.by_id("x").status in {"clear", "verify", "reject"}
    assert 0 <= d.by_id("x").combination <= 1
