"""Value layer theorems L1–L6, checked against brute-force enumeration of joint outcomes."""

from __future__ import annotations

import itertools
import math

import numpy as np
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from cdm import Competitor, Lottery, expected_improvement, expected_max, p_best
from cdm.lottery import TIE_TOL

from .strategies import close, lottery

alphas = st.floats(0.01, 1.0)


def iso(a: Lottery, b: Lottery, tol: float = 1e-9) -> bool:
    return a.x.size == b.x.size and close(a.x, b.x, tol) and close(a.p, b.p, tol)


# well-formedness ---------------------------------------------------------------------------
@given(lottery())
def test_lottery_is_a_normalised_measure_on_sorted_distinct_atoms(x: Lottery) -> None:
    assert abs(x.p.sum() - 1) < 1e-12
    assert np.all(x.p > 0)
    assert np.all(np.diff(x.x) > 0)
    assert x.cdf(x.x[-1]) == 1.0
    assert x.cdf_left(x.x[0]) == 0.0
    assert np.all(np.diff(x.cdf(np.linspace(-6, 6, 50))) >= 0)


# L1 ----------------------------------------------------------------------------------------
@given(lottery(), lottery(), lottery())
def test_l1_independent_sum_is_commutative_and_associative(
    a: Lottery, b: Lottery, c: Lottery
) -> None:
    assert iso(a + b, b + a)
    assert iso((a + b) + c, a + (b + c), 1e-8)
    assert iso(a * b, b * a)


@given(lottery(), lottery(), st.floats(-3, 3), st.floats(0, 1))
def test_l1_expectation_is_linear(a: Lottery, b: Lottery, c: float, q: float) -> None:
    assert math.isclose((a + b).mean(), a.mean() + b.mean(), abs_tol=1e-9)
    assert math.isclose((a * b).mean(), a.mean() * b.mean(), abs_tol=1e-9)
    assert math.isclose(a.scale(c).mean(), c * a.mean(), abs_tol=1e-9)
    assert math.isclose(a.mix(b, q).mean(), (1 - q) * a.mean() + q * b.mean(), abs_tol=1e-9)


@given(lottery(), lottery())
def test_l1_variance_adds_for_independent_sums(a: Lottery, b: Lottery) -> None:
    assert math.isclose((a + b).var(), a.var() + b.var(), abs_tol=1e-8)


# L2 ----------------------------------------------------------------------------------------
@given(lottery(), alphas)
def test_l2_cvar_is_bounded_by_the_mean(x: Lottery, a: float) -> None:
    assert x.x[0] - 1e-9 <= x.cvar(a) <= x.mean() + 1e-9
    assert math.isclose(x.cvar(1.0), x.mean(), abs_tol=1e-9)


@given(lottery(), alphas, st.floats(-3, 3), st.floats(0.01, 3))
def test_l2_cvar_is_translation_equivariant_and_homogeneous(
    x: Lottery, a: float, c: float, k: float
) -> None:
    assert math.isclose(x.shift(c).cvar(a), x.cvar(a) + c, abs_tol=1e-8)
    assert math.isclose(x.scale(k).cvar(a), k * x.cvar(a), abs_tol=1e-8)


@given(lottery(), lottery(), alphas)
def test_l2_cvar_is_superadditive(x: Lottery, y: Lottery, a: float) -> None:
    assert (x + y).cvar(a) >= x.cvar(a) + y.cvar(a) - 1e-8


@given(lottery(), alphas, alphas)
def test_l2_cvar_is_monotone_in_the_tail(x: Lottery, a: float, b: float) -> None:
    lo, hi = sorted((a, b))
    assert x.cvar(lo) <= x.cvar(hi) + 1e-9


@given(lottery(), st.lists(st.floats(0, 2), min_size=6, max_size=6), alphas)
def test_l2_cvar_and_mean_respect_dominance(x: Lottery, ups: list[float], a: float) -> None:
    better = Lottery.of(x.x + np.array(ups[: x.x.size]), x.p)
    assert better.dominates(x)
    assert better.cvar(a) >= x.cvar(a) - 1e-9
    assert better.mean() >= x.mean() - 1e-9


# L3 ----------------------------------------------------------------------------------------
@given(st.lists(st.floats(0, 1), min_size=20, max_size=60, unique=True), st.integers(3, 15), alphas)
def test_l3_compaction_keeps_mass_and_mean_and_bounds_cvar(
    xs: list[float], grid: int, a: float
) -> None:
    ps = np.linspace(1, 2, len(xs))
    exact = Lottery.of(xs, ps)
    assume(exact.x.size > grid and exact.x[-1] > exact.x[0])
    small = Lottery.of(xs, ps, max_atoms=grid)
    assert small.x.size <= grid
    assert abs(small.p.sum() - 1) < 1e-12
    assert math.isclose(small.mean(), exact.mean(), abs_tol=1e-9)
    h = (exact.x[-1] - exact.x[0]) / (grid - 1)
    assert abs(small.cvar(a) - exact.cvar(a)) <= h + 1e-9


# L4, L5: brute-force oracle -------------------------------------------------------------------
def outcomes(c: Competitor) -> list[tuple[float, float]]:
    out = [(float(x), (1 - c.fail) * float(p)) for x, p in zip(c.value.x, c.value.p, strict=True)]
    return [*out, (-math.inf, c.fail)] if c.fail > 0 else out


def brute(cs: list[Competitor]) -> tuple[np.ndarray, float]:
    share, emax = np.zeros(len(cs)), 0.0
    for combo in itertools.product(*(outcomes(c) for c in cs)):
        prob = math.prod(p for _, p in combo)
        vals = [v for v, _ in combo]
        top = max(vals)
        winners = [i for i, v in enumerate(vals) if v == top]
        for i in winners:
            share[i] += prob / len(winners)
        emax += prob * top
    return share, emax


@st.composite
def race(draw: st.DrawFn) -> list[Competitor]:
    n = draw(st.integers(1, 4))
    grid = st.sampled_from([0.0, 0.25, 0.5, 0.75, 1.0])  # coarse grid forces ties
    cs = [Competitor(draw(lottery_on(grid)), 0.0)]
    cs += [
        Competitor(draw(lottery_on(grid)), draw(st.sampled_from([0.0, 0.3, 1.0]))) for _ in range(n)
    ]
    return cs


@st.composite
def lottery_on(draw: st.DrawFn, grid: st.SearchStrategy[float]) -> Lottery:
    xs = draw(st.lists(grid, min_size=1, max_size=3))
    ps = draw(st.lists(st.floats(0.05, 1), min_size=len(xs), max_size=len(xs)))
    return Lottery.of(xs, ps)


@settings(max_examples=150)
@given(race())
def test_l4_l5_p_best_and_expected_max_match_enumeration(cs: list[Competitor]) -> None:
    share, emax = brute(cs)
    got = p_best(cs)
    assert close(got, share, 1e-9)
    assert abs(got.sum() - 1) < 1e-9
    assert math.isclose(expected_max(cs), emax, abs_tol=1e-9)


@settings(max_examples=60)
@given(race(), st.randoms())
def test_l4_p_best_is_permutation_equivariant(cs: list[Competitor], rnd: object) -> None:
    perm = list(range(len(cs)))
    rnd.shuffle(perm)  # type: ignore[attr-defined]
    assert close(p_best([cs[i] for i in perm]), p_best(cs)[perm])


@given(lottery(), st.integers(2, 5))
def test_l4_identical_competitors_share_equally(x: Lottery, n: int) -> None:
    assert close(p_best([Competitor(x)] * n), np.full(n, 1 / n), 1e-9)


@given(st.lists(lottery(), min_size=1, max_size=4))
def test_l5_evpi_is_non_negative(vs: list[Lottery]) -> None:
    """E[max_i U_i] ≥ max_i E[U_i] (Jensen: max is convex)."""
    cs = [Competitor(v) for v in vs]
    assert expected_max(cs) >= max(v.mean() for v in vs) - 1e-9


@given(st.lists(st.floats(-3, 3), min_size=1, max_size=5))
def test_l5_no_uncertainty_no_value_of_information(vals: list[float]) -> None:
    cs = [Competitor(Lottery.point(v)) for v in vals]
    # values closer than the tie tolerance are one value, so E[max] may sit at the smaller one
    assert abs(expected_max(cs) - max(vals)) <= TIE_TOL * max(1.0, abs(max(vals)))


# L6 ----------------------------------------------------------------------------------------
@given(lottery(), lottery(), st.floats(0.01, 2), st.floats(0, 1))
def test_l6_ei_rewards_mean_preserving_spreads(
    v: Lottery, base: Lottery, h: float, fail: float
) -> None:
    k = int(np.argmax(v.p))
    spread = Lottery.of(
        np.concatenate([np.delete(v.x, k), [v.x[k] - h, v.x[k] + h]]),
        np.concatenate([np.delete(v.p, k), [v.p[k] / 2, v.p[k] / 2]]),
    )
    assert math.isclose(spread.mean(), v.mean(), abs_tol=1e-9)
    assert expected_improvement(spread, base, fail) >= expected_improvement(v, base, fail) - 1e-9


@given(lottery(), lottery(), st.floats(0, 1))
def test_l6_ei_is_at_least_the_mean_gain_and_vanishes_on_certain_failure(
    v: Lottery, base: Lottery, fail: float
) -> None:
    ei = expected_improvement(v, base, fail)
    assert ei >= (1 - fail) * max(v.mean() - base.mean(), 0) - 1e-9  # Jensen
    assert expected_improvement(v, base, 1.0) == 0
