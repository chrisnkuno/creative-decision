"""Aitchison geometry: (Δ°, ⊕, ⊙) is a real vector space and clr is a linear isomorphism."""

from __future__ import annotations

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from cdm.simplex import (
    Vec,
    clr,
    clr_inv,
    combine,
    cyclic_orbit,
    distance,
    interior,
    inverse,
    perturb,
    power,
    present,
    relabel,
    reversal_orbit,
    uniform,
)

from .strategies import close, same_size_simplices, simplex

scalar = st.floats(-3, 3)


@given(simplex())
def test_clr_is_a_bijection_onto_the_zero_sum_hyperplane(p: Vec) -> None:
    assert abs(clr(p).sum()) < 1e-9
    assert close(clr_inv(clr(p)), p)


@given(same_size_simplices(3))
def test_perturbation_is_an_abelian_group(ps: list[Vec]) -> None:
    p, q, r = ps
    u = uniform(p.size)
    assert close(perturb(p, q), perturb(q, p))
    assert close(perturb(perturb(p, q), r), perturb(p, perturb(q, r)))
    assert close(perturb(p, u), p)
    assert close(perturb(p, inverse(p)), u)


@given(same_size_simplices(2), scalar, scalar)
def test_powering_is_a_scalar_multiplication(ps: list[Vec], a: float, b: float) -> None:
    p, q = ps
    assert close(power(1.0, p), p)
    assert close(power(a, power(b, p)), power(a * b, p))
    assert close(power(a + b, p), perturb(power(a, p), power(b, p)))
    assert close(power(a, perturb(p, q)), perturb(power(a, p), power(a, q)))


@given(same_size_simplices(2), scalar)
def test_clr_is_linear(ps: list[Vec], a: float) -> None:
    p, q = ps
    assert close(clr(perturb(p, q)), clr(p) + clr(q))
    assert close(clr(power(a, p)), a * clr(p))


@given(same_size_simplices(3), st.lists(scalar, min_size=3, max_size=3))
def test_combine_is_the_linear_combination(ps: list[Vec], w: list[float]) -> None:
    expected = perturb(*[power(wi, p) for p, wi in zip(ps, w, strict=True)])
    assert close(combine(ps, w), expected)


@given(same_size_simplices(3))
def test_aitchison_distance_is_a_metric(ps: list[Vec]) -> None:
    p, q, r = ps
    assert distance(p, p) < 1e-9
    assert abs(distance(p, q) - distance(q, p)) < 1e-9
    assert distance(p, r) <= distance(p, q) + distance(q, r) + 1e-9
    # perturbation is an isometry (translation invariance)
    assert abs(distance(perturb(p, r), perturb(q, r)) - distance(p, q)) < 1e-9


@given(simplex(), st.floats(0, 1))
def test_interior_stays_in_the_simplex(p: Vec, eta: float) -> None:
    q = interior(p, eta)
    assert abs(q.sum() - 1) < 1e-12
    assert np.all(q >= min(p.min(), 1 / p.size) - 1e-12)
    assert close(interior(p, 0.0), p)


@given(st.integers(2, 8), st.data())
def test_relabel_inverts_present_for_every_orbit_member(n: int, data: st.DataObject) -> None:
    p = data.draw(simplex(n))
    for order in cyclic_orbit(n) + reversal_orbit(n):
        assert sorted(order) == list(range(n))
        assert close(relabel(present(p, order), order), p)


@given(st.integers(2, 8))
def test_cyclic_orbit_is_transitive_and_balanced(n: int) -> None:
    seen = np.zeros((n, n), dtype=int)  # label × position
    for order in cyclic_orbit(n):
        for pos, label in enumerate(order):
            seen[label, pos] += 1
    assert np.all(seen == 1)
