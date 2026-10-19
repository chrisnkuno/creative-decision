"""Hypothesis strategies shared by the invariant suites."""

from __future__ import annotations

import numpy as np
from hypothesis import strategies as st
from numpy.typing import ArrayLike

from cdm import Lottery
from cdm.simplex import Vec, closure

finite = st.floats(-5, 5, allow_nan=False, allow_infinity=False)
pos_mass = st.floats(0.01, 1.0)


@st.composite
def simplex(draw: st.DrawFn, k: int | None = None, min_k: int = 2, max_k: int = 6) -> Vec:
    n = k if k is not None else draw(st.integers(min_k, max_k))
    return closure(draw(st.lists(pos_mass, min_size=n, max_size=n)))


@st.composite
def same_size_simplices(draw: st.DrawFn, count: int, max_k: int = 6) -> list[Vec]:
    k = draw(st.integers(2, max_k))
    return [draw(simplex(k)) for _ in range(count)]


@st.composite
def lottery(draw: st.DrawFn, max_atoms: int = 6, lo: float = -5, hi: float = 5) -> Lottery:
    n = draw(st.integers(1, max_atoms))
    xs = draw(st.lists(st.floats(lo, hi), min_size=n, max_size=n))
    ps = draw(st.lists(pos_mass, min_size=n, max_size=n))
    return Lottery.of(xs, ps)


def close(a: ArrayLike, b: ArrayLike, tol: float = 1e-9) -> bool:
    return bool(np.allclose(a, b, atol=tol, rtol=tol))
