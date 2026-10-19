"""Value layer: exact arithmetic on finite probability measures on ℝ ("lotteries").

A Lottery is μ = Σ_k p_k δ_{x_k} with sorted, distinct atoms x_k and masses p_k > 0, Σ p_k = 1.
Atoms closer than a relative 1e-9 are merged, so float noise never creates or breaks a tie.
Binary operations combine *independent* random variables, so `X + X` is the convolution of
two independent copies, not 2X.

Above `max_atoms`, atoms are moved onto a uniform grid by splitting each atom's mass between its
two neighbouring grid points in proportion to distance. This keeps total mass and the mean
exactly, and moves any quantile or CVaR by at most one grid step.

Theorems (tested):

  L1  (+, ·) on independent lotteries are commutative and associative; E is linear;
      E[mix(X, Y; q)] = (1−q)E[X] + qE[Y].
  L2  The lower-tail mean CVaR_α (a reward measure) is FSD-monotone, translation-equivariant,
      positively homogeneous, superadditive (CVaR(X+Y) ≥ CVaR(X)+CVaR(Y)), and satisfies
      CVaR_α ≤ E with CVaR_1 = E.
  L3  Compaction preserves mass and mean, and moves CVaR by at most one grid step.
  L4  For independent competitors (V_j with failure mass q_j ↦ −∞), P(best) with fair splitting
      of ties is  Σ_x p_i(x) ∫₀¹ Π_{j≠i} (F_j(x⁻) + p_j(x) t) dt,  because ∫₀¹ t^k dt = 1/(k+1)
      averages 1/(1+#ties). It sums to 1 and is equivariant under permuting competitors.
  L5  E[max_j U_j] = Σ_x x ΔΠ_j F_j(x);  EVPI = E[max] − max E ≥ 0 (U_j ≤ max_k U_k).
  L6  EI(V; B) = E[max(V − B, 0)] is monotone in the convex order: a mean-preserving spread of V
      never lowers it.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Vec = NDArray[np.float64]
MAX_ATOMS = 4096
TIE_TOL = 1e-9  # relative gap below which two values are equal (atoms merge, ties split)


def _normalise(x: Vec, p: Vec, max_atoms: int) -> tuple[Vec, Vec]:
    """Sort, drop zero mass, and merge atoms that agree up to float noise.

    Rounding each result to fixed decimals is not enough: errors accumulate along different
    computation paths, so equal values could stay distinct atoms and break tie handling (L4).
    A merged group keeps its smallest value, so exact duplicates stay exact and no atom moves by
    more than the relative tolerance.
    """
    keep = p > 0
    x, p = x[keep], p[keep]
    order = np.argsort(x, kind="stable")
    x, p = x[order], p[order]
    group = _groups(x)
    first = np.concatenate([[True], np.diff(group) > 0])
    m = np.bincount(group, weights=p)
    u = x[first]
    m = m / m.sum()
    if u.size > max_atoms:
        u, m = _compact(u, m, max_atoms)
    return u, m


def _compact(x: Vec, p: Vec, g: int) -> tuple[Vec, Vec]:
    lo, hi = float(x[0]), float(x[-1])
    grid = np.linspace(lo, hi, g)
    pos = (x - lo) / (hi - lo) * (g - 1)
    i = np.clip(np.floor(pos).astype(int), 0, g - 2)
    frac = pos - i
    m = np.zeros(g)
    np.add.at(m, i, p * (1 - frac))
    np.add.at(m, i + 1, p * frac)
    keep = m > 0
    return grid[keep], m[keep]


@dataclass(frozen=True)
class Lottery:
    x: Vec
    p: Vec

    # construction -------------------------------------------------------------------------
    @staticmethod
    def of(
        x: Sequence[float] | Vec, p: Sequence[float] | Vec, max_atoms: int = MAX_ATOMS
    ) -> Lottery:
        xa, pa = np.asarray(x, dtype=float), np.asarray(p, dtype=float)
        if xa.shape != pa.shape or xa.ndim != 1 or xa.size == 0:
            raise ValueError("support and masses must be equal-length non-empty vectors")
        if np.any(pa < 0) or not np.isfinite(xa).all() or pa.sum() <= 0:
            raise ValueError("masses must be non-negative with positive total; atoms finite")
        return Lottery(*_normalise(xa, pa, max_atoms))

    @staticmethod
    def point(v: float) -> Lottery:
        return Lottery(np.array([float(v)]), np.array([1.0]))

    @staticmethod
    def bernoulli(ok: float, fail: float, q: float) -> Lottery:
        """`ok` with probability 1−q, `fail` with probability q."""
        return Lottery.of([ok, fail], [1 - q, q])

    # algebra (independent copies) ---------------------------------------------------------
    def combine(self, other: Lottery, op: Callable[[Vec, Vec], Vec]) -> Lottery:
        xs = op(self.x[:, None], other.x[None, :]).ravel()
        ps = (self.p[:, None] * other.p[None, :]).ravel()
        return Lottery.of(xs, ps)

    def __add__(self, other: Lottery) -> Lottery:
        return self.combine(other, np.add)

    def __mul__(self, other: Lottery) -> Lottery:
        return self.combine(other, np.multiply)

    def scale(self, c: float) -> Lottery:
        return Lottery.of(self.x * c, self.p)

    def shift(self, c: float) -> Lottery:
        return Lottery.of(self.x + c, self.p)

    def mix(self, other: Lottery, q: float) -> Lottery:
        """self with probability 1−q, other with probability q."""
        return Lottery.of(
            np.concatenate([self.x, other.x]), np.concatenate([(1 - q) * self.p, q * other.p])
        )

    # functionals --------------------------------------------------------------------------
    def mean(self) -> float:
        return float(self.x @ self.p)

    def var(self) -> float:
        return float(((self.x - self.mean()) ** 2) @ self.p)

    def std(self) -> float:
        return float(np.sqrt(max(self.var(), 0.0)))

    def _cum(self) -> Vec:
        """[0, F(x_1), …, F(x_n)] with F(x_n) = 1 exactly (cumsum can overshoot by an ulp)."""
        c: Vec = np.minimum(np.concatenate([[0.0], np.cumsum(self.p)]), 1.0)
        c[-1] = 1.0
        return c

    def cdf(self, t: Vec | float) -> Vec:
        """F(t) = P(X ≤ t)."""
        return self._cum()[np.searchsorted(self.x, t, side="right")]

    def cdf_left(self, t: Vec | float) -> Vec:
        """F(t⁻) = P(X < t)."""
        return self._cum()[np.searchsorted(self.x, t, side="left")]

    def cvar(self, alpha: float) -> float:
        """Mean of the worst α of the probability mass (lower tail)."""
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")
        c = np.cumsum(self.p)
        take = np.clip(np.minimum(c, alpha) - (c - self.p), 0, None)
        return float(take @ self.x / alpha)

    def dominates(self, other: Lottery, tol: float = 1e-12) -> bool:
        """First-order stochastic dominance: F_self ≤ F_other everywhere."""
        grid = np.union1d(self.x, other.x)
        return bool(np.all(self.cdf(grid) <= other.cdf(grid) + tol))


@dataclass(frozen=True)
class Competitor:
    """An action in a race: value V if it goes ahead, removed (−∞) with probability `fail`."""

    value: Lottery
    fail: float = 0.0


def _groups(x: Vec) -> NDArray[np.intp]:
    """Group index of each sorted value; values closer than the merge tolerance share a group."""
    gap = np.diff(x) > TIE_TOL * np.maximum(1.0, np.abs(x[1:]))
    return np.concatenate([[0], np.cumsum(gap)]).astype(np.intp)


def _tables(cs: Sequence[Competitor]) -> tuple[Vec, Vec, Vec]:
    """Common value grid with, per competitor, P(below the grid point) and P(at it).

    Atoms from different competitors that agree up to float noise land on one grid point, so
    they count as a tie.
    """
    allx = np.concatenate([c.value.x for c in cs])
    owner = np.concatenate([np.full(c.value.x.size, k, dtype=np.intp) for k, c in enumerate(cs)])
    allp = np.concatenate([(1 - c.fail) * c.value.p for c in cs])
    order = np.argsort(allx, kind="stable")
    g = _groups(allx[order])
    n = int(g[-1]) + 1
    grid = allx[order][np.concatenate([[True], np.diff(g) > 0])]
    mass = np.zeros((len(cs), n))
    np.add.at(mass, (owner[order], g), allp[order])
    fail = np.array([c.fail for c in cs])[:, None]
    left = fail + np.cumsum(mass, axis=1) - mass
    return grid, left, mass


def p_best(cs: Sequence[Competitor]) -> Vec:
    """Exact P(competitor i attains the maximum), ties split fairly (theorem L4).

    Requires at least one competitor that never fails, so a maximum always exists.
    """
    if not any(c.fail == 0 for c in cs):
        raise ValueError("at least one competitor must never fail")
    _, left, mass = _tables(cs)
    n = left.shape[0]
    out = np.zeros(n)
    for i in range(n):
        for k in np.nonzero(mass[i] > 0)[0]:
            poly = np.array([1.0])
            for j in range(n):
                if j != i:
                    poly = np.convolve(poly, [left[j, k], mass[j, k]])  # a + b t, low→high
            out[i] += mass[i, k] * float(poly @ (1.0 / np.arange(1, poly.size + 1)))
    return out


def expected_max(cs: Sequence[Competitor]) -> float:
    if not any(c.fail == 0 for c in cs):
        raise ValueError("at least one competitor must never fail")
    grid, left, mass = _tables(cs)
    at_or_below = np.prod(left + mass, axis=0)
    below = np.prod(left, axis=0)
    return float(grid @ (at_or_below - below))


def expected_improvement(v: Lottery, base: Lottery, fail: float = 0.0) -> float:
    """E[max(U − B, 0)] where U = V with prob 1−fail, else B (so the failure branch adds 0)."""
    return (1 - fail) * v.combine(base, lambda a, b: np.maximum(a - b, 0.0)).mean()
