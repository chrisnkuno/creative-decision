"""Aitchison geometry of the probability simplex.

The open simplex Δ°_K = {p ∈ ℝ^K : p_i > 0, Σ p_i = 1} is a (K−1)-dimensional real vector space:

    perturbation  p ⊕ q = C(p · q)      identity: uniform u      inverse: ⊖p = C(1/p)
    powering      α ⊙ p = C(p^α)

where C is closure (rescale to sum 1). The centred log-ratio

    clr(p) = log p − mean(log p)

is a linear isomorphism Δ°_K → H = {x ∈ ℝ^K : Σ x = 0}, so every statement below can be checked
as ordinary linear algebra on clr coordinates. In this geometry:

    Bayesian update by likelihood ℓ     p ⊕ ℓ
    temperature calibration             (1/T) ⊙ p
    logarithmic opinion pool            ⊕_j (w_j ⊙ p_j)
    orbit debiasing                     (1/|G|) ⊙ ⊕_g g⁻¹·p_g

They are all linear maps. The one non-linear step is `interior`, which moves raw model output
(which can contain exact 0s and 1s) into the open simplex where the geometry is defined.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

Vec = NDArray[np.float64]


def _vec(x: ArrayLike) -> Vec:
    return np.asarray(x, dtype=np.float64)


def closure(x: ArrayLike) -> Vec:
    v = _vec(x)
    return _vec(v / v.sum())


def uniform(k: int) -> Vec:
    return np.full(k, 1.0 / k)


def interior(p: ArrayLike, eta: float) -> Vec:
    """Convex shrink toward uniform: (1−η)p + ηu. Maps the closed simplex into Δ° for η > 0."""
    q = closure(p)
    return _vec((1 - eta) * q + eta / q.size)


def clr(p: ArrayLike) -> Vec:
    lp = np.log(_vec(p))
    return _vec(lp - lp.mean())


def clr_inv(x: ArrayLike) -> Vec:
    v = _vec(x)
    e = np.exp(v - v.max())
    return _vec(e / e.sum())


def perturb(*ps: ArrayLike) -> Vec:
    return clr_inv(np.sum([clr(p) for p in ps], axis=0))


def power(alpha: float, p: ArrayLike) -> Vec:
    return clr_inv(alpha * clr(p))


def inverse(p: ArrayLike) -> Vec:
    return clr_inv(-clr(p))


def combine(ps: Sequence[ArrayLike], weights: Sequence[float]) -> Vec:
    """Aitchison linear combination ⊕_j (w_j ⊙ p_j). A barycenter when Σw = 1."""
    if len(ps) != len(weights) or not ps:
        raise ValueError("need one weight per distribution, at least one distribution")
    return clr_inv(np.sum([w * clr(p) for p, w in zip(ps, weights, strict=True)], axis=0))


def distance(p: ArrayLike, q: ArrayLike) -> float:
    return float(np.linalg.norm(clr(p) - clr(q)))


def relabel(p_by_position: ArrayLike, order: Sequence[int]) -> Vec:
    """Map a vector indexed by presented position back to label order.

    `order[k]` is the label shown at position k, so the result r has r[order[k]] = p[k].
    """
    p = _vec(p_by_position)
    out = np.empty_like(p)
    out[np.asarray(order)] = p
    return out


def present(p_by_label: ArrayLike, order: Sequence[int]) -> Vec:
    """Inverse of `relabel`: the vector as seen in presented position order."""
    return _vec(_vec(p_by_label)[np.asarray(order)])


def cyclic_orbit(n: int) -> list[list[int]]:
    """Z_n acting by rotation. Transitive: every label visits every position exactly once."""
    return [[(k + s) % n for k in range(n)] for s in range(n)]


def reversal_orbit(n: int) -> list[list[int]]:
    """Z_2 acting by reversal: the only non-trivial symmetry that keeps an ordinal scale ordinal."""
    return [list(range(n)), list(range(n))[::-1]]
