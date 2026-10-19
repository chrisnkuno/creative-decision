"""Belief layer: calibration, debiasing and pooling, all linear in Aitchison geometry.

Raw model output p goes through:

    decode     relabel each orbit member from presented positions back to labels
    interior   q_g = (1−η) p_g + η u                       (the one non-linear step)
    debias     d   = (1/|G|) ⊙ ⊕_g q_g                     (orbit barycenter)
    calibrate  c   = β ⊙ d,  β = 1/T                        (tempering)
    pool       π   = ⊕_j (w_j ⊙ c_j)                        (log opinion pool over judges)

Theorems (each has a test in tests/):

  B1  Tempering is a one-parameter group action: β₁ ⊙ (β₂ ⊙ p) = (β₁β₂) ⊙ p, 1 ⊙ p = p, and it
      preserves the ranking of labels for β > 0.
  B2  Calibration NLL is convex in β, because C(q^β) is an exponential family in β with
      sufficient statistic log q. So golden-section search finds the global optimum.
  B3  The log pool is symmetric in judges, unanimous (p,…,p ↦ p) when Σw = 1, and externally
      Bayesian: pool(ℓ ⊕ p_j) = ℓ ⊕ pool(p_j) exactly when Σw = 1.
  B4  Position bias: observed = relabel(present(p) ⊕ b). Under a transitive orbit (cyclic),
      every label meets every position once, so the debiased belief equals p exactly. Under
      reversal, the part of clr(b) that is odd under reversal cancels and the even part remains.
  B5  Hedge weights are invariant to adding a constant to every judge's loss, sum to `total`,
      and are antitone in loss.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from .simplex import Vec, closure, combine, interior, power

KINDS = ("score", "choice", "noul")


def temper(p: Vec, temperature: float) -> Vec:
    return power(1.0 / temperature, p)


def orbit_mean(decoded: Sequence[Vec], eta: float) -> Vec:
    """Barycenter of the decoded orbit members after moving them into the interior."""
    qs = [interior(d, eta) for d in decoded]
    return combine(qs, [1.0 / len(qs)] * len(qs))


def log_pool(dists: Sequence[Vec], weights: Sequence[float]) -> Vec:
    return combine(dists, weights)


def nll(beta: float, dists: Sequence[Vec], labels: Sequence[int], eta: float) -> float:
    total = 0.0
    for d, y in zip(dists, labels, strict=True):
        total -= math.log(float(power(beta, interior(d, eta))[y]))
    return total


def golden_min(f: Callable[[float], float], lo: float, hi: float, tol: float = 1e-6) -> float:
    """Minimise a unimodal function on [lo, hi]."""
    g = (math.sqrt(5) - 1) / 2
    a, b = lo, hi
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = f(c), f(d)
    while b - a > tol:
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = f(d)
    return (a + b) / 2


def _default_temperature() -> dict[str, float]:
    # Priors from the independent jev-1.13 audit (Choice/Score overconfident, Noul under).
    return {"score": 1.3, "choice": 1.3, "noul": 0.9}


def _default_smoothing() -> dict[str, float]:
    return {"score": 0.03, "choice": 0.03, "noul": 0.02}


@dataclass
class Calibrator:
    temperature: dict[str, float] = field(default_factory=_default_temperature)
    smoothing: dict[str, float] = field(default_factory=_default_smoothing)

    def belief(self, kind: str, decoded: Sequence[Vec]) -> Vec:
        """decode → interior → orbit barycenter → temper."""
        return temper(orbit_mean(decoded, self.smoothing[kind]), self.temperature[kind])

    def noul(self, p_yes: float) -> float:
        return float(self.belief("noul", [np.array([1.0 - p_yes, p_yes])])[1])

    def fit(
        self,
        kind: str,
        dists: Sequence[Vec],
        labels: Sequence[int],
        bounds: tuple[float, float] = (0.05, 8.0),
    ) -> float:
        """Maximum-likelihood temperature (B2 guarantees the optimum is global)."""
        eta = self.smoothing[kind]
        beta = golden_min(lambda b: nll(b, dists, labels, eta), *bounds)
        self.temperature[kind] = 1.0 / beta
        return self.temperature[kind]


@dataclass
class JudgeWeights:
    """Exponential weights (Hedge) over judges from cumulative log loss on realised outcomes."""

    names: list[str]
    eta: float = 0.5
    total: float = 1.0  # Σw; >1 extremises agreeing independent judges, =1 is externally Bayesian
    loss: dict[str, float] = field(default_factory=dict)

    def weights(self) -> list[float]:
        losses = np.array([self.loss.get(n, 0.0) for n in self.names])
        w = np.exp(-self.eta * (losses - losses.min()))
        return [float(x) for x in self.total * closure(w)]

    def observe(self, name: str, dist: Vec, outcome: int) -> None:
        self.loss[name] = self.loss.get(name, 0.0) - math.log(max(float(dist[outcome]), 1e-12))
