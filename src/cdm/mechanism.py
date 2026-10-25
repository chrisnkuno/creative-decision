"""Mechanism layer: novelty and feasibility of how an option works.

Creative engineering describes a design by what it must do and how it does it: a list of
functions, and for each function the working principle that performs it (the morphological
chart of Zwicky; the function structure and working principles of Pahl & Beitz). An option is a
point in that chart. This module measures novelty on that description rather than on wording,
and keeps a belief per (function, principle) about whether the principle works, so that a test
of one option teaches something about every option that shares a principle.

Classification. For option i and function f, a judge returns π_{i,f}, a distribution over the
function's catalogue of known principles plus NEW (a principle outside the catalogue).

Novelty, in expectation over the classification:

    N_f(p)  = 1 − min(1, c_{f,p} / m_f)   c_{f,p} = precedents using p for f; N_f(NEW) = 1
    ν_i     = Σ_f w̄_f Σ_p π_{i,f}(p) N_f(p)                      w̄_f = w_f / Σ w
    κ_i     = 1 − Σ_{r ∈ R} Π_f π_{i,f}(r_f)    P(the combination appears in no precedent),
                                                 R = distinct fully recorded precedents

m_f is the number of uses after which a principle counts as familiar. The default m_f = 1 makes
N_f(p) the indicator that no precedent used p for f. m_f = C_f, the number of precedents that
record f (`familiar_after=None`), gives the rarity metric of Shah, Vargas-Hernandez & Smith
(2003), which depends on the size of the record: with three precedents, a principle used once
scores 2/3.

Feasibility. θ_{f,p} is the rate at which principle p performs f across settings; Z_{f,p} ~
Bernoulli(θ_{f,p}) says whether it works in this one. The judge's r_{i,f} is P(the way option i
performs f fails here). In order of precedence, the belief behind P(Z_{f,p} = 1) is

    settled   P = 1 for a principle that counts as proven (below)
    evidence  a Beta(a, b) recorded from tests of p for f (shared by every option that uses p)
    opinion   Beta(s₀(1 − r_{i,f}) + c_{f,p}, s₀ r_{i,f}), s₀ = `prior_strength`: the judge's
              estimate held as s₀ pseudo-observations, plus one success per precedent that used p

A principle counts as proven (`settle`) by default only if its recorded evidence gives
θ ≥ `settle_above` with confidence 1 − `settle_alpha` by Cantelli's inequality, which holds for any
law with the Beta's mean and variance: μ − σ·sqrt((1 − α)/α) ≥ settle_above. Precedent use alone
does not prove a principle, because a settled principle is deployed without a check. "precedent"
restores the v0.3.0 rule (any precedent use, until a test is recorded); "never" checks everything.

With independent classifications and distinct (f, p) per function,

    P(mechanism works) = Π_f Σ_p π_{i,f}(p) θ̄_{i,f,p}           q_i^mech = 1 − P(mechanism works)

which enters the policy as one more hard constraint ("mechanism"): an unproven principle is sent
to a prototype test with a fallback, not vetoed. Interactions between principles (two that work
alone but not together) are not modelled.

Learning. One test report Y about principle p, with sensitivity s = P(reports failure | fails)
and specificity c = P(reports pass | works), has a likelihood affine in θ, so the posterior is a
two-component Beta mixture. `Beta.observe_test` replaces it by the Beta with the same mean and
variance (exact when s = c = 1, and the prior itself when the test is uninformative, s + c = 1).

Theorems (tests/test_mechanism.py):

  M1  ν_i, κ_i ∈ [0, 1]; ν is affine in each π_{i,f}; moving classification mass to a principle
      used by fewer precedents never lowers ν; ν depends on the option only through π, so
      rewording that leaves the classification unchanged leaves ν unchanged.
  M2  Adding a precedent that uses p for f never raises N_f(p) and never lowers N_g(p') for any
      other (g, p').
  M3  q^mech ∈ [0, 1]; it is non-decreasing in each r_{i,f} and non-increasing in each θ̄.
  M4  Beta updates: perfect observations are conjugate; `observe_test` preserves the exact
      posterior mean and variance; an uninformative test leaves the belief unchanged; the
      predictive P(pass) is θ̄c + (1 − θ̄)(1 − s).
  M6  The settled rule is conservative: settled ⇒ P(θ < settle_above) ≤ settle_alpha under the
      recorded belief, for any distribution with its mean and variance.
  M5  (one test, next decision) If the next decision acts on p's option iff E[θ] ≥ t, with gain
      α(E[θ] − t), a perfect test raises its expected value by at most α θ̄(1 − θ̄)/(n + 1),
      attained at t = θ̄; the test can change that decision only if t lies in an interval of
      width 1/(n + 1). Ignorance (n = 2) is 34× more informative than a known coin (n = 100).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

Vec = NDArray[np.float64]
NEW = "new"  # a working principle outside the function's catalogue


@dataclass(frozen=True)
class Function:
    """Something the solution must achieve, with the working principles known to achieve it."""

    id: str
    description: str
    principles: dict[str, str]  # principle id -> description
    weight: float = 1.0

    def labels(self) -> tuple[str, ...]:
        """Classification labels in canonical order: the catalogue, then NEW."""
        if NEW in self.principles:
            raise ValueError(f"{self.id}: '{NEW}' is reserved for principles outside the catalogue")
        return (*self.principles, NEW)


@dataclass(frozen=True)
class Beta:
    a: float
    b: float

    def __post_init__(self) -> None:
        if not (self.a > 0 and self.b > 0):
            raise ValueError("Beta parameters must be positive")

    @property
    def n(self) -> float:
        return self.a + self.b

    @property
    def mean(self) -> float:
        return self.a / self.n

    @property
    def var(self) -> float:
        return self.a * self.b / (self.n**2 * (self.n + 1))

    @classmethod
    def of_mean(cls, mean: float, strength: float) -> Beta:
        m = min(max(mean, 1e-6), 1 - 1e-6)
        return cls(m * strength, (1 - m) * strength)

    def observe(self, *, works: bool) -> Beta:
        return Beta(self.a + 1, self.b) if works else Beta(self.a, self.b + 1)

    def p_pass(self, sens: float, spec: float) -> float:
        """Predictive probability that a test of the principle reports a pass."""
        return self.mean * spec + (1 - self.mean) * (1 - sens)

    def observe_test(self, *, passed: bool, sens: float = 1.0, spec: float = 1.0) -> Beta:
        """Posterior after one test report, moment-matched to a Beta (see module docstring)."""
        like_works = spec if passed else 1 - spec
        like_fails = (1 - sens) if passed else sens
        w1, w0 = self.mean * like_works, (1 - self.mean) * like_fails
        if w1 + w0 <= 0:
            raise ValueError("a test report with probability zero cannot be conditioned on")
        if abs(like_works - like_fails) < 1e-15:
            return self
        if sens == 1.0 and spec == 1.0:
            return self.observe(works=passed)
        w1, w0 = w1 / (w1 + w0), w0 / (w1 + w0)
        hi, lo = self.observe(works=True), self.observe(works=False)
        mean = w1 * hi.mean + w0 * lo.mean
        second = w1 * _second_moment(hi) + w0 * _second_moment(lo)
        var = second - mean**2
        strength = mean * (1 - mean) / var - 1
        return Beta(mean * strength, (1 - mean) * strength)


def _second_moment(b: Beta) -> float:
    return b.a * (b.a + 1) / (b.n * (b.n + 1))


def lower_bound(belief: Beta, alpha: float) -> float:
    """A value θ exceeds with probability at least 1 − α under any law with this mean and
    variance (Cantelli's inequality): μ − σ·sqrt((1 − α)/α)."""
    return belief.mean - math.sqrt(belief.var) * math.sqrt((1 - alpha) / alpha)


def next_decision_value_bound(belief: Beta, alpha: float = 1.0) -> float:
    """M5: the most a perfect test of θ can add to one later decision with slope α."""
    return alpha * belief.mean * (1 - belief.mean) / (belief.n + 1)


def next_decision_value(belief: Beta, threshold: float, alpha: float = 1.0) -> float:
    """Exact value of one perfect test for a later decision that acts iff E[θ] ≥ threshold."""
    m = belief.mean
    hi, lo = belief.observe(works=True).mean, belief.observe(works=False).mean
    after = m * max(alpha * (hi - threshold), 0) + (1 - m) * max(alpha * (lo - threshold), 0)
    return after - max(alpha * (m - threshold), 0)


@dataclass
class Morphology:
    """Functions, the precedents' principles, and beliefs about which principles work."""

    functions: list[Function]
    record: list[dict[str, str]] = field(default_factory=list)  # precedent: function -> principle
    evidence: dict[tuple[str, str], Beta] = field(default_factory=dict)
    prior_strength: float = 2.0  # s₀: pseudo-observations behind a judge's estimate
    familiar_after: int | None = 1  # m_f; None gives Shah's rarity (m_f = C_f)
    # When a principle counts as proven, so that options using it are not checked:
    #   "evidence"   its recorded evidence puts θ above `settle_above` with confidence
    #                1 − `settle_alpha` (Cantelli bound); precedent use alone is not enough
    #   "precedent"  any precedent in this setting used it and no test has been recorded (v0.3.0)
    #   "never"      every principle is checked
    settle: str = "evidence"
    settle_above: float = 0.95
    settle_alpha: float = 0.05

    def __post_init__(self) -> None:
        ids = [f.id for f in self.functions]
        if len(set(ids)) != len(ids):
            raise ValueError("function ids must be unique")
        if any(f.weight < 0 for f in self.functions) or (
            self.functions and sum(f.weight for f in self.functions) <= 0
        ):
            raise ValueError("function weights must be non-negative with a positive sum")
        if self.familiar_after is not None and self.familiar_after < 1:
            raise ValueError("familiar_after must be at least 1, or None for Shah's rarity")
        if self.settle not in {"evidence", "precedent", "never"}:
            raise ValueError("settle is 'evidence', 'precedent' or 'never'")
        if not (0 < self.settle_alpha < 1 and 0 < self.settle_above <= 1):
            raise ValueError("settle_alpha ∈ (0, 1) and settle_above ∈ (0, 1]")
        known = {f.id: f.principles for f in self.functions}
        for r in self.record:
            for fid, p in r.items():
                if fid not in known or p not in known[fid]:
                    raise ValueError(f"precedent uses unknown function/principle {fid}/{p}")

    def function(self, fid: str) -> Function:
        return next(f for f in self.functions if f.id == fid)

    def uses(self, fid: str, p: str) -> int:
        return sum(r.get(fid) == p for r in self.record)

    def recorded(self, fid: str) -> int:
        return sum(fid in r for r in self.record)

    def principle_novelty(self, fid: str, p: str) -> float:
        familiar = self.recorded(fid) if self.familiar_after is None else self.familiar_after
        if p == NEW or familiar == 0:
            return 1.0
        return 1.0 - min(1.0, self.uses(fid, p) / familiar)

    def settled(self, fid: str, p: str) -> bool:
        """Whether p counts as proven for f, so that options using it are not checked."""
        if p == NEW or self.settle == "never":
            return False
        if self.settle == "precedent":
            return (fid, p) not in self.evidence and self.uses(fid, p) > 0
        b = self.evidence.get((fid, p))
        return b is not None and lower_bound(b, self.settle_alpha) >= self.settle_above

    def belief(self, fid: str, p: str, risk: float | None = None) -> Beta | None:
        """The Beta behind θ_{f,p}, or None when settled. `risk` is the judge's r_{i,f}.

        Recorded evidence is shared by every option. Without it, the judge's estimate is held as
        s₀ pseudo-observations, and each precedent that used p for f adds one success.
        """
        if self.settled(fid, p):
            return None
        if p != NEW and (fid, p) in self.evidence:
            return self.evidence[(fid, p)]
        if risk is None:
            raise ValueError(f"{fid}/{p} has no evidence: its belief needs the judge's estimate")
        opinion = Beta.of_mean(1.0 - risk, self.prior_strength)
        return Beta(opinion.a + (0 if p == NEW else self.uses(fid, p)), opinion.b)

    def theta(self, fid: str, p: str, risk: float) -> float:
        b = self.belief(fid, p, risk)
        return 1.0 if b is None else b.mean

    def _check(self, pi: Mapping[str, Vec]) -> None:
        for f in self.functions:
            v = pi[f.id]
            if v.shape != (len(f.labels()),) or np.any(v < 0) or abs(v.sum() - 1) > 1e-9:
                raise ValueError(f"{f.id}: classification must be a distribution over labels")

    def novelty(self, pi: Mapping[str, Vec]) -> float:
        """ν: expected novelty of the option's principles, weighted by function."""
        self._check(pi)
        wsum = sum(f.weight for f in self.functions)
        nov = 0.0
        for f in self.functions:
            per = np.array([self.principle_novelty(f.id, p) for p in f.labels()])
            nov += f.weight / wsum * float(pi[f.id] @ per)
        return nov

    def combination_novelty(self, pi: Mapping[str, Vec]) -> float:
        """κ: probability that the option's combination of principles is in no precedent."""
        self._check(pi)
        if not self.functions:
            return 0.0
        full = {
            tuple(r[f.id] for f in self.functions)
            for r in self.record
            if all(f.id in r for f in self.functions)
        }
        index = {f.id: {p: k for k, p in enumerate(f.labels())} for f in self.functions}
        seen = sum(
            float(
                np.prod(
                    [pi[f.id][index[f.id][p]] for f, p in zip(self.functions, combo, strict=True)]
                )
            )
            for combo in full
        )
        return float(min(1.0, max(0.0, 1.0 - seen)))

    def works(self, pi: Mapping[str, Vec], risk: Mapping[str, float]) -> float:
        """P(every function's principle works in this setting). Missing risks default to 0.5."""
        self._check(pi)
        prob = 1.0
        for f in self.functions:
            r = risk.get(f.id, 0.5)
            theta = np.array([self.theta(f.id, p, r) for p in f.labels()])
            prob *= float(pi[f.id] @ theta)
        return prob

    def learning(self, pi: Mapping[str, Vec], risk: Mapping[str, float]) -> float:
        """Σ_f Σ_p π(p) θ̄(1−θ̄)/(n+1): the M5 bound summed over the option's principles.

        Settled principles teach nothing. A NEW principle is option-specific, so what a test
        teaches about it transfers only to options that later name it in the catalogue.
        """
        self._check(pi)
        total = 0.0
        for f in self.functions:
            r = risk.get(f.id, 0.5)
            for k, p in enumerate(f.labels()):
                b = self.belief(f.id, p, r)
                if b is not None:
                    total += float(pi[f.id][k]) * next_decision_value_bound(b)
        return total

    def update(
        self,
        fid: str,
        p: str,
        *,
        passed: bool,
        sens: float = 1.0,
        spec: float = 1.0,
        prior: Beta | None = None,
    ) -> None:
        """Record one test report about catalogued principle p for function f.

        The first report about a principle starts from `prior` (default Beta(1, 1)), or from
        Beta(1 + uses, 1) for a principle that precedents used.
        """
        if p == NEW:
            raise ValueError("name the principle in the catalogue before recording evidence")
        start = self.evidence.get((fid, p))
        if start is None:
            uses = self.uses(fid, p)
            start = Beta(1.0 + uses, 1.0) if uses else (prior or Beta(1.0, 1.0))
        self.evidence[(fid, p)] = start.observe_test(passed=passed, sens=sens, spec=spec)

    def to_test(
        self, pi: Mapping[str, Vec], *, min_confidence: float = 0.5
    ) -> list[tuple[str, str]]:
        """The (function, principle) pairs a prototype test of this option has to cover.

        One pair per function whose modal principle is not settled; NEW included. A function
        whose modal principle has probability below `min_confidence` is listed as NEW, because a
        test cannot be credited to a principle the judge is unsure about.
        """
        self._check(pi)
        out = []
        for f in self.functions:
            k = int(np.argmax(pi[f.id]))
            p = f.labels()[k] if pi[f.id][k] >= min_confidence else NEW
            if not self.settled(f.id, p):
                out.append((f.id, p))
        return out

    def record_test(
        self,
        pi: Mapping[str, Vec],
        *,
        passed: bool | Mapping[str, bool],
        sens: float = 1.0,
        spec: float = 1.0,
        min_confidence: float = 0.5,
    ) -> list[tuple[str, str]]:
        """Credit a test of an option to the principles it uses; returns the pairs updated.

        `passed` is one result for the whole option, or one per function. A whole-option failure
        is recorded against every unsettled principle of the option, which may blame a principle
        that worked; per-function results avoid that. NEW principles are not recorded.
        """
        results = passed if isinstance(passed, Mapping) else None
        done = []
        for fid, p in self.to_test(pi, min_confidence=min_confidence):
            if p == NEW or (results is not None and fid not in results):
                continue
            ok = bool(passed) if results is None else bool(results[fid])
            self.update(fid, p, passed=ok, sens=sens, spec=spec)
            done.append((fid, p))
        return done

    def to_dict(self) -> dict[str, Any]:
        """A JSON-ready snapshot, so that evidence carries over between sessions."""
        return {
            "functions": [
                {
                    "id": f.id,
                    "description": f.description,
                    "principles": f.principles,
                    "weight": f.weight,
                }
                for f in self.functions
            ],
            "record": self.record,
            "evidence": [[f, p, b.a, b.b] for (f, p), b in sorted(self.evidence.items())],
            "prior_strength": self.prior_strength,
            "familiar_after": self.familiar_after,
            "settle": self.settle,
            "settle_above": self.settle_above,
            "settle_alpha": self.settle_alpha,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Morphology:
        return cls(
            functions=[Function(**f) for f in d["functions"]],
            record=[dict(r) for r in d["record"]],
            evidence={(f, p): Beta(a, b) for f, p, a, b in d["evidence"]},
            prior_strength=d["prior_strength"],
            familiar_after=d["familiar_after"],
            settle=d["settle"],
            settle_above=d["settle_above"],
            settle_alpha=d["settle_alpha"],
        )

    def classification(self, choice: Mapping[str, str]) -> dict[str, Vec]:
        """A certain classification, e.g. for the status quo or a hand-labelled option."""
        out = {}
        for f in self.functions:
            labels = f.labels()
            v = np.zeros(len(labels))
            v[labels.index(choice[f.id])] = 1.0
            out[f.id] = v
        return out


def mechanism_summary(
    morph: Morphology, pi: Mapping[str, Vec], risk: Mapping[str, float]
) -> tuple[float, float, float, float]:
    """(ν, κ, q^mech, learning value) for one option."""
    return (
        morph.novelty(pi),
        morph.combination_novelty(pi),
        1.0 - morph.works(pi, risk),
        morph.learning(pi, risk),
    )


def modal(morph: Morphology, pi: Mapping[str, Vec]) -> dict[str, str]:
    """The most probable principle per function (ties to the earlier label)."""
    return {f.id: f.labels()[int(np.argmax(pi[f.id]))] for f in morph.functions}


def shared_principles(
    morph: Morphology, choices: Sequence[Mapping[str, str]]
) -> dict[tuple[str, str], int]:
    """How many options use each catalogued (function, principle): the reach of one test."""
    out: dict[tuple[str, str], int] = {}
    for ch in choices:
        for f in morph.functions:
            p = ch.get(f.id, NEW)
            if p != NEW:
                out[(f.id, p)] = out.get((f.id, p), 0) + 1
    return out
