"""Problem definition: options, value dimensions, constraints, niches."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .mechanism import Morphology


@dataclass(frozen=True)
class Option:
    id: str
    text: str
    baseline: bool = False  # the status quo / incumbent everything is compared against


@dataclass(frozen=True)
class Dimension:
    """A value dimension asked as a Score. Levels go from worst to best."""

    id: str
    question: str
    levels: tuple[str, ...]
    weight: float
    values: tuple[float, ...] | None = None  # utility of each level; default evenly spaced 0..1

    def level_values(self) -> tuple[float, ...]:
        if self.values is not None:
            if len(self.values) != len(self.levels):
                raise ValueError(f"{self.id}: one value per level")
            return self.values
        k = len(self.levels) - 1
        return tuple(i / k for i in range(k + 1))


@dataclass(frozen=True)
class Constraint:
    """Asked as a Noul phrased so that yes == violation."""

    id: str
    violation: str
    hard: bool = False  # hard = irreversible / unacceptable; gates the option out


@dataclass
class Problem:
    goal: str
    context: dict[str, Any]
    options: list[Option]
    dimensions: list[Dimension]
    constraints: list[Constraint] = field(default_factory=list)
    precedents: list[str] = field(default_factory=list)  # what "normal" looks like
    niches: dict[str, str] = field(default_factory=dict)  # approach archetypes for diversity
    morphology: Morphology | None = None  # functions and working principles (mechanism layer)


@dataclass
class Judgment:
    """Calibrated, pooled beliefs about one option."""

    option: Option
    dims: dict[str, NDArray[np.float64]]  # dimension id -> distribution over levels, worst→best
    violations: dict[str, float]  # constraint id -> P(violation)
    novelty: NDArray[np.float64]  # distribution over novelty levels, least→most novel
    niche: dict[str, float] = field(default_factory=dict)  # niche -> probability
    # mechanism layer: function id -> distribution over its principles (catalogue, then NEW),
    # and the judge's P(a NEW principle fails to perform the function)
    mechanism: dict[str, NDArray[np.float64]] = field(default_factory=dict)
    mechanism_risk: dict[str, float] = field(default_factory=dict)
    mechanism_novelty: float | None = None  # ν; replaces the judged novelty when set
    combination: float = 0.0  # κ: P(the combination of principles is in no precedent)
    learning: float = 0.0  # Σ θ̄(1−θ̄)/(n+1) over the option's principles (M5)
