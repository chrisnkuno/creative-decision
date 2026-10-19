"""Builders for problems and judgments used by the behaviour and functional suites."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from hypothesis import strategies as st

from cdm import Constraint, Dimension, Judgment, Option, Problem
from cdm.simplex import Vec, closure

from .strategies import simplex

LEVELS = ("bad", "poor", "ok", "good", "great")


def problem(
    weights: tuple[float, ...] = (0.5, 0.3, 0.2),
    hard: tuple[str, ...] = ("harm",),
    soft: tuple[str, ...] = ("effort",),
    options: list[Option] | None = None,
) -> Problem:
    return Problem(
        goal="g",
        context={},
        options=options or [],
        dimensions=[Dimension(f"d{i}", "q", LEVELS, w) for i, w in enumerate(weights)],
        constraints=[Constraint(c, c, hard=True) for c in hard]
        + [Constraint(c, c, hard=False) for c in soft],
        niches={"a": "a", "b": "b"},
    )


def point(level: int, k: int = 5) -> Vec:
    p = np.full(k, 1e-12)
    p[level] = 1.0
    return closure(p)


def judgment(
    oid: str,
    pr: Problem,
    dims: list[Vec] | Vec,
    *,
    harm: float = 0.0,
    effort: float = 0.0,
    novelty: float = 0.0,
    niche: str = "a",
    baseline: bool = False,
) -> Judgment:
    ds = dims if isinstance(dims, list) else [dims] * len(pr.dimensions)
    nov = closure(np.array([1 - novelty, 0, 0, 0, novelty]) + 1e-12)
    viol = {c.id: (harm if c.hard else effort) for c in pr.constraints}
    return Judgment(
        Option(oid, oid, baseline=baseline),
        {d.id: v for d, v in zip(pr.dimensions, ds, strict=True)},
        viol,
        nov,
        {"a": 1.0 if niche == "a" else 0.0, "b": 1.0 if niche == "b" else 0.0, "other": 0.0},
    )


def with_option(j: Judgment, oid: str) -> Judgment:
    return replace(j, option=replace(j.option, id=oid, text=oid))


@st.composite
def scenario(draw: st.DrawFn) -> tuple[Problem, list[Judgment]]:
    nd = draw(st.integers(1, 3))
    weights = tuple(draw(st.floats(0.1, 1)) for _ in range(nd))
    pr = problem(weights=weights)
    n = draw(st.integers(1, 5))
    has_base = draw(st.booleans())
    js = [
        judgment(
            f"o{i}",
            pr,
            [draw(simplex(5)) for _ in range(nd)],
            harm=draw(st.sampled_from([0.0, 0.05, 0.3, 0.5, 0.9])),
            effort=draw(st.floats(0, 1)),
            novelty=draw(st.floats(0, 1)),
            niche=draw(st.sampled_from(["a", "b"])),
            baseline=has_base and i == 0,
        )
        for i in range(n)
    ]
    return pr, js
