"""Contract boundaries: every malformed input is refused loudly rather than mis-computed."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from cdm import Competitor, Dimension, Lottery, Option, Problem, decide, judge_problem, p_best
from cdm.lottery import expected_max
from cdm.simplex import combine

from .judgments import judgment, point, problem

REFUSED: list[tuple[str, Callable[[], Any]]] = [
    ("empty lottery", lambda: Lottery.of([], [])),
    ("ragged lottery", lambda: Lottery.of([1, 2], [1])),
    ("negative mass", lambda: Lottery.of([1, 2], [1, -0.5])),
    ("infinite atom", lambda: Lottery.of([np.inf], [1])),
    ("cvar alpha 0", lambda: Lottery.point(1).cvar(0.0)),
    ("cvar alpha > 1", lambda: Lottery.point(1).cvar(1.5)),
    ("race w/o a sure competitor", lambda: p_best([Competitor(Lottery.point(1), 0.5)])),
    ("emax w/o a sure competitor", lambda: expected_max([Competitor(Lottery.point(1), 0.5)])),
    ("weights ≠ dists", lambda: combine([np.array([0.5, 0.5])], [0.5, 0.5])),
    ("empty pool", lambda: combine([], [])),
    ("level values ≠ levels", lambda: Dimension("d", "q", ("a", "b"), 1, (0.0,)).level_values()),
]


@pytest.mark.parametrize(("name", "call"), REFUSED, ids=[n for n, _ in REFUSED])
def test_refused(name: str, call: Callable[[], Any]) -> None:
    with pytest.raises(ValueError):  # noqa: PT011 - each case is a distinct contract
        call()


def test_two_baselines_are_refused() -> None:
    pr = problem()
    a = judgment("a", pr, point(1), baseline=True)
    b = judgment("b", pr, point(2), baseline=True)
    with pytest.raises(ValueError, match="baseline"):
        decide(pr, [a, b])


def test_zero_total_weight_is_refused() -> None:
    pr = problem(weights=(0.0,))
    with pytest.raises(ValueError, match="weights"):
        decide(pr, [judgment("a", pr, point(1))])


def test_no_judges_is_refused() -> None:
    pr = Problem("g", {}, [Option("a", "a")], [])
    with pytest.raises(ValueError, match="judge"):
        asyncio.run(judge_problem(pr, []))


def test_custom_level_values_are_used() -> None:
    pr = problem(weights=(1.0,))
    convex = replace(pr.dimensions[0], values=(0.0, 0.01, 0.05, 0.2, 1.0))
    pr = replace(pr, dimensions=[convex])
    d = decide(pr, [judgment("a", pr, np.array([0.0, 0, 0, 1.0, 0]))])
    assert d.by_id("a").mean == pytest.approx(0.2)
