"""Judges: decision models mapping (state, typed questions) to distributions.

Anything with the System One JSON shape fits `Judge`: TypeSafe Jev, Cloudflare Clef
(Jev-compatible), self-hosted Kev or Strands Decider behind the same shape, or an LLM wrapped
AnyJev-style.

Each judgment is asked over a group orbit of presentations, and the answers are pulled back to
label order and combined at the orbit barycenter (belief.B4):

    Score  (ordinal)    reversal orbit Z_2    removes direction bias (odd part of position bias)
    Choice (unordered)  cyclic orbit  Z_n     removes multiplicative position bias exactly
"""

from __future__ import annotations

import asyncio
import hashlib
import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import httpx
import numpy as np

from .belief import Calibrator, JudgeWeights, log_pool
from .mechanism import NEW
from .model import Judgment, Option, Problem
from .simplex import Vec, cyclic_orbit, relabel, reversal_orbit

Questions = dict[str, dict[str, Any]]
Answers = dict[str, dict[str, Any]]

NOVELTY_LEVELS = (
    "Essentially the same as one of the `precedents`",
    "A minor variation on a precedent",
    "Recombines precedents in a way none of them does alone",
    "A different mechanism from every precedent, though known elsewhere",
    "A mechanism with no clear precedent in this domain",
)
OTHER = "other"


class Judge(Protocol):
    name: str

    async def ask(self, state: dict[str, Any], questions: Questions) -> Answers: ...


class JevJudge:
    def __init__(
        self,
        model: str = "jev-latest",
        *,
        api_key: str | None = None,
        client: httpx.AsyncClient | None = None,
        base_url: str = "https://api.typesafe.ai",
        retries: int = 4,
        backoff: float = 0.5,
    ) -> None:
        self.name = model
        self.model = model
        self.retries, self.backoff = retries, backoff
        key = api_key or os.environ.get("TYPESAFE_API_KEY")
        if client is None and not key:
            raise RuntimeError("TYPESAFE_API_KEY is not set")
        self.client = client or httpx.AsyncClient(base_url=base_url, timeout=30)
        self.headers = {"Authorization": f"Bearer {key}"} if key else {}

    async def ask(self, state: dict[str, Any], questions: Questions) -> Answers:
        body = {"state": state, "model": self.model, "questions": questions}
        for attempt in range(self.retries):
            r = await self.client.post("/v1/systemone", json=body, headers=self.headers)
            if r.status_code in (429, 529) and attempt < self.retries - 1:
                await asyncio.sleep(self.backoff * 2**attempt)
                continue
            r.raise_for_status()
            answers: Answers = r.json()["answers"]
            missing = set(questions) - set(answers)
            if missing:
                raise ValueError(f"judge omitted answers: {sorted(missing)}")
            return answers
        raise AssertionError("unreachable")


class MockJudge:
    """Deterministic offline stand-in so the pipeline runs without an API key."""

    def __init__(self, name: str = "mock") -> None:
        self.name = name

    async def ask(self, state: dict[str, Any], questions: Questions) -> Answers:
        out: Answers = {}
        for qid, q in questions.items():
            seed = hashlib.sha256(f"{self.name}{state['option']}{qid}".encode()).digest()
            rng = np.random.default_rng(int.from_bytes(seed[:8]))
            if q["type"] == "noul":
                out[qid] = {"type": "noul", "noul": float(rng.beta(1, 4))}
                continue
            crit = q["criteria"]
            keys = list(crit) if q["type"] == "choice" else [str(i) for i in range(len(crit))]
            p = rng.dirichlet(np.full(len(keys), 0.6))
            out[qid] = {
                "type": q["type"],
                "probabilities": dict(zip(keys, map(float, p), strict=True)),
            }
        return out


@dataclass(frozen=True)
class Presentation:
    """One orbit member: question id plus `order[k]` = label shown at position k."""

    qid: str
    kind: str  # score | choice
    labels: tuple[str, ...]
    order: tuple[int, ...]


def _score_orbit(key: str, q: str, levels: Sequence[str], qs: Questions) -> list[Presentation]:
    out = []
    for g, order in enumerate(reversal_orbit(len(levels))):
        qid = f"{key}@{g}"
        qs[qid] = {"type": "score", "instructions": q, "criteria": [levels[i] for i in order]}
        out.append(Presentation(qid, "score", tuple(levels), tuple(order)))
    return out


def _choice_orbit(key: str, q: str, crit: dict[str, str], qs: Questions) -> list[Presentation]:
    labels = list(crit)
    out = []
    for g, order in enumerate(cyclic_orbit(len(labels))):
        qid = f"{key}@{g}"
        qs[qid] = {
            "type": "choice",
            "instructions": q,
            "criteria": {labels[i]: crit[labels[i]] for i in order},
        }
        out.append(Presentation(qid, "choice", tuple(labels), tuple(order)))
    return out


def build_questions(problem: Problem) -> tuple[Questions, dict[str, list[Presentation]]]:
    """One request per option. All questions are independent, so one parallel call."""
    qs: Questions = {}
    orbits: dict[str, list[Presentation]] = {}
    for d in problem.dimensions:
        q = f"{d.question} Judge only `option`, in the setting described by `goal` and `context`."
        orbits[f"dim:{d.id}"] = _score_orbit(f"dim:{d.id}", q, d.levels, qs)
    for c in problem.constraints:
        qs[f"con:{c.id}"] = {
            "type": "noul",
            "instructions": f"Would carrying out `option` {c.violation}?",
            "criteria": {"true": f"Yes, it would {c.violation}", "false": "No, it would not"},
        }
    nov = "How different is the approach in `option` from every approach listed in `precedents`?"
    orbits["novelty"] = _score_orbit("novelty", nov, NOVELTY_LEVELS, qs)
    if problem.niches:
        crit = {**problem.niches, OTHER: "None of the other approach types fits"}
        ask = "Which type of approach does `option` mainly take?"
        orbits["niche"] = _choice_orbit("niche", ask, crit, qs)
    for f in problem.morphology.functions if problem.morphology else []:
        crit = {**f.principles, NEW: "A working principle not described in any other answer"}
        ask = (
            f"Which working principle does `option` rely on to {f.description}? Judge how it "
            "would actually work, not how it is described or what it is called."
        )
        orbits[f"mech:{f.id}"] = _choice_orbit(f"mech:{f.id}", ask, crit, qs)
        qs[f"risk:{f.id}"] = {
            "type": "noul",
            "instructions": (
                f"In the setting described by `goal` and `context`, would the way `option` "
                f"tries to {f.description} fail to work in practice?"
            ),
            "criteria": {"true": "Yes, that part would not work", "false": "No, it would work"},
        }
    return qs, orbits


def decode(answer: dict[str, Any], pres: Presentation) -> Vec:
    """Pull one presented answer back to canonical label order."""
    probs = answer["probabilities"]
    if pres.kind == "score":  # keyed by presented position
        by_pos = np.array([float(probs[str(k)]) for k in range(len(pres.order))])
        return relabel(by_pos, pres.order)
    return np.array([float(probs[label]) for label in pres.labels])  # keyed by label


def state_for(problem: Problem, o: Option) -> dict[str, Any]:
    return {
        "goal": problem.goal,
        "context": problem.context,
        "precedents": problem.precedents,
        "option": o.text,
    }


def fuse(
    problem: Problem,
    *,
    option: Option,
    answers: Sequence[Answers],
    orbits: dict[str, list[Presentation]],
    cal: Calibrator,
    weights: Sequence[float],
) -> Judgment:
    """Per judge: decode → interior → orbit barycenter → temper; then log-pool across judges."""

    def belief(key: str) -> Vec:
        per_judge = [
            cal.belief(orbits[key][0].kind, [decode(a[p.qid], p) for p in orbits[key]])
            for a in answers
        ]
        return log_pool(per_judge, weights)

    dims = {d.id: belief(f"dim:{d.id}") for d in problem.dimensions}
    violations = {}
    for c in problem.constraints:
        ps = [cal.noul(float(a[f"con:{c.id}"]["noul"])) for a in answers]
        violations[c.id] = float(log_pool([np.array([1 - p, p]) for p in ps], weights)[1])
    niche: dict[str, float] = {}
    if problem.niches:
        niche = dict(zip(orbits["niche"][0].labels, map(float, belief("niche")), strict=True))
    mechanism: dict[str, Vec] = {}
    risk: dict[str, float] = {}
    for f in problem.morphology.functions if problem.morphology else []:
        mechanism[f.id] = belief(f"mech:{f.id}")
        ps = [cal.noul(float(a[f"risk:{f.id}"]["noul"])) for a in answers]
        risk[f.id] = float(log_pool([np.array([1 - p, p]) for p in ps], weights)[1])
    return Judgment(option, dims, violations, belief("novelty"), niche, mechanism, risk)


async def judge_problem(
    problem: Problem,
    judges: Sequence[Judge],
    calibrator: Calibrator | None = None,
    weights: JudgeWeights | None = None,
) -> list[Judgment]:
    if not judges:
        raise ValueError("need at least one judge")
    cal = calibrator or Calibrator()
    w = (weights or JudgeWeights([j.name for j in judges])).weights()
    questions, orbits = build_questions(problem)
    raw = await asyncio.gather(
        *(j.ask(state_for(problem, o), questions) for o in problem.options for j in judges)
    )
    n = len(judges)
    return [
        fuse(problem, option=o, answers=raw[k * n : (k + 1) * n], orbits=orbits, cal=cal, weights=w)
        for k, o in enumerate(problem.options)
    ]
