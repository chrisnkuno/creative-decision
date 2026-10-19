"""Functional suites: HTTP contract, end-to-end bias recovery, exact vs simulated, live API."""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any

import httpx
import numpy as np
import pytest

from cdm import Calibrator, JevJudge, Judgment, MockJudge, Option, Policy, decide, judge_problem
from cdm.judges import NOVELTY_LEVELS, OTHER, Answers, Questions, build_questions
from cdm.lottery import Lottery
from cdm.simplex import Vec, closure, clr_inv, perturb, present, relabel

from .judgments import problem

# HTTP contract --------------------------------------------------------------------------------
QS: Questions = {"q": {"type": "noul", "instructions": "x?"}}
OK = {"model": "jev-1.13.0", "answers": {"q": {"type": "noul", "noul": 0.7}}, "usage": {}}


def judge_with(handler: Any) -> JevJudge:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="https://t")
    return JevJudge(api_key="secret", client=client, backoff=0.0)


def test_request_shape_and_auth() -> None:
    seen: list[httpx.Request] = []

    def handler(r: httpx.Request) -> httpx.Response:
        seen.append(r)
        return httpx.Response(200, json=OK)

    out = asyncio.run(judge_with(handler).ask({"a": 1}, QS))
    assert out["q"]["noul"] == 0.7
    (r,) = seen
    assert r.method == "POST"
    assert r.url.path == "/v1/systemone"
    assert r.headers["authorization"] == "Bearer secret"
    body = json.loads(r.content)
    assert body == {"state": {"a": 1}, "model": "jev-latest", "questions": QS}


def test_retries_rate_limits_then_succeeds() -> None:
    codes = iter([429, 529, 200])

    def handler(r: httpx.Request) -> httpx.Response:
        c = next(codes)
        return httpx.Response(c, json=OK if c == 200 else {})

    assert asyncio.run(judge_with(handler).ask({}, QS))["q"]["noul"] == 0.7


def test_gives_up_after_retries_and_surfaces_server_errors() -> None:
    def limited(r: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={})

    def broken(r: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={})

    for h in (limited, broken):
        with pytest.raises(httpx.HTTPStatusError):
            asyncio.run(judge_with(h).ask({}, QS))


def test_missing_answers_are_an_error_not_a_silent_gap() -> None:
    def handler(r: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"answers": {}})

    with pytest.raises(ValueError, match="omitted"):
        asyncio.run(judge_with(handler).ask({}, QS))


def test_refuses_to_start_without_a_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        JevJudge()


# End-to-end with a synthetic, position-biased judge -----------------------------------------
class BiasedJudge:
    """Knows the true beliefs; distorts every answer by a multiplicative position bias."""

    name = "biased"

    def __init__(self, truth: dict[tuple[str, str], Any], bias: dict[int, Vec]) -> None:
        self.truth, self.bias = truth, bias

    async def ask(self, state: dict[str, Any], questions: Questions) -> Answers:
        out: Answers = {}
        for qid, q in questions.items():
            key = qid.split("@")[0]
            t = self.truth[(state["option"], key)]
            if q["type"] == "noul":
                out[qid] = {"type": "noul", "noul": t}
                continue
            canon, order = t["labels"], None
            shown = list(q["criteria"])
            order = [canon.index(c) for c in shown]
            seen = perturb(present(t["p"], order), self.bias[len(order)])
            if q["type"] == "score":
                out[qid] = {"probabilities": {str(k): float(v) for k, v in enumerate(seen)}}
            else:
                by_label = relabel(seen, order)
                out[qid] = {
                    "probabilities": {canon[i]: float(by_label[i]) for i in range(len(canon))}
                }
        return out


def odd_bias(n: int, rng: np.random.Generator) -> Vec:
    c = rng.normal(size=n)
    return clr_inv((c - c[::-1]) / 2)


def test_debiasing_recovers_the_truth_end_to_end() -> None:
    rng = np.random.default_rng(1)
    pr = problem(
        options=[Option("base", "base", baseline=True), Option("x", "x"), Option("y", "y")]
    )
    _, orbits = build_questions(pr)
    niche_labels = list(orbits["niche"][0].labels)
    truth: dict[tuple[str, str], Any] = {}
    expected = []
    for o in pr.options:
        dims = {}
        for d in pr.dimensions:
            p = closure(rng.dirichlet(np.ones(5)) + 0.01)
            truth[(o.text, f"dim:{d.id}")] = {"labels": list(d.levels), "p": p}
            dims[d.id] = p
        nov = closure(rng.dirichlet(np.ones(5)) + 0.01)
        nich = closure(rng.dirichlet(np.ones(len(niche_labels))) + 0.01)
        truth[(o.text, "novelty")] = {"labels": list(NOVELTY_LEVELS), "p": nov}
        truth[(o.text, "niche")] = {"labels": niche_labels, "p": nich}
        viol = {c.id: float(rng.uniform(0, 0.5)) for c in pr.constraints}
        for c, q in viol.items():
            truth[(o.text, f"con:{c}")] = q
        expected.append(Judgment(o, dims, viol, nov, dict(zip(niche_labels, nich, strict=True))))

    # Scores get reversal-odd bias (removed exactly by Z_2); choices get arbitrary bias (Z_n).
    bias = {
        5: odd_bias(5, rng),
        len(niche_labels): closure(rng.dirichlet(np.ones(len(niche_labels)))),
    }
    exact = Calibrator(
        temperature={"score": 1.0, "choice": 1.0, "noul": 1.0},
        smoothing={"score": 0.0, "choice": 0.0, "noul": 0.0},
    )
    got = asyncio.run(judge_problem(pr, [BiasedJudge(truth, bias)], exact))
    for g, e in zip(got, expected, strict=True):
        for d in pr.dimensions:
            assert np.allclose(g.dims[d.id], e.dims[d.id], atol=1e-9)
        assert np.allclose(g.novelty, e.novelty, atol=1e-9)
        assert np.allclose(list(g.niche.values()), list(e.niche.values()), atol=1e-9)
        assert g.niche.keys() == e.niche.keys()
        assert OTHER in g.niche
        assert all(abs(g.violations[c] - e.violations[c]) < 1e-9 for c in e.violations)
    assert decide(pr, got).commit == decide(pr, expected).commit


def test_mock_pipeline_is_deterministic() -> None:
    pr = problem(options=[Option("base", "b", baseline=True), Option("x", "x")])
    a = decide(pr, asyncio.run(judge_problem(pr, [MockJudge(), MockJudge("m2")])))
    b = decide(pr, asyncio.run(judge_problem(pr, [MockJudge(), MockJudge("m2")])))
    assert [s.mean for s in a.stats] == [s.mean for s in b.stats]
    assert a.commit == b.commit


# Exact engine vs an independent Monte Carlo of the generative model ------------------------
def simulate(
    pr: Any, js: list[Judgment], pol: Policy, n: int, rng: np.random.Generator
) -> tuple[Vec, Vec]:
    hard = {c.id for c in pr.constraints if c.hard}
    wsum = sum(d.weight for d in pr.dimensions)

    def value(j: Judgment) -> Vec:
        v = np.zeros(n)
        for d in pr.dimensions:
            v += d.weight * rng.choice(np.array(d.level_values()), size=n, p=j.dims[d.id])
        v /= wsum
        for c, q in j.violations.items():
            if c not in hard:
                v *= np.where(rng.random(n) < q, 1 - pol.soft_penalty, 1.0)
        return v

    base = value(js[0])
    U, race = [base], [base]
    for j in js[1:]:
        fails = np.zeros(n, dtype=bool)
        for c, q in j.violations.items():
            if c in hard:
                fails |= rng.random(n) < q
        v = value(j)
        U.append(np.where(fails, base, v))  # realised utility: fall back to the status quo
        race.append(np.where(fails, -np.inf, v))  # in the race a failed option *is* the status quo
    return np.stack(U), np.stack(race)


def test_exact_engine_matches_simulation() -> None:
    rng = np.random.default_rng(3)
    pr = problem()
    from .judgments import judgment  # noqa: PLC0415

    js = [
        judgment("base", pr, [closure(rng.dirichlet(np.ones(5))) for _ in range(3)], baseline=True),
        judgment(
            "x", pr, [closure(rng.dirichlet(np.ones(5))) for _ in range(3)], harm=0.3, effort=0.4
        ),
        judgment(
            "y", pr, [closure(rng.dirichlet(np.ones(5))) for _ in range(3)], harm=0.1, effort=0.1
        ),
    ]
    pol = Policy(reject_above=0.99)
    d = decide(pr, js, pol)
    n = 400_000
    U, R = simulate(pr, js, pol, n, rng)
    for i, s in enumerate(d.stats):
        se = U[i].std() / np.sqrt(n)
        assert abs(U[i].mean() - s.mean) < 5 * se + 1e-9
        k = int(pol.alpha * n)
        assert abs(np.sort(U[i])[:k].mean() - s.cvar) < 0.01
        assert abs(Lottery.of(U[i], np.ones(n)).mean() - s.mean) < 5 * se + 1e-9
    top = R.max(axis=0)
    winners = top == R
    shares = (winners / winners.sum(axis=0)).mean(axis=1)
    assert np.allclose(shares, [s.p_best for s in d.stats], atol=0.01)


# Live -------------------------------------------------------------------------------------
@pytest.mark.live
@pytest.mark.skipif(not os.environ.get("TYPESAFE_API_KEY"), reason="needs TYPESAFE_API_KEY")
def test_live_jev_rejects_deception_and_keeps_contracts() -> None:
    import runpy  # noqa: PLC0415

    pr = runpy.run_path("demo.py", run_name="demo")["problem"]
    js = asyncio.run(judge_problem(pr, [JevJudge()]))
    d = decide(pr, js)
    assert d.by_id("fake").status == "reject"
    assert d.commit != "fake"
    assert abs(sum(s.p_best for s in d.stats) - 1) < 1e-9
    for j in js:
        assert all(abs(p.sum() - 1) < 1e-9 for p in j.dims.values())
