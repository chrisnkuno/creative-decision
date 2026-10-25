"""Does a judge's novelty follow the working mechanism or the wording? (live Jev, ground truth)

Four design problems, each described by two functions with a catalogue of three working
principles and a record of three precedents (Morphology). Each problem has seven mechanisms with
a known classification: two that repeat a precedent, three that use a catalogued principle no
precedent used, and two that use a principle outside the catalogue (NEW). Each mechanism is
written twice with the same mechanistic sentence: plainly, and wrapped in hype ("a radical,
first-of-its-kind breakthrough"). That gives 56 options with true mechanism novelty ν* known by
construction, and a wording factor that leaves ν* unchanged.

Jev answers through cdm's pipeline: the ordinal novelty Score used by v0.2 (judged novelty) and,
per function, a Choice over the catalogue plus NEW (mechanism novelty ν, computed from the
classification by Morphology.novelty). Raw answers are cached.

    TYPESAFE_API_KEY=... uv run python scripts/mechanism_jev.py collect
    uv run python scripts/mechanism_jev.py analyze        # offline, writes paper_figs/
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from cdm import NEW, Calibrator, Dimension, Function, JevJudge, Morphology, Option, Problem
from cdm.decide import apply_mechanism
from cdm.judges import Answers, Questions, judge_problem

RAW = Path("paper_figs/mechanism_raw.json")
OUT = Path("paper_figs/mechanism_jev.json")
SEED = 20261025
HYPE = (
    "A radical, first-of-its-kind breakthrough that reinvents the whole approach: {m} "
    "Nothing like this has been tried before.",
    "Our disruptive, never-before-seen innovation changes the game entirely: {m} "
    "It is a bold leap beyond anything on the market.",
)
LEVELS = ("Not at all", "A little", "Somewhat", "A lot", "Completely")

DOMAINS: list[dict[str, Any]] = [
    {
        "goal": "Sign up the first 100 small shops in Kigali for a stock-keeping app",
        "context": {"team": "3 people", "price": "a monthly fee of about $4"},
        "functions": [
            Function(
                "reach",
                "find shop owners and convince them to sign up",
                {
                    "door": "Staff visit shops in person and demonstrate the app",
                    "ads": "Paid social-media advertising aimed at shop owners",
                    "referral": "Existing shop owners are rewarded for bringing in other owners",
                },
            ),
            Function(
                "pay",
                "collect the monthly fee from shops",
                {
                    "cash": "Staff collect the fee in cash during a visit",
                    "card": "Shops pay by bank card through a card terminal or website",
                    "momo": "Shops pay from their mobile-money wallet on their phone",
                },
            ),
        ],
        "record": [
            {"reach": "door", "pay": "cash"},
            {"reach": "door", "pay": "cash"},
            {"reach": "ads", "pay": "card"},
        ],
        "baseline": "Staff visit shops to demonstrate the app and collect the fee in cash.",
        "mechanisms": [
            (
                "Staff visit each shop, show the app on a tablet and come back monthly to collect "
                "the fee in cash.",
                {"reach": "door", "pay": "cash"},
            ),
            (
                "Run paid Facebook and Instagram ads aimed at shop owners, and charge the fee to "
                "a bank card on a sign-up website.",
                {"reach": "ads", "pay": "card"},
            ),
            (
                "Give every shop owner one free month for each other owner they bring in, and "
                "collect the fee in cash on a monthly visit.",
                {"reach": "referral", "pay": "cash"},
            ),
            (
                "Staff visit each shop to demonstrate the app, and the fee is taken each month "
                "from the owner's mobile-money wallet.",
                {"reach": "door", "pay": "momo"},
            ),
            (
                "Reward owners who bring in other owners with a free month, and charge the fee "
                "to a bank card.",
                {"reach": "referral", "pay": "card"},
            ),
            (
                "Ask the wholesalers who deliver stock to the shops to introduce the app on their "
                "delivery rounds, and collect the fee in cash on a monthly visit.",
                {"reach": NEW, "pay": "cash"},
            ),
            (
                "Staff visit each shop to demonstrate the app, and the fee is added to the shop's "
                "invoice from its wholesale supplier, who passes it on.",
                {"reach": "door", "pay": NEW},
            ),
        ],
    },
    {
        "goal": "Win 500 new customers for a meal-kit delivery service in one city",
        "context": {"team": "4 people", "budget": "about $8k a month"},
        "functions": [
            Function(
                "acquire",
                "get people to place a first order",
                {
                    "ads": "Paid social-media advertising",
                    "codes": "Discount codes for the first box",
                    "influencers": "Paying food influencers to post about the kits",
                },
            ),
            Function(
                "deliver",
                "get the kits to customers",
                {
                    "vans": "The company's own delivery vans and drivers",
                    "courier": "A third-party courier company",
                    "pickup": "Customers collect kits from pickup points such as corner shops",
                },
            ),
        ],
        "record": [
            {"acquire": "ads", "deliver": "vans"},
            {"acquire": "codes", "deliver": "vans"},
            {"acquire": "ads", "deliver": "courier"},
        ],
        "baseline": "Run Instagram ads and deliver kits with our own vans.",
        "mechanisms": [
            (
                "Run Instagram ads for the kits and deliver every order with the company's own "
                "vans.",
                {"acquire": "ads", "deliver": "vans"},
            ),
            (
                "Hand out 50%-off codes for the first box and deliver with the company's own vans.",
                {"acquire": "codes", "deliver": "vans"},
            ),
            (
                "Pay local food influencers to cook a kit on camera, and deliver with the "
                "company's own vans.",
                {"acquire": "influencers", "deliver": "vans"},
            ),
            (
                "Run Instagram ads, and let customers collect their kits from refrigerated "
                "lockers in corner shops instead of home delivery.",
                {"acquire": "ads", "deliver": "pickup"},
            ),
            (
                "Pay food influencers to post about the kits, and deliver through a third-party "
                "courier company.",
                {"acquire": "influencers", "deliver": "courier"},
            ),
            (
                "Give free sample kits to employees of large offices through their HR "
                "departments, and deliver with the company's own vans.",
                {"acquire": NEW, "deliver": "vans"},
            ),
            (
                "Run Instagram ads, and send kits as parcels on the city's scheduled public "
                "buses to stops near customers.",
                {"acquire": "ads", "deliver": NEW},
            ),
        ],
    },
    {
        "goal": "Keep vaccines cold on the way to 40 remote health posts",
        "context": {"terrain": "hills and unpaved roads", "grid power": "unreliable"},
        "functions": [
            Function(
                "transport",
                "move vaccines from the district store to the health posts",
                {
                    "truck": "A pickup truck on a weekly route",
                    "motorbike": "Health workers on motorbikes",
                    "drone": "Small delivery drones flying from the district store",
                },
            ),
            Function(
                "cool",
                "keep the vaccines between 2 and 8 °C",
                {
                    "grid_fridge": "An electric refrigerator powered by the grid",
                    "ice": "Insulated cool boxes with frozen ice packs",
                    "solar_fridge": "A solar-powered refrigerator",
                },
            ),
        ],
        "record": [
            {"transport": "truck", "cool": "grid_fridge"},
            {"transport": "motorbike", "cool": "ice"},
            {"transport": "truck", "cool": "ice"},
        ],
        "baseline": "Carry vaccines by truck in cool boxes with ice packs.",
        "mechanisms": [
            (
                "Carry the vaccines on a weekly truck route, packed in cool boxes with frozen ice "
                "packs.",
                {"transport": "truck", "cool": "ice"},
            ),
            (
                "Health workers carry vaccines on motorbikes in cool boxes with frozen ice packs.",
                {"transport": "motorbike", "cool": "ice"},
            ),
            (
                "Fly the vaccines to each post with small delivery drones, packed with frozen ice "
                "packs.",
                {"transport": "drone", "cool": "ice"},
            ),
            (
                "Carry vaccines by truck and store them at each post in a solar-powered "
                "refrigerator.",
                {"transport": "truck", "cool": "solar_fridge"},
            ),
            (
                "Fly vaccines to the posts by drone and store them in solar-powered refrigerators.",
                {"transport": "drone", "cool": "solar_fridge"},
            ),
            (
                "Hand sealed cool boxes to the drivers of scheduled long-distance buses, who drop "
                "them at the stop nearest each post.",
                {"transport": NEW, "cool": "ice"},
            ),
            (
                "Carry vaccines by truck and store them at each post in a clay-pot cooler that "
                "stays cold as water evaporates from wet sand between two pots.",
                {"transport": "truck", "cool": NEW},
            ),
        ],
    },
    {
        "goal": "Cut the number of bad releases that reach users of a web service",
        "context": {"team": "6 engineers", "releases": "about 10 a week"},
        "functions": [
            Function(
                "detect",
                "catch a bad release",
                {
                    "manual_qa": "A person tests the release by hand before it ships",
                    "tests": "An automated test suite runs before every release",
                    "canary": "The release goes to 1% of users first and an alarm watches "
                    "the error rate",
                },
            ),
            Function(
                "recover",
                "undo a bad release",
                {
                    "manual": "The on-call engineer rolls back by hand",
                    "script": "A script redeploys the previous version",
                    "flags": "New code sits behind feature flags that can be switched off",
                },
            ),
        ],
        "record": [
            {"detect": "manual_qa", "recover": "manual"},
            {"detect": "tests", "recover": "manual"},
            {"detect": "tests", "recover": "script"},
        ],
        "baseline": "Run the automated test suite and have the on-call engineer roll back.",
        "mechanisms": [
            (
                "Run the automated test suite before every release, and have the on-call "
                "engineer roll back by hand when something breaks.",
                {"detect": "tests", "recover": "manual"},
            ),
            (
                "Run the automated test suite and redeploy the previous version with a script "
                "when a release is bad.",
                {"detect": "tests", "recover": "script"},
            ),
            (
                "Ship each release to 1% of users first with an alarm on the error rate, and "
                "have the on-call engineer roll back by hand.",
                {"detect": "canary", "recover": "manual"},
            ),
            (
                "Run the automated test suite, and put all new code behind feature flags that "
                "can be switched off instantly.",
                {"detect": "tests", "recover": "flags"},
            ),
            (
                "Ship each release to 1% of users first with an error-rate alarm, and put new "
                "code behind feature flags that can be switched off.",
                {"detect": "canary", "recover": "flags"},
            ),
            (
                "Replay a copy of yesterday's production traffic against the new release and "
                "compare its responses with the current version's, then have the on-call "
                "engineer roll back by hand.",
                {"detect": NEW, "recover": "manual"},
            ),
            (
                "Run the automated test suite, and keep the old and new versions running side by "
                "side so the load balancer can switch traffic back in one step.",
                {"detect": "tests", "recover": NEW},
            ),
        ],
    },
]
KIND = ("precedent", "precedent", "catalogue", "catalogue", "catalogue", "new", "new")


def morphology(d: dict[str, Any]) -> Morphology:
    return Morphology(list(d["functions"]), [dict(r) for r in d["record"]])


def items(d: dict[str, Any], di: int) -> list[dict[str, Any]]:
    out = [{"id": f"d{di}base", "text": d["baseline"], "baseline": True}]
    for mi, (text, truth) in enumerate(d["mechanisms"]):
        out.append({"id": f"d{di}m{mi}p", "text": text, "truth": truth, "hype": False, "mech": mi})
        wrap = HYPE[mi % len(HYPE)]
        out.append(
            {
                "id": f"d{di}m{mi}h",
                "text": wrap.format(m=text),
                "truth": truth,
                "hype": True,
                "mech": mi,
            }
        )
    return out


def problem_for(d: dict[str, Any], its: list[dict[str, Any]]) -> Problem:
    return Problem(
        goal=d["goal"],
        context=d["context"],
        precedents=[" + ".join(f.principles[r[f.id]] for f in d["functions"]) for r in d["record"]],
        options=[Option(it["id"], it["text"], baseline=bool(it.get("baseline"))) for it in its],
        dimensions=[Dimension("value", "How much would `option` advance `goal`?", LEVELS, 1.0)],
        morphology=morphology(d),
    )


class Recorder:
    def __init__(self, judge: JevJudge, limit: int = 8) -> None:
        self.judge, self.name = judge, judge.name
        self.raw: dict[str, Answers] = {}
        self.sem = asyncio.Semaphore(limit)

    async def ask(self, state: dict[str, Any], questions: Questions) -> Answers:
        async with self.sem:
            a = await self.judge.ask(state, questions)
        self.raw[f"{state['goal']}||{state['option']}"] = a
        return a


class Replay:
    def __init__(self, raw: dict[str, Answers]) -> None:
        self.raw, self.name = raw, "jev-latest"

    async def ask(self, state: dict[str, Any], questions: Questions) -> Answers:  # noqa: ARG002
        return self.raw[f"{state['goal']}||{state['option']}"]


async def collect() -> None:
    rec = Recorder(JevJudge())
    for di, d in enumerate(DOMAINS):
        await judge_problem(problem_for(d, items(d, di)), [rec])
    RAW.write_text(json.dumps({"seed": SEED, "answers": rec.raw}, indent=1))
    print(f"cached {len(rec.raw)} Jev answers -> {RAW}")


# ---------------------------------------------------------------------------------------------
def boot_ci(x: np.ndarray, rng: np.random.Generator, n: int = 10000) -> list[float]:
    idx = rng.integers(0, x.size, size=(n, x.size))
    m = x[idx].mean(axis=1)
    return [float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))]


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def analyze() -> dict[str, Any]:
    raw = json.loads(RAW.read_text())["answers"]
    judge = Replay(raw)
    rng = np.random.default_rng(SEED)
    rows = []
    for di, d in enumerate(DOMAINS):
        its = items(d, di)
        pr = problem_for(d, its)
        m = morphology(d)
        js = asyncio.run(judge_problem(pr, [judge]))
        _, applied = apply_mechanism(pr, js)
        raw_js = asyncio.run(
            judge_problem(pr, [judge], Calibrator(temperature={"score": 1, "choice": 1, "noul": 1}))
        )
        for it, j, ja, jr in zip(its, js, applied, raw_js, strict=True):
            if it.get("baseline"):
                continue
            truth = m.classification(it["truth"])
            judged = float(np.arange(5) @ j.novelty / 4)
            rows.append(
                {
                    "domain": di,
                    "mech": it["mech"],
                    "kind": KIND[it["mech"]],
                    "hype": it["hype"],
                    "nu_true": m.novelty(truth),
                    "kappa_true": m.combination_novelty(truth),
                    "judged": judged,
                    "judged_raw": float(np.arange(5) @ jr.novelty / 4),
                    "nu": float(ja.mechanism_novelty or 0.0),
                    "kappa": ja.combination,
                    "q_mech": ja.violations["mechanism"],
                    "correct": {
                        f.id: f.labels()[int(np.argmax(j.mechanism[f.id]))] == it["truth"][f.id]
                        for f in m.functions
                    },
                    "p_true": {
                        f.id: float(j.mechanism[f.id][f.labels().index(it["truth"][f.id])])
                        for f in m.functions
                    },
                    "is_new": {f.id: it["truth"][f.id] == NEW for f in m.functions},
                }
            )

    def arr(key: str, sel: list[dict[str, Any]] | None = None) -> np.ndarray:
        return np.array([r[key] for r in (rows if sel is None else sel)], dtype=float)

    plain = [r for r in rows if not r["hype"]]
    hype = [r for r in rows if r["hype"]]
    key = lambda r: (r["domain"], r["mech"])  # noqa: E731
    plain, hype = sorted(plain, key=key), sorted(hype, key=key)
    out: dict[str, Any] = {"n_options": len(rows), "n_mechanisms": len(plain)}

    # wording effect: paired over mechanisms
    for measure in ("judged", "judged_raw", "nu", "kappa", "q_mech"):
        diff = arr(measure, hype) - arr(measure, plain)
        out[f"hype_effect_{measure}"] = {"mean": float(diff.mean()), "ci": boot_ci(diff, rng)}

    # tracking the true mechanism novelty
    nu_true = arr("nu_true")
    for measure in ("judged", "nu"):
        x = arr(measure)
        slope, intercept = np.polyfit(nu_true, x, 1)
        out[f"track_{measure}"] = {
            "spearman": spearman(x, nu_true),
            "pearson": float(np.corrcoef(x, nu_true)[0, 1]),
            "slope": float(slope),
            "intercept": float(intercept),
            "mae": float(np.abs(x - nu_true).mean()),
        }

    # by kind of mechanism and wording
    out["by_kind"] = {
        k: {
            w: {
                "judged": float(
                    arr("judged", [r for r in rows if r["kind"] == k and r["hype"] == h]).mean()
                ),
                "nu": float(
                    arr("nu", [r for r in rows if r["kind"] == k and r["hype"] == h]).mean()
                ),
                "nu_true": float(
                    arr("nu_true", [r for r in rows if r["kind"] == k and r["hype"] == h]).mean()
                ),
                "q_mech": float(
                    arr("q_mech", [r for r in rows if r["kind"] == k and r["hype"] == h]).mean()
                ),
            }
            for w, h in (("plain", False), ("hype", True))
        }
        for k in ("precedent", "catalogue", "new")
    }

    # separating novel from known mechanisms, threshold-free (AUROC) and at two gates
    known = [r for r in rows if r["kind"] == "precedent"]
    novel = [r for r in rows if r["kind"] != "precedent"]
    for measure in ("judged", "nu"):
        res: dict[str, float] = {"auroc": auroc(arr(measure, novel), arr(measure, known))}
        for gate in (0.25, 0.5):
            for name, group in (("known", known), ("novel", novel)):
                for w, h in (("plain", False), ("hype", True)):
                    res[f"{name}_{w}_ge_{gate}"] = float(
                        np.mean([r[measure] >= gate for r in group if r["hype"] == h])
                    )
        out[f"gate_{measure}"] = res

    # classification
    cells = [(r, f) for r in rows for f in r["correct"]]
    out["classification"] = {
        "accuracy": float(np.mean([r["correct"][f] for r, f in cells])),
        "accuracy_plain": float(np.mean([r["correct"][f] for r, f in cells if not r["hype"]])),
        "accuracy_hype": float(np.mean([r["correct"][f] for r, f in cells if r["hype"]])),
        "accuracy_new": float(np.mean([r["correct"][f] for r, f in cells if r["is_new"][f]])),
        "accuracy_catalogued": float(
            np.mean([r["correct"][f] for r, f in cells if not r["is_new"][f]])
        ),
        "mean_p_true": float(np.mean([r["p_true"][f] for r, f in cells])),
        "n_cells": len(cells),
    }
    out["rows"] = rows
    OUT.write_text(json.dumps(out, indent=1))
    return out


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "analyze"
    if cmd == "collect":
        asyncio.run(collect())
    res = analyze()
    show = {k: v for k, v in res.items() if k != "rows"}
    print(json.dumps(show, indent=1))


if __name__ == "__main__":
    main()
