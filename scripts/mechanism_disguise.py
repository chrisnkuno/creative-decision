"""Disguise test: can wording make an old mechanism look new, or a new one look old? (live Jev)

Extends scripts/mechanism_jev.py. The same 28 mechanisms (4 problems × 7) are written in four
wordings that leave the mechanism, and so the true mechanism novelty ν*, unchanged:

    plain     the neutral description
    hype      the plain sentence wrapped in promotional claims of novelty
    jargon    the same mechanism in technical vocabulary that sounds new ("field activation
              squad", "phase-change thermal containment"); every mechanism
    mundane   a new mechanism in deliberately ordinary words ("a fridge that runs on a solar
              panel"); the 20 mechanisms with ν* > 0

Jargon applied to a repeated mechanism is the case where a judged novelty score could admit an
old idea to the innovation slot; mundane wording of a new mechanism is the case where it could
exclude a new one. All 104 options are asked in one collection run.

    TYPESAFE_API_KEY=... uv run python scripts/mechanism_disguise.py collect
    uv run python scripts/mechanism_disguise.py analyze
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from mechanism_jev import DOMAINS, HYPE, KIND, Recorder, Replay, auroc, boot_ci, morphology

from cdm import Dimension, JevJudge, Option, Problem
from cdm.decide import apply_mechanism
from cdm.judges import judge_problem

RAW = Path("paper_figs/disguise_raw.json")
OUT = Path("paper_figs/mechanism_disguise.json")
SEED = 20261026
LEVELS = ("Not at all", "A little", "Somewhat", "A lot", "Completely")
GATES = (0.25, 0.5)

JARGON = [
    [
        "Deploy a field activation squad that runs hands-on product immersion sessions inside "
        "each shop, with a recurring in-person revenue capture cycle settled in physical currency.",
        "Launch a precision-targeted social acquisition funnel aimed at shop owners, converting "
        "through a card-enabled digital checkout flow.",
        "Activate a merchant-to-merchant viral growth loop that rewards owners with a free month "
        "per converted peer, with fees settled in physical currency on a monthly field visit.",
        "Run in-store field activation sessions, with recurring revenue auto-captured from the "
        "owner's mobile wallet rails.",
        "Seed a peer-incentivised network acquisition loop and settle subscriptions through "
        "card-based digital payments.",
        "Leverage upstream supply-chain partners as an embedded distribution channel on their "
        "delivery rounds, with fees settled in cash on a monthly field visit.",
        "Run in-store field demonstrations, and embed the subscription into the B2B supplier "
        "invoicing stack as a pass-through line item.",
    ],
    [
        "Run a performance-marketing acquisition engine on Instagram with an owned last-mile "
        "fulfilment fleet.",
        "Drive first-purchase conversion with a 50% introductory incentive and fulfil through our "
        "owned last-mile fleet.",
        "Activate creator-led content partnerships in which local food creators cook a kit on "
        "camera, fulfilled by our owned last-mile fleet.",
        "Run Instagram acquisition, and replace home delivery with a distributed network of "
        "refrigerated micro-hubs in corner shops for customer self-collection.",
        "Scale through creator-led content partnerships and outsource last-mile fulfilment to a "
        "third-party logistics provider.",
        "Unlock a B2B2C enterprise channel by seeding complimentary sample kits to employee "
        "cohorts via HR benefits teams, fulfilled by our owned fleet.",
        "Run Instagram acquisition, and piggyback last-mile fulfilment on the city's scheduled "
        "public-transit network as a parcel layer.",
    ],
    [
        "Operate a scheduled overland logistics loop with passive thermal containment using "
        "phase-change ice media.",
        "Deploy agile two-wheeled last-mile health couriers carrying passive thermal containment "
        "with frozen phase-change packs.",
        "Deploy an autonomous aerial logistics network with passive thermal payload containment "
        "using frozen packs.",
        "Run the scheduled overland loop and terminate the cold chain at each post in an off-grid "
        "photovoltaic refrigeration unit.",
        "Combine autonomous aerial delivery with off-grid photovoltaic refrigeration at each post.",
        "Integrate cold-chain payloads into existing intercity mass-transit schedules as an "
        "opportunistic logistics layer, with passive ice-pack containment.",
        "Run the overland loop and store vaccines in a passive evaporative-cooling vessel that "
        "exploits latent heat of vaporisation in a sand-filled dual-wall ceramic architecture.",
    ],
    [
        "Gate every release behind a CI-driven automated verification pipeline, with "
        "human-in-the-loop manual reversion by the on-call engineer.",
        "Gate releases behind a CI verification pipeline, with automated scripted reversion to "
        "the last known-good artifact.",
        "Use progressive-exposure deployment to a 1% cohort with SLO-based error-rate alerting, "
        "and human-in-the-loop reversion.",
        "Run the CI verification pipeline and wrap all new code paths in runtime kill switches "
        "that can be toggled off instantly.",
        "Combine progressive-exposure rollouts with SLO alerting and runtime kill switches on "
        "every new code path.",
        "Mirror production traffic into a shadow environment and perform differential response "
        "analysis between versions, with human-in-the-loop reversion.",
        "Run the CI pipeline and maintain parallel blue and green production environments so "
        "that the load balancer can cut traffic back atomically.",
    ],
]

MUNDANE: list[dict[int, str]] = [
    {
        2: "Tell each shop owner they get a month free for every other owner they bring along, "
        "and pick up the fee in cash when we visit each month.",
        3: "Visit the shops to show the app, and the owner pays each month from the money on "
        "their phone.",
        4: "Give owners a free month for each friend they sign up, and take the fee from a bank "
        "card.",
        5: "Ask the people who drop off stock at the shops to mention the app when they deliver, "
        "and pick up the fee in cash each month.",
        6: "Visit the shops to show the app, and add the fee to the bill the shop already gets "
        "from its wholesaler, who hands the money on.",
    },
    {
        2: "Pay some local cooks who post online to make a kit on video, and deliver with our own "
        "vans.",
        3: "Advertise on Instagram, and let people pick up their kit from a fridge at the corner "
        "shop instead of getting it delivered.",
        4: "Pay food bloggers to post about the kits, and have a courier company drop them off.",
        5: "Give free sample kits to people at big offices through their HR staff, and deliver "
        "with our own vans.",
        6: "Advertise on Instagram, and send the kits as parcels on the city buses to the stop "
        "nearest each customer.",
    },
    {
        2: "Send the vaccines to each post with small flying delivery machines, packed with ice "
        "packs.",
        3: "Bring the vaccines by truck and keep them at each post in a fridge that runs on a "
        "solar panel.",
        4: "Fly the vaccines in by small drone and keep them in a fridge powered by a solar panel.",
        5: "Give the sealed cool boxes to the drivers of the regular long-distance buses, who "
        "leave them at the stop closest to each post.",
        6: "Bring the vaccines by truck and keep them in a clay pot set inside a bigger pot with "
        "wet sand between, which stays cool as the water dries off.",
    },
    {
        2: "Give each new version to 1 user in 100 first and sound an alarm if errors go up; the "
        "on-call engineer undoes it by hand.",
        3: "Run the automatic tests, and put every new piece of code behind an on/off switch we "
        "can flip off straight away.",
        4: "Release to 1 user in 100 first with an alarm on errors, and put new code behind "
        "on/off switches.",
        5: "Play yesterday's real requests against the new version and compare its answers with "
        "the old one's; the on-call engineer undoes it by hand if needed.",
        6: "Run the automatic tests, and keep the old and new versions both running so traffic "
        "can be switched back to the old one in one step.",
    },
]


def items(di: int) -> list[dict[str, Any]]:
    d = DOMAINS[di]
    out: list[dict[str, Any]] = [{"id": f"d{di}base", "text": d["baseline"], "baseline": True}]
    for mi, (text, truth) in enumerate(d["mechanisms"]):
        variants = {
            "plain": text,
            "hype": HYPE[mi % len(HYPE)].format(m=text),
            "jargon": JARGON[di][mi],
        }
        if mi in MUNDANE[di]:
            variants["mundane"] = MUNDANE[di][mi]
        for w, t in variants.items():
            out.append(
                {"id": f"d{di}m{mi}{w}", "text": t, "truth": truth, "wording": w, "mech": mi}
            )
    return out


def problem_for(di: int, its: list[dict[str, Any]]) -> Problem:
    d = DOMAINS[di]
    return Problem(
        goal=d["goal"],
        context=d["context"],
        precedents=[" + ".join(f.principles[r[f.id]] for f in d["functions"]) for r in d["record"]],
        options=[Option(it["id"], it["text"], baseline=bool(it.get("baseline"))) for it in its],
        dimensions=[Dimension("value", "How much would `option` advance `goal`?", LEVELS, 1.0)],
        morphology=morphology(d),
    )


async def collect() -> None:
    rec = Recorder(JevJudge())
    for di in range(len(DOMAINS)):
        await judge_problem(problem_for(di, items(di)), [rec])
    RAW.write_text(json.dumps({"seed": SEED, "answers": rec.raw}, indent=1))
    print(f"cached {len(rec.raw)} Jev answers -> {RAW}")


def rows() -> list[dict[str, Any]]:
    raw = json.loads(RAW.read_text())["answers"]
    judge = Replay(raw)
    out = []
    for di in range(len(DOMAINS)):
        its = items(di)
        pr = problem_for(di, its)
        m = morphology(DOMAINS[di])
        js = asyncio.run(judge_problem(pr, [judge]))
        _, applied = apply_mechanism(pr, js)
        for it, j, ja in zip(its, js, applied, strict=True):
            if it.get("baseline"):
                continue
            out.append(
                {
                    "domain": di,
                    "mech": it["mech"],
                    "kind": KIND[it["mech"]],
                    "wording": it["wording"],
                    "nu_true": m.novelty(m.classification(it["truth"])),
                    "judged": float(np.arange(5) @ j.novelty / 4),
                    "nu": float(ja.mechanism_novelty or 0.0),
                    "q_mech": ja.violations["mechanism"],
                    "correct": [
                        f.labels()[int(np.argmax(j.mechanism[f.id]))] == it["truth"][f.id]
                        for f in m.functions
                    ],
                    "pi": {f.id: [float(x) for x in j.mechanism[f.id]] for f in m.functions},
                    "risk": dict(j.mechanism_risk),
                    "value": [float(x) for x in j.dims["value"]],
                }
            )
    return out


def analyze() -> dict[str, Any]:
    rs = rows()
    rng = np.random.default_rng(SEED)
    out: dict[str, Any] = {"n_options": len(rs)}
    by = {(r["domain"], r["mech"], r["wording"]): r for r in rs}

    def paired(wording: str, kinds: tuple[str, ...], measure: str) -> dict[str, Any]:
        keys = [
            (d, m)
            for (d, m, w) in by
            if w == wording and KIND[m] in kinds and (d, m, "plain") in by
        ]
        diff = np.array(
            [by[(d, m, wording)][measure] - by[(d, m, "plain")][measure] for d, m in keys]
        )
        return {"n": len(keys), "mean": float(diff.mean()), "ci": boot_ci(diff, rng)}

    repeated, novel = ("precedent",), ("catalogue", "new")
    for measure in ("judged", "nu", "q_mech"):
        out[f"{measure}_jargon_repeated"] = paired("jargon", repeated, measure)
        out[f"{measure}_jargon_novel"] = paired("jargon", novel, measure)
        out[f"{measure}_mundane_novel"] = paired("mundane", novel, measure)
        out[f"{measure}_hype_all"] = paired("hype", (*repeated, *novel), measure)

    # level by kind and wording
    out["levels"] = {
        f"{k}/{w}": {
            m: float(np.mean([r[m] for r in rs if r["kind"] in ks and r["wording"] == w]))
            for m in ("judged", "nu", "nu_true", "q_mech")
        }
        for k, ks in (("repeated", repeated), ("novel", novel))
        for w in ("plain", "hype", "jargon", "mundane")
        if any(r["kind"] in ks and r["wording"] == w for r in rs)
    }

    # innovation-slot eligibility: false admissions of repeated mechanisms, missed novel ones
    for measure in ("judged", "nu"):
        res: dict[str, Any] = {}
        for g in GATES:
            for w in ("plain", "hype", "jargon", "mundane"):
                rep = [r[measure] >= g for r in rs if r["kind"] in repeated and r["wording"] == w]
                nov = [r[measure] >= g for r in rs if r["kind"] in novel and r["wording"] == w]
                res[f"{w}@{g}"] = {
                    "repeated_admitted": float(np.mean(rep)) if rep else None,
                    "novel_admitted": float(np.mean(nov)) if nov else None,
                }
            elig = [r for r in rs if r[measure] >= g]
            res[f"precision@{g}"] = float(np.mean([r["kind"] in novel for r in elig]))
            res[f"recall@{g}"] = float(np.mean([r[measure] >= g for r in rs if r["kind"] in novel]))
        pos = np.array([r[measure] for r in rs if r["kind"] in novel])
        neg = np.array([r[measure] for r in rs if r["kind"] in repeated])
        res["auroc"] = auroc(pos, neg)
        nu_true = np.array([r["nu_true"] for r in rs])
        res["mae"] = float(np.abs(np.array([r[measure] for r in rs]) - nu_true).mean())
        out[f"gate_{measure}"] = res

    out["accuracy"] = {
        w: float(np.mean([c for r in rs if r["wording"] == w for c in r["correct"]]))
        for w in ("plain", "hype", "jargon", "mundane")
    }
    out["rows"] = rs
    OUT.write_text(json.dumps(out, indent=1))
    return out


def main() -> None:
    if (sys.argv[1] if len(sys.argv) > 1 else "analyze") == "collect":
        asyncio.run(collect())
    res = analyze()
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}, indent=1))


if __name__ == "__main__":
    main()
