"""Where does learning per working principle stop paying? (simulated robustness sweep)

Varies one assumption of scripts/mechanism_benchmark.py at a time, from its default
(status quo 0.5, pilot share 0.05, 24 decisions):

    p_correct   P(the judge's modal classification is the true principle): 0.5 … 1.0
    interact    P(two untested principles fail together although each works); the failure is
                then misattributed to both principles: 0 … 0.5
    untested    untested principles per function, i.e. how many options share each one: 1 … 12

For each setting it reports the paired gain of learning (ν + learning against ν without
learning) and the gain of the innovation pilot with learning against committing alone.

    uv run python scripts/mechanism_robustness.py    # writes paper_figs/mechanism_robustness.json
"""

from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import mechanism_benchmark as mb

OUT = Path("paper_figs/mechanism_robustness.json")
RUNS = 200
DEFAULT = {"p_correct": 0.95, "interact": 0.0, "untested": 3}
SWEEPS: dict[str, list[Any]] = {
    "p_correct": [0.5, 0.7, 0.85, 0.95, 1.0],
    "interact": [0.0, 0.15, 0.3, 0.5],
    "untested": [1, 3, 6, 12],
}
NAMES = ("commit only, learning", "mechanism novelty, no learning", "mechanism novelty + learning")
SPECS = [s for s in mb.SPECS if s.name in NAMES]


def one(job: tuple[str, Any, int]) -> dict[str, Any]:
    knob, value, runs = job
    cfg = {**DEFAULT, knob: value}
    mb.CFG["p_correct"] = cfg["p_correct"]
    mb.CFG["interact"] = cfg["interact"]
    mb.CFG["untested"] = tuple(f"u{k + 1}" for k in range(int(cfg["untested"])))
    res = mb.run(0.5, 0.05, 24, runs=runs, specs=SPECS)
    pol = res["policies"]  # type: ignore[index]
    learn = pol["mechanism novelty + learning"]
    return {
        "knob": knob,
        "value": value,
        "learning_gain": learn["gain_vs_no_learning"],
        "learning_gain_ci": learn["gain_ci"],
        "pilot_vs_commit": learn["gain_vs_commit_only"],
        "pilot_vs_commit_ci": learn["gain_vs_commit_only_ci"],
        "regret": {n: pol[n]["regret"] for n in NAMES},
        "bad_last8": {n: pol[n]["bad_last8"] for n in NAMES},
        "harm": {n: pol[n]["harm"] for n in NAMES},
    }


def main() -> None:
    runs = int(sys.argv[1]) if len(sys.argv) > 1 else RUNS
    jobs = [(k, v, runs) for k, vs in SWEEPS.items() for v in vs if v != DEFAULT[k]]
    jobs.insert(0, ("default", None, runs))
    with ProcessPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(one_or_default, jobs))
    OUT.write_text(json.dumps({"runs": runs, "default": DEFAULT, "results": results}, indent=1))
    for r in results:
        lo, hi = r["learning_gain_ci"]
        plo, phi = r["pilot_vs_commit_ci"]
        print(
            f"{r['knob']:9s} {r['value']!s:5s} learning {r['learning_gain']:+.4f} "
            f"[{lo:+.4f},{hi:+.4f}]  pilot vs commit {r['pilot_vs_commit']:+.4f} "
            f"[{plo:+.4f},{phi:+.4f}]  bad {r['bad_last8']['mechanism novelty + learning']:.2f}"
            f" vs {r['bad_last8']['mechanism novelty, no learning']:.2f}"
        )


def one_or_default(job: tuple[str, Any, int]) -> dict[str, Any]:
    if job[0] == "default":
        return {**one(("p_correct", DEFAULT["p_correct"], job[2])), "knob": "default", "value": "-"}
    return one(job)


if __name__ == "__main__":
    main()
