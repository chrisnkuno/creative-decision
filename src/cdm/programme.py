"""A programme of related decisions that learns per working principle.

`decide` is stateless. A programme keeps the morphology, and with it the evidence about each
working principle, across decisions: every reported test or outcome is credited to the principles
the option uses (`Morphology.record_test`), so the next decision prices every option that shares
them. Evidence can be saved and loaded, so learning carries over between sessions.

    programme = Programme(morphology, Policy(...))
    decision = programme.decide(problem, judgments)
    programme.report("momo", passed={"pay": False})          # a prototype test, per function
    programme.report("street", passed=True, checked=False)   # an outcome observed in use
    programme.save("evidence.json")
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from .decide import Decision, Policy, decide
from .mechanism import Morphology
from .model import Judgment, Problem


@dataclass
class Programme:
    morphology: Morphology
    policy: Policy = field(default_factory=Policy)
    min_confidence: float = 0.5  # below this, a test is not credited to the modal principle
    history: list[dict[str, Any]] = field(default_factory=list)
    _judgments: dict[str, Judgment] = field(default_factory=dict, repr=False)

    def decide(self, problem: Problem, judgments: list[Judgment]) -> Decision:
        """Decide with the programme's current evidence, and remember the judgments."""
        self._judgments = {j.option.id: j for j in judgments}
        return decide(replace(problem, morphology=self.morphology), judgments, self.policy)

    def report(
        self, option_id: str, *, passed: bool | dict[str, bool], checked: bool = True
    ) -> list[tuple[str, str]]:
        """Record what happened to an option from the last decision.

        `checked=True` is a test report, with the policy's sensitivity and specificity;
        `checked=False` is an outcome observed in use, taken as exact. Returns the
        (function, principle) pairs whose evidence changed.
        """
        if option_id not in self._judgments:
            raise KeyError(f"{option_id} was not an option of the last decision")
        j = self._judgments[option_id]
        if not j.mechanism:
            raise ValueError(f"{option_id} has no mechanism classification")
        sens, spec = (
            (self.policy.check_sensitivity, self.policy.check_specificity) if checked else (1, 1)
        )
        done = self.morphology.record_test(
            j.mechanism, passed=passed, sens=sens, spec=spec, min_confidence=self.min_confidence
        )
        self.history.append(
            {"option": option_id, "passed": passed, "checked": checked, "updated": done}
        )
        return done

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps({"morphology": self.morphology.to_dict(), "history": self.history})
        )

    @classmethod
    def load(cls, path: str | Path, policy: Policy | None = None) -> Programme:
        d = json.loads(Path(path).read_text())
        prog = cls(Morphology.from_dict(d["morphology"]), policy or Policy())
        prog.history = d["history"]
        return prog
