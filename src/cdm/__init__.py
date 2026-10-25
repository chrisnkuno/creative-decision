"""Creative Decision Making.

Three layers, each with stated theorems and tests:

    belief   Aitchison geometry on the simplex: debias, calibrate, pool    (simplex, belief)
    value    exact finite lotteries: E, CVaR, EI, P(best), EVPI            (lottery)
    policy   gates and selection rules with axioms P1–P8                   (decide)
    mechanism  functions × working principles: novelty of how an option works,
               per-principle feasibility beliefs and what a test teaches   (mechanism)
"""

from .belief import Calibrator, JudgeWeights, log_pool, temper
from .decide import Decision, Policy, apply_mechanism, decide
from .judges import JevJudge, MockJudge, judge_problem
from .lottery import Competitor, Lottery, expected_improvement, expected_max, p_best
from .mechanism import NEW, Beta, Function, Morphology
from .model import Constraint, Dimension, Judgment, Option, Problem

__all__ = [
    "NEW",
    "Beta",
    "Calibrator",
    "Competitor",
    "Constraint",
    "Decision",
    "Dimension",
    "Function",
    "JevJudge",
    "JudgeWeights",
    "Judgment",
    "Lottery",
    "MockJudge",
    "Morphology",
    "Option",
    "Policy",
    "Problem",
    "apply_mechanism",
    "decide",
    "expected_improvement",
    "expected_max",
    "judge_problem",
    "log_pool",
    "p_best",
    "temper",
]
