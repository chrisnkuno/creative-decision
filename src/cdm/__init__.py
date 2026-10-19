"""Creative Decision Making.

Three layers, each with stated theorems and tests:

    belief   Aitchison geometry on the simplex: debias, calibrate, pool    (simplex, belief)
    value    exact finite lotteries: E, CVaR, EI, P(best), EVPI            (lottery)
    policy   gates and selection rules with axioms P1–P7                   (decide)
"""

from .belief import Calibrator, JudgeWeights, log_pool, temper
from .decide import Decision, Policy, decide
from .judges import JevJudge, MockJudge, judge_problem
from .lottery import Competitor, Lottery, expected_improvement, expected_max, p_best
from .model import Constraint, Dimension, Judgment, Option, Problem

__all__ = [
    "Calibrator",
    "Competitor",
    "Constraint",
    "Decision",
    "Dimension",
    "JevJudge",
    "JudgeWeights",
    "Judgment",
    "Lottery",
    "MockJudge",
    "Option",
    "Policy",
    "Problem",
    "decide",
    "expected_improvement",
    "expected_max",
    "judge_problem",
    "log_pool",
    "p_best",
    "temper",
]
