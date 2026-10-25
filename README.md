# Creative Decision Making (cdm)

Combine calibrated decision models (TypeSafe Jev and its peers) with decision theory so you can
pick high-quality options **and** keep placing bounded bets on new ones.

**Paper:** [Unknown Is Not Risky: Creative Decision Making with Calibrated Decision Models](paper.pdf)
(`paper.tex`, v0.3.0). It covers the maths with derivations, a synthetic benchmark with held-out
tuning, a benchmark on 237 real Jev judgments with known ground truth, and, new in v0.3, a mechanism
layer from creative engineering: a live Jev test of whether novelty follows the working mechanism
or the wording, and a simulated programme that evaluates learning per principle and the innovation
slot.

```
DIVERGE ─▶ JUDGE ─▶ CALIBRATE ─▶ POOL ─▶ GATE ─▶ VALUE ─▶ SELECT ─▶ LEARN
humans/LLM  Jev etc.  per-type T    log pool  clear/     exact    commit +     refit T,
generate    typed Qs  + smoothing   + Hedge   verify/    lottery  innovation + judge weights
options     per option              weights   reject              portfolio    from outcomes
```

Decision models select; they don't generate. So divergence (writing the options) belongs to people
or a generative LLM, and convergence belongs to typed judgments plus math you control in code.

## The decision-model landscape (Oct 2026)

| Model | Shape | Notes |
|---|---|---|
| TypeSafe **Jev** 1.13 | Noul / Choice / Score + probabilities | Hosted, ~$0.042/M input tokens, 70–500 ms. An audit found Choice/Score overconfident, Noul underconfident, and a first-option bias in Choice. |
| Cloudflare **Clef** / Clef-flash | Jev-compatible API | Stronger at intent routing, weaker at judgment; costs more per token |
| Amazon **Strands Decider** 2B | yes/no or pick + confidence | Open weights; always picks an answer, so you need an escape option |
| **Kev** 0.8B / 4B / 9B | typed decisions | Apache-2.0; the closest self-hosted drop-in |
| **Laya** 421M | encoder, per-option [MASK] | CPU; temperature-fit ECE 0.466 → 0.081 |
| **AnyJev** | wraps any LLM | Reads label logits; L0 rotation debiasing, L1 temperature, L2 linear head |

## The math

Four layers, each with stated theorems. **[MATH.md](MATH.md)** maps every claim to its test.

| layer | structure | key results |
|---|---|---|
| belief (`simplex`, `belief`) | Aitchison vector space on Δ° | calibration, pooling, debiasing and Bayes are all linear; cyclic orbits remove position bias exactly |
| value (`lottery`) | exact finite measures on ℝ | coherent CVaR; closed-form fair-tie P(best) via ∫₀¹Π(F+pt)dt; EI monotone in convex order |
| policy (`decide`) | gates and selection rules | axioms P1–P8: baseline reliability, order, scale and novelty invariance, IIA, risk monotonicity, dominance, utility-origin invariance |
| mechanism (`mechanism`) | functions × working principles (morphological chart) | novelty of *how an option works*, invariant to wording (M1); unproven principles go to a prototype test; per-principle Beta beliefs, so one test informs every option sharing the principle (M4); value of a test for later decisions ≤ θ̄(1−θ̄)/(n+1) (M5) |

The central creative mechanism: P(violation) ≈ 0.5 on an unfamiliar option means *unknown*, so it
becomes **verify** (check before acting, fall back to the status quo if the check fails). The
downside is bounded by construction, which lets the system back new ideas without betting the
business.

## Creative engineering (v0.3)

Give the problem a `Morphology`: the functions a solution must perform, a catalogue of working
principles for each, and the precedents' choices. Jev then classifies each option per function,
and novelty is measured on the mechanism instead of the wording:

```python
from cdm import Function, Morphology, Problem

pay = Function(
    "pay", "collect the monthly fee", {"cash": "cash on a visit", "momo": "mobile money"}
)
reach = Function(
    "reach", "find shop owners", {"door": "visit in person", "referral": "owners refer owners"}
)
problem = Problem(
    ..., morphology=Morphology([reach, pay], record=[{"reach": "door", "pay": "cash"}])
)
```

An option built on an unused principle gets ν > 0 and is sent to a prototype test (the
`mechanism` hard check) rather than vetoed. Record test results with `morphology.update(f, p,
passed=..., sens=..., spec=...)`, and every option that shares the principle is re-priced.
`Policy(learning_weight=w)` lets the innovation slot prefer options whose test would teach most.

## Run

```bash
uv sync
TYPESAFE_API_KEY=... uv run python demo.py   # falls back to MockJudge without a key
./scripts/check.sh                           # ruff format+lint, mypy --strict, 112 tests, ≥90% branch coverage
uv run pytest -m live                        # against the real Jev API
```

Reproduce the paper:

```bash
uv run python scripts/experiments.py              # synthetic benchmark and sweeps (fixed seeds)
uv run python scripts/jev_benchmark.py analyze    # Jev benchmark, offline from cached answers
TYPESAFE_API_KEY=... uv run python scripts/jev_benchmark.py collect   # re-query Jev
TYPESAFE_API_KEY=... uv run python scripts/mechanism_jev.py collect   # mechanism vs wording, live Jev
uv run python scripts/mechanism_jev.py analyze    # offline from cached answers
TYPESAFE_API_KEY=... uv run python scripts/mechanism_disguise.py collect   # jargon / mundane wording
export PYTHONPATH=scripts
uv run python scripts/mechanism_benchmark.py      # sequential learning benchmark (~20 min)
uv run python scripts/mechanism_robustness.py     # where learning stops paying (~15 min)
uv run python scripts/mechanism_jevloop.py        # sequential decisions on Jev's real judgments
uv run python scripts/figures.py                  # every figure
uv run python scripts/verify_figures.py           # checks every number the paper quotes
```

Example (live Jev, 6 options, about 2 s):

```
option    E[U]    sd  CVaR   nov Pbest     EI  rel  status  niche / checks
street    0.39  0.14  0.23  0.59  0.21  0.051  yes  verify  network ['deceptive', 'legal']
momo      0.39  0.14  0.22  0.71  0.23  0.056  yes  verify  channel ['legal']
door      0.36  0.12  0.21  0.21  0.34  0.000  yes  clear   direct
fake      0.36  0.12  0.21  0.74  0.00  0.000   no  reject  direct
reverse   0.35  0.13  0.19  0.70  0.11  0.029  yes  verify  demand ['deceptive', 'legal']
ads       0.29  0.11  0.17  0.11  0.10  0.025  yes  clear   paid

commit: street   innovation: momo   EVPI 0.086 -> escalate ['door', 'momo']
```

`door` (the baseline) gets the P(best) credit for every world where a verify option fails its
check, because in those worlds you end up doing door.

## Known limits

- Checks are assumed perfect unless you set `Policy(check_sensitivity=..., harm_value=...)`. With
  imperfect checks, verify no longer bounds the downside: the residual risk is f(1−s), and the
  default tolerance τ = 0.05 becomes strict (see the paper's benchmark).
- Everything is treated as independent (use `hard_dependence="robust"` for the worst E[U] over
  any dependence between checks). Correlated judge errors (Jev overrating a whole style of
  idea) are not modelled; pool a second, different judge to reduce them.
- The calibration defaults are guesses until fitted on outcomes. Score levels are not numerically
  calibrated, so use them as ordinal positions only.
- Escalation is reported, not executed. Wire `Decision.escalate` to a reasoning model.
- In both benchmarks the gain comes from checking uncertain options; the CVaR reliability gate
  added nothing measurable. On the Jev benchmark the fitted Score temperature (≈2.2–2.5) is far
  from the audit prior (1.3): refit per domain.
