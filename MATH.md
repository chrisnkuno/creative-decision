# The mathematics of Creative Decision Making

A creative decision is a map

```
options × judges ──belief──▶ Δ°(levels)^D ──value──▶ Lotteries(ℝ) ──policy──▶ (commit, innovation, portfolio, escalate)
```

Each arrow is a layer with its own algebra. Each claim below names the test that checks it.
Tests are property-based (Hypothesis) unless marked *scenario* or *oracle*.

## 1. Belief layer: Aitchison geometry (`simplex.py`, `belief.py`)

The open simplex Δ° is a real vector space with perturbation `p ⊕ q = C(p·q)` and powering
`α ⊙ p = C(p^α)`. `clr(p) = log p − mean log p` is a linear isomorphism onto `{x : Σx = 0}`.
Every belief operation is linear in this geometry:

| operation | formula | meaning |
|---|---|---|
| Bayes update | `ℓ ⊕ p` | evidence |
| calibration | `(1/T) ⊙ p` | correct over- or under-confidence |
| log opinion pool | `⊕_j w_j ⊙ p_j` | combine judges |
| orbit debiasing | `(1/|G|) ⊙ ⊕_g g⁻¹·p_g` | cancel position bias |

The only non-linear step is `interior(p, η) = (1−η)p + ηu`, which moves raw output (with exact
0/1 values) into Δ°.

| # | claim | test |
|---|---|---|
| S | (Δ°, ⊕, ⊙) satisfies the vector-space axioms; clr is linear and bijective; Aitchison distance is a metric and perturbation is an isometry | `test_simplex_invariants.py` |
| B1 | tempering is a group action (`T₁∘T₂ = T₁T₂`, `T=1` is the identity), preserves ranking, `T→∞` gives uniform, and 0.5 stays 0.5 | `test_b1_*` |
| B2 | calibration NLL is convex in β = 1/T (exponential family in β), so the golden-section fit is global; it recovers a planted T* = 2 from data | `test_b2_*` |
| B3 | the pool is symmetric and unanimous (Σw = 1), and satisfies `pool(ℓ⊕p_j) = (Σw)⊙ℓ ⊕ pool(p_j)`, which is exact external Bayesianity when Σw = 1; one confident judge vetoes a label | `test_b3_*` |
| B4 | with a multiplicative position bias b, the cyclic orbit Z_n removes **any** b exactly (transitive and balanced); the reversal orbit Z₂ removes exactly the reversal-odd part of clr(b); an unbiased judge is a fixed point | `test_b4_*` |
| B5 | Hedge weights `w ∝ exp(−η·loss)` are shift-invariant, sum to `total`, are antitone in loss, and learn which judge is right | `test_b5_*` |

Score questions use Z₂, because reversal is the only re-ordering that keeps an ordinal scale
ordinal. Choice questions use Z_n.

## 2. Value layer: exact finite lotteries (`lottery.py`)

`V_i = Σ_d (w_d/Σw)·X_{i,d} · Π_soft M_{i,c}` is computed by exact convolution. If it would exceed
4096 atoms, it is compacted by splitting each atom's mass between its two neighbouring grid points.

| # | claim | test |
|---|---|---|
| L0 | atoms are sorted and distinct, masses are positive and sum to 1, the CDF is monotone and ends at exactly 1 | `test_lottery_is_a_normalised_measure…` (this caught an ulp overshoot in the CDF) |
| L1 | ⊕ and ⊗ of independent lotteries are commutative and associative; E is linear; variances add | `test_l1_*` |
| L2 | lower-tail CVaR_α is FSD-monotone, translation-equivariant, positively homogeneous and superadditive, bounded by the mean, monotone in α, with CVaR₁ = E | `test_l2_*` |
| L3 | compaction keeps mass and mean exactly and moves CVaR by at most one grid step | `test_l3_*` |
| L4 | `P(best_i) = Σ_x p_i(x) ∫₀¹ Π_{j≠i}(F_j(x⁻) + p_j(x)t) dt` with fair ties; matches brute-force enumeration exactly, sums to 1, is permutation-equivariant, splits equally between identical competitors | `test_l4_*` (*oracle*) |
| L5 | `E[max] = Σ_x x·ΔΠ_j F_j(x)` matches enumeration; EVPI ≥ 0 (Jensen); EVPI = 0 without uncertainty | `test_l5_*` (*oracle*) |
| L6 | EI = E[max(V−B, 0)] is monotone in the convex order (a mean-preserving spread never lowers it), is at least the mean gain, and is 0 under certain failure | `test_l6_*` |

The ∫₀¹ trick: `∫₀¹ t^k dt = 1/(k+1)` averages `1/(1+#ties)` over the tie count, so fair tie
splitting reduces to integrating a polynomial.

## 3. Policy layer (`decide.py`)

```
f_i = 1 − Π_hard(1 − q_ic)          U_i = V_i w.p. 1−f_i, else B         (verify, else fall back)
gate: clear q ≤ δ · verify δ<q<ρ · reject q ≥ ρ
reliable ⇔ ¬reject ∧ CVaR_α(U_i) ≥ CVaR_α(B) − τ
commit = argmax_reliable E[U]          ties → status quo, then id
innovation = argmax_{reliable, novelty ≥ ν, ≠ commit} EI
portfolio = P(best) over {B} ∪ niche elites       escalate ⇔ EVPI > κ
```

In the P(best) race, an option that fails its check is removed rather than tied with the status
quo, because a failed option *is* the status-quo action. The Monte Carlo oracle encodes the same
semantics.

| # | axiom | test |
|---|---|---|
| P1 | the baseline is reliable, so a commit always exists; P(best) and the portfolio each sum to 1; EVPI ≥ 0 | `test_p1_*` |
| P2 | outputs are invariant to option order; renaming changes nothing except an exact E[U] tie between two non-baseline options, which goes to the smaller id | `test_p2_*` |
| P3 | scaling every weight by c > 0 changes nothing | `test_p3_*` |
| P4 | IIA: adding a rejected or FSD-dominated option leaves `commit` unchanged (Hypothesis forced the status-quo tie rule) | `test_p4_*` |
| P5 | novelty only gates the innovation slot; it never changes E, CVaR, P(best), commit, portfolio or EVPI | `test_p5_*` |
| P6 | more hard risk never improves status or lowers f; with perfect checks E[U] moves toward E[B]; with imperfect checks and H ≤ min B, EI never rises (E[U] may move away from E[B]) | `test_p6_*` |
| P7 | with perfect checks, V-dominance and lower f imply higher P(best), and higher E[U] **provided E[V_j] ≥ E[B]** (the counterexample is a regression test) | `test_p7_*` |
| P8 | x ↦ ax + b on level values, H and the null value (and x ↦ ax on τ, κ, k) maps E, CVaR, EI affinely and leaves every decision unchanged; soft penalties shrink toward the utility floor, so they never raise utility | `test_p8_*`, `test_soft_penalties_*` |
| V1 | with imperfect checks (sensitivity s, specificity c, cost k, harm H) every U is normalised, P(best) and the portfolio sum to 1, and U has an atom of mass ≥ f(1−s) at H − k | `test_v1_*` |
| V2 | if H is below the status quo, raising s never lowers E[U] or CVaR | `test_v2_*` |
| V3 | `hard_dependence="robust"` picks the end of the Fréchet interval [max q, min(1, Σq)] with the lower E[U]; that is never above the independent E[U], and gate states do not change | `test_v3_*` |

Ties: values that agree to a relative 1e-9 are equal. Atoms merge, P(best) splits them fairly
and the argmax treats them as tied. Rounding each result to 12 decimals was not enough, because
errors accumulate along different computation paths; P8's test caught this.

P4 is stated for options whose E[U] is strictly below the commit's. An exact tie is settled by the
status-quo rule and then by id, so an equal option with a smaller id could take the commit.

Imperfect checks (`Policy.check_sensitivity`, `check_specificity`, `check_cost`, `harm_value`):

```
U_i = V_i − k  w.p. (1−f)c,   B − k  w.p. fs + (1−f)(1−c),   H − k  w.p. f(1−s)
```

The defaults s = c = 1, k = 0 give back the perfect-check model above. The residual risk f(1−s)
is reported as `OptionStats.residual`. Once misses are priced, the default τ = 0.05 is strict;
see the τ sweep in `scripts/experiments.py`.

Named creative behaviours (*scenario*):

- **Unknown is not risky:** q = 0.5 gives verify, not a veto, and the option can still be the commit.
- **Fallback bounds the downside:** a bold bet's U is essentially never below the status quo.
- **Evidence of harm** is rejected everywhere.
- **Spread wins the innovation slot:** at equal means, the wider distribution takes it.
- **No novelty, no innovation slot.**
- **Near-ties escalate; clear winners don't.**
- **No baseline** falls back to ∅, "do nothing".
- **The portfolio** keeps one elite per niche.

## 4. Functional (`test_functional.py`, `test_contracts.py`)

- **HTTP contract** (`httpx.MockTransport`): request shape, bearer auth, retries on 429/529,
  surfacing 5xx errors, an error when answers are missing, refusal to start without a key.
- **End-to-end bias recovery:** a synthetic judge distorts every answer with a planted position
  bias. The full pipeline (questions → orbit → decode → pool) recovers the true beliefs to 1e-9
  and reaches the same decision as the ground truth.
- **Exact vs simulated:** 400k-sample Monte Carlo of the generative model agrees with the exact
  E, CVaR and P(best).
- **Contracts:** every malformed input raises an error.
- **Live** (`-m live`): real Jev rejects the deceptive option and the output satisfies the
  invariants.

## Assumptions (where the maths stops)

1. **Independence.** Dimensions, options and judges are treated as independent; checks are too,
   unless `hard_dependence="robust"` (worst E[U] over the Fréchet interval) is set.
   Correlated judge error (one model overrating a whole style of idea) is not modelled. Pooling
   different judges is the mitigation.
2. **Interval scale.** Level utilities are policy (`Dimension.values`). Jev's Score levels are
   ordinal and not numerically calibrated.
3. **Position-bias model.** B4 is exact for multiplicative position bias. Bias that interacts with
   content is only reduced, not removed.
4. **Point estimates of q.** For one decision this is enough: the value of a check depends on
   the belief about q only through its mean. The shape of the belief matters for learning across
   similar options, which v0.2.0 does not do.
5. **Calibration priors** (T = 1.3 / 1.3 / 0.9) come from one audit. Refit them with
   `Calibrator.fit` on your own outcomes.
