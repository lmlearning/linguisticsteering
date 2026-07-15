# Certified Segment-Level Shapley Attribution for LLM Prompts

**A follow-up research proposal to the linguistic-steering project**

*Status: proposal — July 2026*

---

## 1. Motivation and relation to the current project

The current project (`estimate_importance.py` and companions) measures how individual
adjectives injected into an MMLU/ARC prompt instruction ("Your answer should be
{adjectives}…") steer answer accuracy, using a KernelSHAP-style estimator over ~200
sampled coalitions per question, across several API models.

Three limitations motivate the follow-up:

1. **Granularity.** Players are single adjectives. The practically important question is
   about *segments*: whole instructions in a system prompt, few-shot examples, persona
   blocks, retrieved documents, safety clauses. "How much does instruction #3 contribute
   to end-task quality, accounting for its interactions with the other instructions?" is
   exactly a group-level Shapley question, and answering it enables principled prompt
   compression, instruction debugging, and redundancy/conflict detection.
2. **Cost.** ~200 LLM calls per question per model is the dominant expense, and the
   number 200 is arbitrary: there is no stopping rule, no error bar on any Shapley value,
   and no way to know if 50 calls would have sufficed or 500 were needed.
3. **Statistical soundness.** The coalition sampler draws a uniform size then a uniform
   subset, which is *not* the Shapley-kernel distribution the weighted regression
   assumes; the value function is a single binary correctness draw (maximally noisy); and
   no guarantee connects the regression output to the true Shapley values.

The follow-up keeps the core idea — Shapley values of prompt components with respect to a
task-quality utility — and rebuilds it around two goals the current state of the art does
not deliver together: **minimal query cost** and **provable error bounds**.

## 2. What the state of the art does and does not cover (survey summary)

We surveyed the 2021–2026 literature in two directions. Full citations in §7.

**Applied LLM/prompt attribution (no guarantees).**
TokenSHAP (2024) does Monte-Carlo Shapley over prompt tokens with a similarity utility;
TextGenSHAP (2024) accelerates document/token Shapley with white-box tricks (attention
masking, speculative decoding) and heuristic hierarchy; ContextCite (NeurIPS 2024) fits a
sparse *linear* surrogate to context-ablation data (~32–256 calls, but weights are not
Shapley values and linearity ignores interactions); AttriBoT (ICLR 2025) makes
leave-one-out context attribution >300× faster via KV-cache reuse and proxy models (LOO
misses redundancy by construction); TracLLM (USENIX Sec 2025) searches long contexts
heuristically; and **ProCut (2025)** — the closest work to our target — segments
industrial prompt templates and prunes low-attribution segments (SHAP/LOO/LASSO variants),
reporting 78% token reduction *with zero statistical guarantees*. The single
LLM-attribution work with formal bounds, Cluster Shapley (2025/26), values *retrieved
documents in summarization marketplaces* under clustering-similarity assumptions — a
different game with different players.

**Estimation theory (guarantees, but no LLM/noise/cost model).**
Permutation sampling with Hoeffding bounds needs O(n r² log(n/δ)/ε²) evaluations (Castro
2009; Maleki 2013); stratified refinements with Neyman allocation (Castro 2017) and
empirical-Bernstein adaptive allocation (Burgess & Chapman, IJCAI 2021) give
variance-adaptive bounds; complementary contributions (Zhang et al., SIGMOD 2023) and
(Stratified) SVARM (AAAI 2024) let **one coalition evaluation update all n players**,
achieving per-player MSE = O(σ² log n / T); Leverage SHAP (ICLR 2025) proves the first
non-asymptotic guarantee in the KernelSHAP family with **O(n log(n/δ) + n/(εδ))**
evaluations; Faith-Shap (JMLR 2023), SHAP-IQ/SVARM-IQ, k-additive surrogates and
FourierSHAP exploit low interaction order or sparsity; SHAP@k (AAAI 2024) casts top-k
Shapley identification as a PAC bandit problem with gap-dependent complexity; Data
Banzhaf (AISTATS 2023) is the lone analysis of value estimation under a *noisy* utility
oracle — and it abandons the Shapley value for the Banzhaf value.

**Confirmed open gaps this proposal claims** (none of the below exists as of mid-2026):

- **G1.** (ε,δ)-certified Shapley attributions for the *segments/instructions of an LLM
  prompt*, treating the LLM (+ judge / eval set) as a stochastic oracle.
- **G2.** Joint analysis of the **two-level sampling problem** — sample more coalitions
  vs. replicate evaluations within a coalition — with an optimal budget split and joint
  concentration bound.
- **G3.** Owen / hierarchical group-Shapley estimation with non-asymptotic query
  complexity guarantees (Leverage SHAP has no Owen analogue; Partition-SHAP has no
  statistics at all).
- **G4.** **Cache-aware Shapley**: sampling theory where query cost is non-uniform and
  depends on prefix overlap with previously evaluated coalitions (KV-cache economics).
  Flagged as open in our survey; no theory exists.
- **G5.** Distribution-free *post-hoc validation certificates* for surrogate-based
  (sparse / k-additive) Shapley estimates, replacing unverifiable sparsity assumptions.
- **G6.** Query-complexity **lower bounds** for group-level Shapley estimation with a
  noisy oracle (the only tight semivalue lower bound is Data Banzhaf's, for Banzhaf).

## 3. Formal setup (shared by all three approaches)

Let a prompt template decompose into segments N = {1,…,n}, n ≈ 5–30 (instructions,
few-shot examples, persona/format blocks). For S ⊆ N, let π(S) be the prompt containing
exactly the segments in S (in canonical order), and define the **population utility**

  v(S) = E_{q ~ D, y ~ LLM(π(S), q)} [ u(y, q) ]

where D is the task distribution (e.g., MMLU questions), y is a sampled generation, and
u is a quality score (exact-match correctness, rubric score, or logprob of the reference).
The estimand is the Shapley vector φ(v) ∈ ℝⁿ (and, where relevant, Owen values for a
hierarchy and pairwise interaction indices for redundancy/conflict detection).

Two properties distinguish this from the classical setting and drive all the theory:

- **Doubly-stochastic oracle.** We never observe v(S); we observe
  v̂(S) = (1/r) Σ u(yⱼ, qⱼ) from r question/generation draws. Total noise decomposes into
  between-coalition variance and within-coalition variance σ²(S)/r. Every guarantee must
  hold over both levels jointly (G2), and every budget decision is two-dimensional:
  (#coalitions, replicates per coalition).
- **Non-uniform, cache-dependent query cost.** The cost of evaluating coalition S is not
  1 call but ≈ (uncached prefill tokens) + (decode tokens). Coalitions sharing a prefix
  with previously evaluated ones are cheaper (prompt caching / KV reuse). We therefore
  measure all methods in **tokens**, not calls, and optimize sampling *schedules* under
  this cost model (G4). A key structural fact: the n+1 coalitions along one permutation
  chain ∅ ⊂ {σ₁} ⊂ {σ₁,σ₂} ⊂ … ⊂ N are nested prefixes if segments are appended in
  permutation order, so an entire permutation's marginals cost roughly one full prefill —
  but only if we accept evaluating segment orders that differ from the canonical prompt.
  Order sensitivity is handled explicitly (§5, C3 in risks).

A remark on well-posedness: prompts are ordered, coalitions are not. We fix the canonical
segment order for the *estimand* (v(S) always renders surviving segments in template
order), which keeps the game well-defined and all axioms intact; order-*sensitivity* of
the LLM then only enters as a question of whether cache-friendly reorderings bias cheap
evaluations, which we quantify empirically and correct with a debiasing term (§4, A).

## 4. Three proposed approaches

The approaches are deliberately complementary — an *assumption-free estimator*, a
*structure-exploiting adaptive method*, and a *surrogate/amortized method* — and they sit
on a spectrum of assumptions vs. savings. All three consume the identical coalition
oracle and cost model, hence one experimental harness (§5).

### Approach A — CC-Bernstein: certified segment Shapley from paired coalitions (assumption-free)

**Estimator.** Sample *complementary pairs* (S, N∖S) stratified by coalition size; each
pair updates the running estimate of **every** segment's Shapley value (complementary
contributions, Zhang et al. 2023 = paired sampling, Covert & Lee 2021, unified with the
Stratified-SVARM decomposition). This is the maximal statistical reuse available without
assumptions: per-player MSE = O(σ̄² log n / T) for T evaluations, vs. O(n σ̄²/T) for naive
permutation sampling.

**Novel theory (G1, G2, G4).**
1. *Two-level empirical-Bernstein confidence sequences.* Extend Burgess & Chapman's
   stratified empirical-Bernstein bounds to the doubly-stochastic oracle: within-coalition
   replicates shrink σ²(S)/r, coalition sampling shrinks stratum error, and one anytime-
   valid confidence sequence per segment covers both. Yields the first certificate of the
   form "with prob. ≥ 0.95, instruction 3 contributes between −0.1 and +0.4 accuracy
   points" — and a principled stopping rule replacing the current repo's fixed 200 calls.
2. *Optimal (coalitions × replicates) allocation.* Closed-form Neyman-style split of a
   token budget between new coalitions and replicates, using pilot estimates of the
   variance decomposition; adaptive re-allocation as strata concentrate.
3. *Cache-aware scheduling.* Order the sampled coalitions to maximize shared prefixes
   (greedy prefix-tree batching); prove the schedule leaves unbiasedness intact (schedule
   is chosen measurable w.r.t. the coalition draws, not the observed values) and quantify
   token savings. Where reordering segments unlocks nested-prefix chains, estimate the
   order-effect bias on a small audit sample and either correct or fall back.

**Expected bound.** For n segments, range-1 utility, simultaneous ℓ∞ error ε with prob.
1−δ at token cost Õ((σ²_btw + σ²_win/r) · log(n/δ)/ε²) *pairs*, with the constant
adaptively instance-dependent (small when the game is close to additive) and a measured
2–5× token reduction from prefix scheduling on top.

**Role in the trio:** the reference method — weakest assumptions, certificate always valid.

### Approach B — TreeSHAP-Elim: hierarchical Owen values with certified adaptive pruning

**Idea.** Prompts are hierarchical (prompt → sections → instructions → sentences).
Estimating at the coarsest level first and *recursing only where it matters* can be
exponentially cheaper when importance is concentrated — which prompt practice suggests it
is. TextGenSHAP does this heuristically; we do it with guarantees.

**Estimator.** A successive-elimination procedure over the hierarchy tree:
1. Compute confidence intervals (via the Approach-A estimator restricted to the m coarse
   groups, m ≪ n) for the *group* (Owen) values.
2. Bandit-style elimination (SHAP@k / SAR adapted from top-k identification to a
   user-set relevance threshold τ): groups whose CI lies inside [−τ, τ] are certified
   negligible and **never expanded**; groups whose CI excludes the threshold are split,
   and the procedure recurses with the surrounding coalition context marginalized
   according to the Owen (quotient-game) semantics.

**Novel theory (G3, G6).**
1. First non-asymptotic query-complexity bounds for **Owen value** estimation
   (stratified/paired sampling in the quotient game and within-group games; the survey
   found only asymptotic Hoeffding-type results for Owen values, and no Leverage-SHAP
   analogue).
2. *Instance-dependent complexity* for the full recursive procedure: total evaluations
   scale with Σ_groups 1/max(Δ_g, τ)² — the gap structure of the tree — rather than with
   the number of leaves; when importance is s-sparse at the leaves, cost is
   O(s · depth · polylog) group-games instead of one 2ⁿ game. PAC guarantee: with prob.
   1−δ, every leaf segment with |φ| ≥ τ + ε is surfaced, and every reported value carries
   a valid CI.
3. A matching **lower-bound program**: adapt the Data-Banzhaf-style minimax argument to
   show gap-dependence is necessary — i.e., no algorithm certifies τ-negligibility of a
   group without Ω(1/Δ²) noisy evaluations (G6).

**Deliverable beyond estimation:** this is directly the "instruction efficiency" tool —
its output is a certified triage of a prompt: *keep / delete (certified ≤ τ) / needs more
budget*, at instruction granularity, refined only where the budget buys information.

**Role in the trio:** exploits hierarchy + sparsity of importance; the practical
prompt-debugging method. Assumption-light (guarantees never break; sparsity only affects
*cost*).

### Approach C — SurroSHAP-Cert: validated low-order surrogates with amortization across a prompt library

**Idea.** ContextCite's empirical success implies segment games are close to sparse and
low-interaction-order. If v is exactly k-additive, the Faith-Shap projection means the
Shapley vector is recoverable from a regression with O(nᵏ) unknowns — fit from
O(nᵏ · polylog) cheap coalition evaluations instead of Ω(1/ε²) generic sampling; if the
Möbius/Fourier spectrum is s-sparse, O(s·polylog(n)) suffices. The catch in all prior
work: sparsity is *assumed*, unverifiable, and a wrong assumption silently corrupts the
answer. We make the assumption *testable* and the output *certified regardless*.

**Estimator.**
1. Fit a k-additive (k = 1, 2, escalating) surrogate game ĝ by Lasso/least squares on
   sampled coalitions with Shapley-kernel leverage weights (Leverage SHAP sampling gives
   the fitting-stage guarantee).
2. **Post-hoc validation certificate (G5):** on *fresh held-out* coalitions, bound the
   surrogate residual ‖v − ĝ‖ under the Shapley-kernel distribution with an empirical-
   Bernstein bound; propagate it through the (Lipschitz, explicitly computable) linear map
   from game space to Shapley space to get a distribution-free error bar on φ(ĝ) − φ(v).
   Residual too large → escalate k or fall back to Approach A with the budget already
   spent reused as samples (nothing is wasted).
3. **Amortization across a prompt library.** Organizations hold many related templates
   sharing instruction types. Learn the surrogate *support* (which interactions are
   nonzero) and warm-start priors across templates; per-template work reduces to
   re-estimating coefficients on the learned support + running the validation certificate.
   Certificates stay valid per-template because validation never depends on how the
   support was guessed. Target: amortized per-template cost sublinear in n, the first
   attribution-transfer scheme with per-instance guarantees (the survey found none).

**Novel theory (G5):** the certificate construction itself — a conformal-flavored,
assumption-free wrapper converting *any* surrogate into a Shapley estimate with a valid
(data-dependent) error bar; analysis of the certificate's tightness as a function of true
residual and validation budget; and the escalation policy's total-cost bound.

**Role in the trio:** the aggressive-savings method. When structure is present (the
expected case), it is 10–100× cheaper than A; when absent, its certificate says so and it
degrades gracefully *into* A.

### How the three fit together

| | A: CC-Bernstein | B: TreeSHAP-Elim | C: SurroSHAP-Cert |
|---|---|---|---|
| Assumptions | none | none (hierarchy helps cost only) | none for validity; sparsity helps cost |
| Output | values + CIs, all segments | certified keep/delete triage + values where refined | values + CIs via surrogate |
| Query complexity | Õ(σ² log n / ε²) pairs | gap-dependent, Σ 1/max(Δ,τ)² | O(nᵏ polylog) fit + O(1/ε²_cert) validation |
| Best when | n small, dense importance | importance sparse/hierarchical | low-order interactions; many related templates |
| Main novelty | two-level noise bounds + cache-aware scheduling | first bounded Owen estimation + PAC pruning + lower bound | post-hoc certificates + amortized transfer |

All three share the coalition oracle, the token-cost model, the confidence-interval
machinery, and (A ↔ C fallback, B built on A's estimator) substantial code — which is
what makes a single experimental design possible.

## 5. Shared experimental design

One harness, three plug-in estimators, common baselines and metrics. The harness
generalizes `estimate_importance.py`: a `CoalitionOracle` abstraction (provider-agnostic
async API layer reused from the current repo) exposing `evaluate(S, r)` → r utility
samples + token-cost accounting, with response caching keyed by (coalition, question,
seed) so every method sees identical randomness where possible (common random numbers —
itself a variance-reduction and a fairness guarantee for method comparison).

**Testbeds** (increasing realism):

- **T0 — Synthetic games** (free oracle): random k-additive games, calibrated noise;
  validates bounds exactly (coverage of CIs, tightness vs. theory).
- **T1 — Legacy continuity: 8 adjectives × MMLU/ARC** (this repo's setting). n = 8 ⇒
  exhaustive 2⁸ = 256 coalitions with heavy replication gives *exact-up-to-noise ground
  truth*, and directly re-answers the original project's question with error bars.
  Backwards comparison: how wrong / how overconfident was the original estimator?
- **T2 — Instruction attribution** (the headline setting): system prompts with n = 10–20
  heterogeneous instructions (format constraints, persona, chain-of-thought triggers,
  safety clauses, redundant duplicates and *deliberately conflicting pairs* planted as
  known positives) evaluated on IFEval-style instruction-following and MMLU-Pro/GSM8K
  accuracy. Ground truth by exhaustive enumeration where n ≤ 14, and by a
  10⁶-evaluation "gold" run of estimator A for larger n.
- **T3 — Hierarchical/RAG**: prompts with sections × instructions (for B's hierarchy) and
  retrieved-document segments (comparability with ContextCite/Cluster Shapley); a
  50-template prompt library with shared instruction types (for C's amortization).

**Baselines:** permutation MC (Castro), stratified Castro-2017, KernelSHAP as implemented
in the current repo (including its off-distribution sampler, as an ablation), unbiased
KernelSHAP + paired sampling (Covert & Lee), Leverage SHAP, SVARM, ContextCite weights,
LOO/AttriBoT-style, ProCut-style pruning.

**Metrics (identical across approaches — this is the shared design):**

1. *Accuracy vs. budget*: ℓ∞/ℓ₂ error and Kendall-τ to ground truth as a function of
   **tokens spent** (primary) and calls (secondary) — cost curves, not point estimates.
2. *Certificate quality*: empirical coverage of the claimed (ε,δ) intervals (target:
   ≥ 1−δ, and *not* ≫ 1−δ, i.e., tightness), CI width vs. budget.
3. *Decision quality*: precision/recall of certified-negligible and top-k sets against
   planted redundant/conflicting instructions (T2) and against ground-truth ranking;
   downstream check à la ProCut — accuracy of the pruned prompt vs. token reduction.
4. *Efficiency accounting*: uncached-prefill-token savings from cache-aware scheduling
   (A), evaluations saved by pruning (B, vs. flat estimation at equal certified output),
   amortization curve across the template library (C: per-template cost vs. #templates).
5. *Robustness*: two models (one open-weights via local vLLM with real prefix caching —
   measured, not simulated — one API model), two utility types (binary correctness,
   rubric/judge score), order-sensitivity audit (canonical vs. cache-friendly ordering
   deltas).

**Shared infrastructure quirks worth stating up front:** all randomness seeded and
logged (the current repo's resume-from-JSON pattern, generalized); every LLM response
cached to disk so that *no experiment is ever paid for twice* — across methods, the
cache is itself the biggest operation-count optimization; judge prompts and eval sets
version-pinned.

**Success criteria.** (i) Certified CIs with ≥95% empirical coverage at ≤ 1/10 the token
cost of permutation sampling at equal ℓ∞ error on T1/T2; (ii) B certifies ≥90% of planted
negligible instructions with zero false eliminations at gap-predicted budgets; (iii) C
matches A's accuracy at ≤ 1/10 cost on structured T2 games and its certificate correctly
*refuses* on adversarial dense games (T0); (iv) at least one theorem per approach (the
two-level Bernstein bound; the Owen sample-complexity bound + elimination PAC guarantee;
the surrogate certificate), plus the Ω(1/Δ²) lower bound as a stretch goal.

## 6. Risks and mitigations

- **R1 — Order sensitivity breaks the coalition abstraction.** Mitigation: canonical-order
  estimand (§3) keeps the game well-defined; the order-audit (metric 5) bounds the bias
  of cache-friendly reordering empirically; OrdShap-style position-aware extensions are a
  documented fallback/extension, not a blocker.
- **R2 — Within-coalition noise dominates (binary 0/1 utilities).** This is a *feature*
  for G2 (it is exactly the regime where the two-level allocation matters); rubric/logprob
  utilities on T2 give a lower-noise counterpoint.
- **R3 — Segment games turn out dense/high-order.** C degrades into A by design; B's
  guarantees never depend on sparsity; the trio was chosen so at least the assumption-free
  track always lands.
- **R4 — API prompt caching is opaque (billing granularity, eviction).** Primary
  cache-economics results on local vLLM with explicit prefix caching; API models report
  call/token counts under the provider's published caching discounts.

## 7. Key references

*Applied LLM attribution:* TokenSHAP (arXiv:2407.10114); TextGenSHAP (arXiv:2312.01279);
ContextCite (arXiv:2409.00729); AttriBoT (arXiv:2411.15102); TracLLM (arXiv:2506.04202);
ProCut (arXiv:2508.02053); Cluster Shapley (arXiv:2505.23842); MaxShapley
(arXiv:2512.05958); llmSHAP (arXiv:2511.01311); OrdShap (arXiv:2507.11855); SPEX
(arXiv:2502.13870); ProxySPEX (arXiv:2505.17495).

*Estimation theory:* Castro et al. 2009 (Comput. Oper. Res.); Maleki et al.
(arXiv:1306.4265); Castro et al. 2017 (stratified/Neyman); Burgess & Chapman (IJCAI
2021); Zhang et al., complementary contributions (SIGMOD 2023); SVARM / Stratified SVARM
(arXiv:2302.00736); SVARM-IQ (arXiv:2401.13371); Covert & Lee (arXiv:2012.01536);
Leverage SHAP (arXiv:2410.01917); Mitchell et al. (JMLR 2022, arXiv:2104.12199);
Faith-Shap (JMLR 2023, arXiv:2203.00870); k-additive surrogates (arXiv:2502.04763);
FourierSHAP (arXiv:2410.06300); SHAP@k (arXiv:2307.04850); antithetic top-k
(arXiv:2504.02019); Data Banzhaf (arXiv:2205.15466); Kernel Banzhaf (arXiv:2410.08336);
One-Sample-Fits-All (arXiv:2410.23808); FastSHAP (arXiv:2107.07436); stochastic
amortization (arXiv:2401.15866); group testing (arXiv:1902.10275) + corrected note
(arXiv:2302.11431); groupShapley (arXiv:2106.12228); Owen stratified sampling
(Saavedra-Nieves et al., Ann. OR 2022); O-Shap (arXiv:2602.17107); supermodular FPRAS
(Liben-Nowell et al., COCOON 2012).
