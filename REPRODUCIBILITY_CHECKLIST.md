# AAAI-27 Reproducibility Checklist

Responses for *Anytime-Valid Certificates for Prompt-Segment Attribution*.

## 1. General

- **This paper makes theoretical contributions.** Yes.
  - All assumptions and restrictions stated (bounded metric `u∈[0,1]`;
    predictable allocation/stakes; conditional-mean per stratum): **yes** (§Theory).
  - Formal claims numbered and cross-referenced: **yes** (Thm. 1, Cor. 1,
    Prop. 1, Thms. 2–3, Remark).
  - Proofs or proof sketches for all claims: **yes** (paper; full proofs in the
    technical appendix). Constructions realized in `segshap/` (see
    `CLAIMS_TO_ARTIFACTS.md`).
- **This paper makes experimental contributions.** Yes.

## 2. Theoretical claims

- Statements of all theorems, lemmas, propositions numbered: **yes**.
- Clearly stated assumptions: **yes** (bounded `u`; predictable, data-dependent
  policies; two-level noise oracle).
- Complete proofs / proof sketches with full versions in appendix: **yes**.

## 3. Datasets

- All novel datasets introduced: the segment/injection prompt sets are novel and
  documented verbatim in `PROMPTS.md`; the induced coalition "games" are exact
  functions of a fixed question set.
- All datasets drawn from existing sources cited: **yes** — MMLU
  (Hendrycks et al.) is the task distribution `D`; sampled question ids and the
  sampler seed are in `experiments/t1_questions.json` / `run_*` scripts.
- Train/test splits: not applicable (no model training; attribution is
  post-hoc). The "game" is the finite empirical value function over the fixed
  question set, which every estimator targets.
- Synthetic data generators included: **yes**
  (`segshap.games.random_sparse_game`, `SyntheticMobiusGame`; T0, stress, gate,
  ablation-synthetic).

## 4. Code

- Code to reproduce all experimental results is included: **yes**
  (`segshap/`, `experiments/`, `tests/`).
- Instructions to reproduce, with exact commands: **yes** (`SUPPLEMENT.md`).
- Dependencies and versions pinned: **yes** (`requirements.txt`: Python 3.12,
  NumPy 2.4.6, scikit-learn 1.9.0, shapiq 1.4.1). Estimator library is CPU-only.
- License: **yes** (MIT, `LICENSE`).
- LLM responses: cached and replayed for exact reproduction; scripts run in a
  hard offline mode that forbids live calls (`CacheMiss` on any miss;
  `live_calls == 0` asserted). Cache regeneration from an OpenAI-compatible API
  is documented (`SUPPLEMENT.md`).

## 5. Experimental setup and results

- Range of hyperparameters and how selected: `δ=0.05` (0.10 in a stress block);
  `r` (replicates) swept `{1,…,16}` in E4, `r=1` (call-optimal) elsewhere;
  screening `τ` and gate `τ_Δ` stated per experiment; betting grid = 401
  candidate means. All in the scripts and paper.
- Number of runs / seeds: reported per experiment (8 seeds T2; 10 seeds
  T0/T1/T4; 1000/300/1000 trials in the stress test; 36 runs for Perm-Bet
  false-sign check). **Stated in every table.**
- Central tendency and variation: mean ± standard deviation over seeds; Wilson
  upper bounds for coverage failure rates.
- Statistical significance / uncertainty: the paper's core deliverable *is* a
  certified confidence interval; coverage is empirically validated (1.00 in all
  cells; 0 failures across the stress test).
- Compute: estimator experiments are CPU-only (minutes on a laptop). The LLM
  cache was produced for ≈ \$13 total across all six grids via a hosted API;
  reproduction from cache uses no accelerator and no API.
- Exact command for each result: `CLAIMS_TO_ARTIFACTS.md`.

## 6. Broader considerations

- Compute cost reported: **yes** (above).
- Reproducibility of LLM calls: temperature 0, reasoning disabled, fixed seeds,
  responses cached; the paper notes results are `D`-averages, not worst-case.
- Limitations stated: **yes** (fixed segment order; idealized token-cost model;
  judge-dependence; dual-use), §Discussion.
