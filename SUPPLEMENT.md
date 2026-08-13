# Code & Data Supplement

**Anytime-Valid Certificates for Prompt-Segment Attribution: Certifying Safe and Harmful Segments of LLM Prompts**

This archive is the code and data supplement for the paper. It regenerates every
empirical claim, table, and figure. The estimator library is CPU-only Python;
the LLM experiments replay from a saved response cache and require no API key to
reproduce the reported numbers from the committed results, or an
OpenAI-compatible key to regenerate the cache from scratch.

## Contents

```
segshap/                core library (games, certificates, estimators)
  games.py              doubly-stochastic pay-per-call coalition oracle
  bounds.py             betting confidence sequences + empirical-Bernstein
  estimators/
    cc_bernstein.py     CC-Bet (Alg. 1)
    perm_bet.py         Perm-Bet (Cor. 1); betting or EB intervals
    tree_elim.py        screening stage of screen-and-gate (Alg. 2)
    gate.py             intervention gate on Delta_P (Thm. "Intervention gate")
    surrogate.py        Mobius surrogate control variate
    baselines.py        permutation MC, KernelSHAP
    external.py         Leverage SHAP, SVARM (shapiq)
  llm.py                PromptSegmentGame (OpenAI-compatible; offline cache mode)
  metrics.py
experiments/            one runnable script per experiment (see CLAIMS_TO_ARTIFACTS.md)
tests/                  22 correctness/coverage tests vs exact ground truth
results/                machine-readable outputs backing every table/figure
figures/                fig_main.pdf, fig_t4.pdf (+ generator)
*_RESULTS.md            human-readable result tables
PROMPTS.md              every segment, injection, and metric, verbatim
REPRODUCIBILITY_CHECKLIST.md   AAAI reproducibility checklist, filled
CLAIMS_TO_ARTIFACTS.md  paper claim -> script -> result-file map
requirements.txt        pinned dependency versions
LICENSE                 MIT
```

The ~2 GB LLM response cache is **excluded** (it exceeds supplement size
limits). All reported numbers are recomputable from the committed `results/`
JSON without it; regenerating the cache from the API is documented below.

## Install

```bash
python3 -m venv venv && source venv/bin/activate      # Python 3.12
pip install -r requirements.txt
pip install -e .
```

## Reproduce without any API key

```bash
pytest tests/                          # 22 tests vs exact ground truth
python3 experiments/run_synthetic.py   # E1/E2 synthetic (T0)
python3 experiments/run_gate_demo.py   # Thm gate: unanimity + planted-null
python3 experiments/run_stress.py      # E5 1000-trial coverage stress
python3 experiments/run_t1_removal.py  # Delta vs phi on T1 (from committed grid)
python3 experiments/run_ablation.py    # E5 decomposition x interval (cache replay)
python3 experiments/run_perm_bet.py    # Perm-Bet on T2/T4 (cache replay)
python3 experiments/run_t4_compound.py # compound metric (cache replay)
python3 experiments/run_replicate_allocation.py   # E4 allocation (cache replay)
python3 experiments/make_figures.py    # fig_main.pdf, fig_t4.pdf
```

Scripts that read the LLM cache run in a hard **offline mode**
(`PromptSegmentGame.openrouter(..., offline=True)`): no client is created, no
key is read, and any cache miss raises `CacheMiss`. Each such script asserts
`live_calls == 0`. Because the committed `results/*.json` already contain the
exact ground truth and estimator outputs, every table can also be inspected
directly with no execution.

## Regenerate the cache from the API (optional)

```bash
export OPENROUTER_API_KEY=sk-...     # or any OpenAI-compatible endpoint
python3 experiments/run_t1_openrouter.py                        # T1  (~$0.3)
python3 experiments/run_t2_openrouter.py                        # T2 Qwen (~$4)
python3 experiments/run_t2_openrouter.py --model google/gemma-4-26b-a4b-it
python3 experiments/run_t3_scale.py                             # T3 n=20 (~$4)
python3 experiments/run_t4_safety.py                            # T4 Qwen (~$0.3)
python3 experiments/run_t4_safety.py --model google/gemma-4-26b-a4b-it
```

Temperature 0, reasoning disabled, seeds fixed; every response is written to
`cache/` so re-runs are deterministic and free. Models are served through
OpenRouter (`qwen/qwen3.5-9b`, `google/gemma-4-26b-a4b-it`).

See `CLAIMS_TO_ARTIFACTS.md` for the exact script and result file behind each
numbered claim, table, and figure, and `REPRODUCIBILITY_CHECKLIST.md` for the
AAAI checklist.
