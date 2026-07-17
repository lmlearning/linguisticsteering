# Reproducibility guide

This package contains the full source, experiments, results, and paper for
*Certified Attribution of Prompt Segments* (`paper/segshap.pdf`).

## Install

```bash
pip install -r requirements.txt
pip install -e .            # installs the `segshap` package
```

Python 3.10+. The core library (games, estimators, certificates) has no
LLM dependency; `openai` is needed only for the live experiments T1–T4.

## What reproduces without any API key

These use only synthetic games or the exact ground truth shipped in `results/`:

```bash
pytest tests/                          # 17 tests vs exact ground truth
python3 experiments/run_synthetic.py   # T0: regenerates RESULTS.md
```

The estimator-comparison and allocation studies **replay from the on-disk
response cache** and therefore need no API calls — *if* the cache is present.
The cache (~2 GB) is intentionally excluded from this package. Two options:

- **Inspect the committed outputs directly** — every table in the paper is
  backed by a `*_RESULTS.md` file and a `results/*.json` at the repo root
  (see `paper/README.md` for the paper-element → file map). No compute needed.
- **Regenerate the cache** by rerunning the live experiments with a key
  (below); same seeds are deterministic, so a fresh run reproduces the numbers.

## Live experiments (need an OpenAI-compatible key)

All four use OpenRouter by default via `PromptSegmentGame.openrouter(...)`:

```bash
export OPENROUTER_API_KEY=sk-...
python3 experiments/run_t1_openrouter.py                       # 8 segments, ~$0.3
python3 experiments/run_t2_openrouter.py                       # 12 segments, ~$4
python3 experiments/run_t2_openrouter.py --model google/gemma-4-26b-a4b-it
python3 experiments/run_t3_scale.py                            # 20 segments, ~$4
python3 experiments/run_t4_safety.py                           # injection resistance, ~$0.3
python3 experiments/run_replicate_allocation.py               # free (needs T2 caches)
python3 experiments/compare_t2_models.py results/t2_qwen3.5-9b.json \
                                         results/t2_gemma-4-26b-a4b-it.json
```

Each script writes its `*_RESULTS.md` and `results/*.json`, and caches every
response under `cache/`, so a re-run costs nothing beyond new coalitions.
All runs are resumable: rerunning continues from whatever the cache holds.

## Layout

```
segshap/                 core library
  games.py               budgeted doubly-stochastic coalition oracle
  bounds.py              betting confidence sequences + empirical Bernstein
  estimators/            cc_bernstein, tree_elim, surrogate, baselines, external
  llm.py                 PromptSegmentGame (OpenAI-compatible, cached)
  metrics.py
experiments/             T0–T4 + allocation + cross-model driver
results/                 machine-readable outputs (exact truth + estimator rows)
*_RESULTS.md             human-readable result tables
paper/                   segshap.tex, segshap.pdf, README.md
FOLLOW_UP_PROPOSAL.md    the research programme and SOTA survey
```

## Theory ↔ code

- Thm. 1 (certificate validity, adaptivity-safe) →
  `segshap/bounds.py::betting_interval`, `StratifiedEstimate.intervals`
- Prop. 2 (two-level allocation) → `experiments/run_replicate_allocation.py`
- Thm. 3 (safe pruning) → `segshap/estimators/tree_elim.py::hierarchical_owen`

## Paper

```bash
cd paper && pdflatex segshap.tex && pdflatex segshap.tex
```

Compiles with a stock TeX Live install via an embedded AAAI-style shim; swap
in the official `aaai2026.sty` for camera-ready (see `paper/README.md`).
