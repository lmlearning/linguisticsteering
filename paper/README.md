# Paper: Certified Attribution of Prompt Segments

`segshap.tex` — AAAI Alignment-track submission draft. Builds a 6-page,
two-column PDF with a stock TeX Live install:

```bash
pdflatex segshap.tex   # twice for cross-references
```

The preamble is an **AAAI-style shim** (a self-contained approximation of the
two-column AAAI layout) so the paper compiles anywhere. For the official
camera-ready, replace the block marked `AAAI-STYLE SHIM` with

```latex
\documentclass[letterpaper]{article}
\usepackage{aaai2026}
```

from the AAAI author kit; the body uses only standard commands
(`amsthm`, `algorithm`/`algpseudocode`, `booktabs`).

## Where the numbers come from

Every table is populated from the committed experiment outputs at the repo
root, all reproducible from the response caches:

| Paper element | Source |
|---|---|
| Table 1 (T2 estimator comparison) | `T2_RESULTS.md`, `results/t2_qwen3.5-9b.json` |
| Table 2 (T3 scale, certified gold) | `T3_RESULTS.md`, `results/t3_scale_qwen3.5-9b.json` |
| Table 3 (T4 injection-resistance) | `T4_RESULTS.md`, `results/t4_safety_qwen3.5-9b.json` |
| Betting vs Bernstein widths | `RESULTS.md`, git history of the CI rerun |
| Allocation (Prop. 2) | `REPLICATE_ALLOCATION.md` |
| Sabotage certificate (T1) | `T1_RESULTS.md` |

## Theory ↔ code

- Thm. 1 (certificate validity) → `segshap/bounds.py::betting_interval`,
  `StratifiedEstimate.intervals`
- Prop. 2 (two-level allocation) → `experiments/run_replicate_allocation.py`
- Thm. 3 (safe pruning) → `segshap/estimators/tree_elim.py::hierarchical_owen`
