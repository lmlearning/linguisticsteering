# T2 results: 12-instruction attribution with hierarchy on a real LLM

Model: `google/gemma-4-26b-a4b-it` via OpenRouter (reasoning disabled, temperature 0). 12 instruction segments in 4 groups x 30 MMLU questions; exhaustive 2^12 = 4,096-coalition grid (122871 calls, $8.56, 163 min) gives exact Shapley, Owen, and interaction ground truth.

Accuracy with no segments: 0.800; with all twelve: 0.867.

## Exact segment Shapley values (ground truth)

| segment | group | exact phi | exact Owen |
|---|---|---|---|
| brief | format | +0.0680 | +0.1116 |
| letter_only | format | +0.0376 | +0.0838 |
| verbose | format | +0.0269 | +0.0269 |
| trivia | junk | +0.0171 | +0.0310 |
| date | junk | +0.0137 | +0.0269 |
| weather | junk | +0.0072 | +0.0144 |
| teacher | persona | -0.0000 | -0.0176 |
| bias_a | persona | -0.0001 | -0.0051 |
| expert | persona | -0.0105 | -0.0218 |
| eliminate | reasoning | -0.0273 | -0.0681 |
| doublecheck | reasoning | -0.0304 | -0.0542 |
| steps | reasoning | -0.0356 | -0.0611 |

## Planted structure recovered from the exact grid

Group Shapley (quotient game): persona -0.0444, format +0.2222, reasoning -0.1833, junk +0.0722

Top-5 pairwise Mobius interactions:

| pair | interaction |
|---|---|
| doublecheck + weather | +0.2000 |
| doublecheck + trivia | +0.2000 |
| doublecheck + date | +0.2000 |
| steps + eliminate | -0.2000 |
| teacher + doublecheck | +0.1667 |

## Flat-estimator comparison (8 seeds, replayed from cache)

| budget | estimator | linf | kendall | P[all CIs cover] | mean CI width |
|---|---|---|---|---|---|
| 3000 | permutation_mc | 0.0682 | 0.498 | 1.00 | 1.2502 |
| 3000 | kernel_shap | 0.0501 | 0.686 | n/a | n/a |
| 3000 | cc_bernstein | 0.0392 | 0.731 | 1.00 | 6.3472 |
| 3000 | surrogate_cc | 0.0410 | 0.716 | 1.00 | 16.4736 |
| 10000 | permutation_mc | 0.0371 | 0.629 | 1.00 | 0.7346 |
| 10000 | kernel_shap | 0.0381 | 0.814 | n/a | n/a |
| 10000 | cc_bernstein | 0.0185 | 0.833 | 1.00 | 2.2240 |
| 10000 | surrogate_cc | 0.0260 | 0.795 | 1.00 | 4.1656 |
| 30000 | permutation_mc | 0.0240 | 0.792 | 1.00 | 0.4500 |
| 30000 | kernel_shap | 0.0295 | 0.902 | n/a | n/a |
| 30000 | cc_bernstein | 0.0115 | 0.913 | 1.00 | 0.9156 |
| 30000 | surrogate_cc | 0.0149 | 0.894 | 1.00 | 1.4963 |

## Hierarchical triage (TreeSHAP-Elim, tau = 0.08, 200k replayed calls, 8 seeds)

- 'junk' group certified negligible in 0% of runs
- false eliminations (true |group value| > tau certified anyway): 0
- member Owen CIs cover exact Owen values: 100% of runs

Total new spend for T2: $8.56 (everything after the grid replays the cache).

Reproduce: `OPENROUTER_API_KEY=... python3 experiments/run_t2_openrouter.py`.
