# T2 results: 12-instruction attribution with hierarchy on a real LLM

Model: `qwen/qwen3.5-9b` via OpenRouter (reasoning disabled, temperature 0). 12 instruction segments in 4 groups x 30 MMLU questions; exhaustive 2^12 = 4,096-coalition grid (0 calls, $0.00, 0 min) gives exact Shapley, Owen, and interaction ground truth.

Accuracy with no segments: 0.633; with all twelve: 0.767.

## Exact segment Shapley values (ground truth)

| segment | group | exact phi | exact Owen |
|---|---|---|---|
| verbose | format | +0.1142 | +0.1213 |
| brief | format | +0.0886 | +0.0824 |
| letter_only | format | +0.0298 | +0.0491 |
| trivia | junk | +0.0151 | +0.0250 |
| bias_a | persona | +0.0143 | +0.0148 |
| weather | junk | -0.0022 | +0.0042 |
| date | junk | -0.0060 | +0.0069 |
| doublecheck | reasoning | -0.0079 | -0.0148 |
| teacher | persona | -0.0131 | -0.0185 |
| expert | persona | -0.0175 | -0.0157 |
| eliminate | reasoning | -0.0348 | -0.0398 |
| steps | reasoning | -0.0473 | -0.0815 |

## Planted structure recovered from the exact grid

Group Shapley (quotient game): persona -0.0194, format +0.2528, reasoning -0.1361, junk +0.0361

Top-5 pairwise Mobius interactions:

| pair | interaction |
|---|---|
| eliminate + doublecheck | -0.2667 |
| verbose + eliminate | -0.2333 |
| expert + eliminate | -0.2000 |
| letter_only + steps | +0.2000 |
| brief + steps | +0.2000 |

## Flat-estimator comparison (8 seeds, replayed from cache)

| budget | estimator | linf | kendall | P[all CIs cover] | mean CI width |
|---|---|---|---|---|---|
| 3000 | permutation_mc | 0.0780 | 0.455 | 1.00 | 1.2502 |
| 3000 | kernel_shap | 0.0583 | 0.678 | n/a | n/a |
| 3000 | cc_bernstein | 0.0488 | 0.629 | 1.00 | 0.8519 |
| 3000 | surrogate_cc | 0.0572 | 0.598 | 1.00 | 1.9369 |
| 3000 | leverage_shap | 0.0811 | 0.451 | n/a | n/a |
| 3000 | svarm | 0.0781 | 0.470 | n/a | n/a |
| 10000 | permutation_mc | 0.0446 | 0.583 | 1.00 | 0.7346 |
| 10000 | kernel_shap | 0.0364 | 0.818 | n/a | n/a |
| 10000 | cc_bernstein | 0.0263 | 0.803 | 1.00 | 0.4129 |
| 10000 | surrogate_cc | 0.0276 | 0.822 | 1.00 | 0.7089 |
| 10000 | leverage_shap | 0.0741 | 0.542 | n/a | n/a |
| 10000 | svarm | 0.0788 | 0.530 | n/a | n/a |
| 30000 | permutation_mc | 0.0275 | 0.780 | 1.00 | 0.4500 |
| 30000 | kernel_shap | 0.0289 | 0.905 | n/a | n/a |
| 30000 | cc_bernstein | 0.0109 | 0.920 | 1.00 | 0.2208 |
| 30000 | surrogate_cc | 0.0151 | 0.894 | 1.00 | 0.3008 |
| 30000 | leverage_shap | 0.0595 | 0.648 | n/a | n/a |
| 30000 | svarm | 0.0518 | 0.686 | n/a | n/a |

## Hierarchical triage (TreeSHAP-Elim, tau = 0.08, 200k replayed calls, 8 seeds)

- 'junk' group certified negligible in 75% of runs
- false eliminations (true |group value| > tau certified anyway): 0
- member Owen CIs cover exact Owen values: 100% of runs

Total new spend for T2: $0.00 (everything after the grid replays the cache).

Reproduce: `OPENROUTER_API_KEY=... python3 experiments/run_t2_openrouter.py`.
