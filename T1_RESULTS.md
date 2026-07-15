# T1 results: certified instruction attribution on a real LLM

Model: `qwen/qwen3.5-9b` via OpenRouter (reasoning disabled, temperature 0). 8 instruction segments x 60 MMLU questions; exhaustive 2^8 = 256-coalition grid (0 calls, $0.00) gives exact ground truth.

Accuracy with no segments: 0.733; with all eight: 0.267.

## Exact segment Shapley values (ground truth) vs. certified estimate

CC-Bernstein at 8,000 oracle calls, simultaneous 95% CIs:

| segment | exact phi | estimate | 95% CI |
|---|---|---|---|
| format | +0.0144 | +0.0165 | [-0.199, +0.210] |
| expert | +0.0092 | -0.0042 | [-0.221, +0.186] |
| doublecheck | +0.0074 | +0.0024 | [-0.216, +0.179] |
| distractor | -0.0039 | -0.0013 | [-0.209, +0.189] |
| cot | -0.0078 | -0.0247 | [-0.229, +0.178] |
| concise | -0.0253 | -0.0135 | [-0.229, +0.176] |
| french | -0.0308 | -0.0234 | [-0.257, +0.172] |
| sabotage | -0.4299 | -0.4534 | [-0.599, -0.259] |

## Estimator comparison (10 seeds, replayed from the response cache)

| budget | estimator | linf | kendall | P[all CIs cover] | mean CI width |
|---|---|---|---|---|---|
| 1000 | permutation_mc | 0.0956 | 0.494 | 1.00 | 1.6776 |
| 1000 | kernel_shap | 0.0846 | 0.543 | n/a | n/a |
| 1000 | cc_bernstein | 0.0846 | 0.457 | 1.00 | 1.2183 |
| 1000 | surrogate_cc | 0.0752 | 0.500 | 1.00 | 3.1900 |
| 3000 | permutation_mc | 0.0592 | 0.498 | 1.00 | 1.0465 |
| 3000 | kernel_shap | 0.0693 | 0.486 | n/a | n/a |
| 3000 | cc_bernstein | 0.0341 | 0.693 | 1.00 | 0.6768 |
| 3000 | surrogate_cc | 0.0384 | 0.514 | 1.00 | 1.3039 |
| 8000 | permutation_mc | 0.0339 | 0.593 | 1.00 | 0.6809 |
| 8000 | kernel_shap | 0.0569 | 0.707 | n/a | n/a |
| 8000 | cc_bernstein | 0.0250 | 0.771 | 1.00 | 0.3880 |
| 8000 | surrogate_cc | 0.0252 | 0.729 | 1.00 | 0.5835 |

Total spend for everything above: $0.00. All estimator runs replay the cached grid responses, so they cost nothing beyond the grid — the shared-cache design at work.

Reproduce: `OPENROUTER_API_KEY=... python3 experiments/run_t1_openrouter.py`.
