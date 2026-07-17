# T3 results: 20 segments — beyond exhaustive enumeration

Model: `qwen/qwen3.5-9b`. n = 20 segments (2^20 ~ 1.05M coalitions: exact enumeration impossible). Reference = certified gold standard: CC-Bernstein at 120,000 calls whose simultaneous 95% betting CIs (mean width 0.1633) bound its own error.

## Certified gold attribution (top and bottom segments)

| segment | group | gold phi | 95% CI |
|---|---|---|---|
| verbose | format | +0.1085 | [+0.030, +0.191] |
| short | style | +0.0415 | [-0.040, +0.124] |
| brief | format | +0.0327 | [-0.049, +0.115] |
| nopunct | format | +0.0251 | [-0.054, +0.106] |
| trivia | junk | +0.0240 | [-0.056, +0.107] |
| letter_only | format | +0.0117 | [-0.071, +0.093] |
| doublecheck | reasoning | -0.0050 | [-0.090, +0.073] |
| confident | persona | -0.0068 | [-0.084, +0.077] |
| expert | persona | -0.0115 | [-0.089, +0.073] |
| steps | reasoning | -0.0224 | [-0.108, +0.058] |

## Estimator comparison at 30,000 calls (vs gold)

| estimator | linf vs gold | kendall vs gold | mean CI width | consistent with gold |
|---|---|---|---|---|
| cc_bernstein | 0.0134 | 0.768 | 0.2909 | yes |
| surrogate_cc | 0.0210 | 0.547 | 0.3544 | yes |
| kernel_shap | 0.0288 | 0.663 | n/a | n/a |
| leverage_shap | 0.0677 | 0.389 | n/a | n/a |
| svarm | 0.0579 | 0.589 | n/a | n/a |

## Hierarchical triage (5 groups of 4, tau = 0.1, 60,000 calls)

| group | value | halfwidth | verdict |
|---|---|---|---|
| persona | +0.0144 | 0.0825 | certified negligible |
| format | +0.1954 | 0.0810 | expanded |
| reasoning | -0.0999 | 0.0795 | expanded |
| junk | +0.0439 | 0.0815 | expanded |
| style | +0.0495 | 0.0835 | expanded |

Total spend for T3: $3.53. Reproduce: `OPENROUTER_API_KEY=... python3 experiments/run_t3_scale.py` (same seeds replay from the response cache).
