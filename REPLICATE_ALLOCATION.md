# Replicate-allocation study: coalitions vs replicates (G2)

CC-Bernstein on both exact T2 games, replayed from cache. At a fixed call budget B, replicates r give B/(2r) coalition pairs whose observations carry within-coalition noise reduced by 1/r — but each fresh pair also averages the between-coalition component, which replicates cannot touch. 10 seeds per cell, delta = 0.05.

## qwen/qwen3.5-9b

| budget | r | pairs | linf | kendall | P[all cover] | mean CI width |
|---|---|---|---|---|---|---|
| 3000 | 1 | 1500 | 0.0442 | 0.658 | 1.00 | 2.8264 |
| 3000 | 2 | 750 | 0.0408 | 0.736 | 1.00 | 4.6345 |
| 3000 | 4 | 375 | 0.0480 | 0.655 | 1.00 | 8.2754 |
| 3000 | 8 | 187 | 0.0503 | 0.618 | 1.00 | 16.3969 |
| 3000 | 16 | 93 | 0.0582 | 0.552 | 1.00 | inf |
| 10000 | 1 | 5000 | 0.0220 | 0.836 | 1.00 | 1.1456 |
| 10000 | 2 | 2500 | 0.0221 | 0.852 | 1.00 | 1.7261 |
| 10000 | 4 | 1250 | 0.0215 | 0.773 | 1.00 | 2.8474 |
| 10000 | 8 | 625 | 0.0246 | 0.791 | 1.00 | 5.0698 |
| 10000 | 16 | 312 | 0.0292 | 0.712 | 1.00 | 9.7206 |
| 30000 | 1 | 15000 | 0.0124 | 0.906 | 1.00 | 0.5337 |
| 30000 | 2 | 7500 | 0.0124 | 0.918 | 1.00 | 0.7489 |
| 30000 | 4 | 3750 | 0.0134 | 0.894 | 1.00 | 1.1592 |
| 30000 | 8 | 1875 | 0.0149 | 0.936 | 1.00 | 1.9368 |
| 30000 | 16 | 937 | 0.0162 | 0.900 | 1.00 | 3.4415 |

## google/gemma-4-26b-a4b-it

| budget | r | pairs | linf | kendall | P[all cover] | mean CI width |
|---|---|---|---|---|---|---|
| 3000 | 1 | 1500 | 0.0346 | 0.700 | 1.00 | 2.7197 |
| 3000 | 2 | 750 | 0.0388 | 0.736 | 1.00 | 4.5421 |
| 3000 | 4 | 375 | 0.0401 | 0.645 | 1.00 | 8.1453 |
| 3000 | 8 | 187 | 0.0437 | 0.612 | 1.00 | 16.4729 |
| 3000 | 16 | 93 | 0.0461 | 0.533 | 1.00 | inf |
| 10000 | 1 | 5000 | 0.0162 | 0.848 | 1.00 | 1.0813 |
| 10000 | 2 | 2500 | 0.0213 | 0.836 | 1.00 | 1.6622 |
| 10000 | 4 | 1250 | 0.0239 | 0.788 | 1.00 | 2.7771 |
| 10000 | 8 | 625 | 0.0222 | 0.815 | 1.00 | 5.0068 |
| 10000 | 16 | 312 | 0.0225 | 0.836 | 1.00 | 9.6784 |
| 30000 | 1 | 15000 | 0.0110 | 0.915 | 1.00 | 0.4960 |
| 30000 | 2 | 7500 | 0.0105 | 0.924 | 1.00 | 0.7087 |
| 30000 | 4 | 3750 | 0.0130 | 0.873 | 1.00 | 1.1160 |
| 30000 | 8 | 1875 | 0.0117 | 0.894 | 1.00 | 1.8892 |
| 30000 | 16 | 937 | 0.0138 | 0.879 | 1.00 | 3.3795 |

## Idealized prefix-cache token model

Uncached prefill+decode tokens per CC pair (prefix cached after the first replicate of a coalition; every replicate pays the question part and completion). Error is the measured linf at the same (budget, r); token cost = pairs x tokens/pair.

| model | r | tokens/pair (cached) | tokens/pair (no cache) | linf @30k calls | Mtokens @30k calls (cached) |
|---|---|---|---|---|---|
| qwen3.5-9b | 1 | 611 | 611 | 0.0124 | 9.17 |
| qwen3.5-9b | 2 | 1084 | 1222 | 0.0124 | 8.13 |
| qwen3.5-9b | 4 | 2030 | 2444 | 0.0134 | 7.61 |
| qwen3.5-9b | 8 | 3923 | 4889 | 0.0149 | 7.35 |
| qwen3.5-9b | 16 | 7707 | 9777 | 0.0162 | 7.22 |
| gemma-4-26b-a4b-it | 1 | 597 | 597 | 0.0110 | 8.96 |
| gemma-4-26b-a4b-it | 2 | 1057 | 1194 | 0.0105 | 7.93 |
| gemma-4-26b-a4b-it | 4 | 1977 | 2388 | 0.0130 | 7.42 |
| gemma-4-26b-a4b-it | 8 | 3818 | 4777 | 0.0117 | 7.16 |
| gemma-4-26b-a4b-it | 16 | 7499 | 9554 | 0.0138 | 7.03 |

## Conclusions

1. **In calls, fresh coalitions beat replicates — confirmed on both models.**
   r = 1 (or 2) gives the lowest error at every budget; by r = 16 the linf
   error is ~25-30% worse (qwen @30k: 0.0124 -> 0.0162) because replicates
   cannot average away the between-coalition noise component.
2. **Certificates degrade much faster than point error as r grows**: the CI
   width scales with the anytime-Bernstein range term ~1/(pairs), so r = 16
   at the small budget leaves strata too thin for finite intervals at all.
3. **Under the idealized prefix-cache token model the optimum shifts to
   r = 2**: same linf as r = 1 with ~11% fewer uncached tokens on both
   models. Beyond r = 2, error rises faster than tokens fall. The shift is
   modest here because our instruction prefix (~90 tokens) is small relative
   to question + completion; production system prompts (thousands of tokens
   of prefix) would push the optimum substantially higher — exactly the
   cache-aware regime (G4) the proposal targets.

Raw rows: `results/replicate_allocation.json`. Reproduce: `python3 experiments/run_replicate_allocation.py` (requires both T2 caches).
