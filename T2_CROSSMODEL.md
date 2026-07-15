# T2 cross-model robustness: exact instruction attributions

Models: `qwen/qwen3.5-9b` vs `google/gemma-4-26b-a4b-it` — identical segments, questions, and protocol; both attribution vectors are exact (exhaustive grids), so every disagreement below is a real model difference.

## Segment-level agreement

- Spearman rho (Shapley): **0.881**;  Kendall tau: 0.727
- Spearman rho (Owen): 0.867
- Sign agreement: 75% of segments

| segment | phi (qwen3.5-9b) | phi (gemma-4-26b-a4b-it) |
|---|---|---|
| verbose | +0.1142 | +0.0269 |
| brief | +0.0886 | +0.0680 |
| steps | -0.0473 | -0.0356 |
| eliminate | -0.0348 | -0.0273 |
| letter_only | +0.0298 | +0.0376 |
| expert | -0.0175 | -0.0105 |
| trivia | +0.0151 | +0.0171 |
| bias_a | +0.0143 | -0.0001 |
| teacher | -0.0131 | -0.0000 |
| doublecheck | -0.0079 | -0.0304 |
| date | -0.0060 | +0.0137 |
| weather | -0.0022 | +0.0072 |

## Group-level (quotient game) comparison

| group | qwen3.5-9b | gemma-4-26b-a4b-it |
|---|---|---|
| persona | -0.0194 | -0.0444 |
| format | +0.2528 | +0.2222 |
| reasoning | -0.1361 | -0.1833 |
| junk | +0.0361 | +0.0722 |

## Top-5 pairwise interactions

Overlap of top-5 |Mobius| pairs: 0/5 (none)

`qwen/qwen3.5-9b`: eliminate+doublecheck -0.267; verbose+eliminate -0.233; expert+eliminate -0.200; letter_only+steps +0.200; brief+steps +0.200

`google/gemma-4-26b-a4b-it`: doublecheck+weather +0.200; doublecheck+trivia +0.200; doublecheck+date +0.200; steps+eliminate -0.200; teacher+doublecheck +0.167
