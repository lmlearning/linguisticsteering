# segshap — certified segment-level Shapley attribution for LLM prompts

How much does each *segment* of a prompt — an instruction, a few-shot example,
a persona block, a retrieved document — actually contribute to end-task
quality, including its interactions with the other segments? `segshap`
answers that with Shapley/Owen values estimated from as few LLM calls as
possible, and every number it reports carries a simultaneous, anytime-valid
(epsilon, delta) confidence certificate.

The research programme, the survey of the 2021–2026 state of the art, and the
three-approach design are laid out in [FOLLOW_UP_PROPOSAL.md](FOLLOW_UP_PROPOSAL.md).
First empirical results on the synthetic testbed are in [RESULTS.md](RESULTS.md).

## The three approaches

| Module | Approach | One-liner |
|---|---|---|
| `segshap.estimators.cc_bernstein` | **A — CC-Bernstein** | Paired complementary coalitions: each evaluated pair (v(S), v(N\\S)) updates *every* segment's estimate; anytime empirical-Bernstein CIs; adaptive size allocation. Assumption-free reference. |
| `segshap.estimators.tree_elim` | **B — TreeSHAP-Elim** | Hierarchical Owen values with certified pruning: groups whose CI fits inside [-tau, tau] are certified negligible and never expanded; survivors are refined member-by-member. The prompt-triage tool. |
| `segshap.estimators.surrogate` | **C — SurroSHAP-Cert** | Fit a sparse k-additive Mobius surrogate, read its Shapley values in closed form, then certify by running Approach A on the *residual* game (phi(v) = phi(g) + phi(v-g), exactly). Big savings when structure exists; degrades gracefully into A when it doesn't. |

Baselines (`segshap.estimators.baselines`): Castro-style permutation Monte
Carlo with Hoeffding CIs, and KernelSHAP with paired sampling.

## Quick start

```bash
pip install -e .
pytest tests/                              # correctness vs exact ground truth
python3 experiments/run_synthetic.py       # regenerates RESULTS.md
```

```python
import numpy as np
from segshap import random_sparse_game, cc_shapley

game = random_sparse_game(n=12, n_terms=20, rng=0, noise="bernoulli")
res = cc_shapley(game, budget_calls=10_000, delta=0.05)
print(res.values)            # Shapley estimates for all 12 segments
print(res.lower, res.upper)  # simultaneous 95% confidence bounds
print(game.exact_shapley)    # ground truth (synthetic games only)
```

Against a real model, `segshap.llm.PromptSegmentGame` wraps any
OpenAI-compatible endpoint: a coalition renders its segments in canonical
template order, one oracle call is one eval-question draw + generation, all
responses are disk-cached so nothing is ever paid for twice, and token usage
(cached vs. uncached prefix) is accounted per call.

```python
from segshap.llm import PromptSegmentGame, mmlu_style_render
from segshap import cc_shapley

game = PromptSegmentGame(
    segments=["Answer concisely.", "Think step by step.", "You are an expert."],
    questions=questions,           # dicts with question/choices/answer
    model="gpt-4o-mini",
    render=mmlu_style_render,
    cache_dir="cache/",
)
res = cc_shapley(game, budget_calls=2_000, replicates=4, delta=0.05)
```

## Origins

This repository previously hosted a study of *linguistic steering* — Shapley
attribution of single adjectives injected into MMLU/ARC prompts across
several API models (`estimate_importance.py` and companion scripts, kept for
reference). That study inspired the present project: same core question
(which prompt components matter?), rebuilt at segment granularity with
provable error bounds and an order-of-magnitude smaller query budget.
