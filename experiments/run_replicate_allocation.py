"""Replicate-allocation study (G2): coalitions vs replicates at fixed budget.

Every oracle call is one question-draw + generation. With replicates r, each
CC pair costs 2r calls and its observation noise is
sigma_btw^2 + 2*sigma_win^2/r, while the number of pairs falls as B/(2r).
Prediction (proposal section 4A): measured in CALLS, r = 1 dominates —
fresh pairs average away both between- and within-coalition noise, replicates
only the latter. Measured in UNCACHED PREFILL TOKENS under a prefix cache,
replicates amortize the instruction prefix and the optimum shifts right.

Both T2 grids are replayed from cache, so the whole study is free.

Usage:
    python3 experiments/run_replicate_allocation.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "experiments")
from run_t2_openrouter import build_game, exact_shapley  # noqa: E402

from segshap import cc_shapley  # noqa: E402
from segshap.metrics import ci_coverage, kendall_tau, linf_error, mean_ci_width  # noqa: E402

MODELS = ["qwen/qwen3.5-9b", "google/gemma-4-26b-a4b-it"]
REPLICATE_GRID = [1, 2, 4, 8, 16]
BUDGETS = [3_000, 10_000, 30_000]
SEEDS = 10
DELTA = 0.05
N = 12


def exact_truth(model: str) -> np.ndarray:
    slug = model.split("/")[-1]
    data = json.loads(Path(f"results/t2_{slug}.json").read_text())
    return np.array(list(data["exact_shapley"].values()))


def measure_tokens(model: str) -> dict:
    """Mean prompt/completion tokens from the cached empty- and full-coalition
    responses, to parameterize the idealized prefix-cache token model."""
    game = build_game(model)
    stats = {}
    for label, coalition in (("empty", frozenset()), ("full", frozenset(range(N)))):
        prompts, completions = [], []
        for q in range(len(game.questions)):
            key = game._cache_key(coalition, q, 0)
            usage = json.loads(game._cache_path(key).read_text())["usage"]
            prompts.append(usage["prompt"])
            completions.append(usage["completion"])
        stats[label] = {"prompt": float(np.mean(prompts)), "completion": float(np.mean(completions))}
    return stats


def token_cost_per_pair(r: int, tok: dict, cached: bool) -> float:
    """Expected uncached tokens for one CC pair at replicates r.

    Idealized model: the instruction prefix of a coalition (mean size |S|=6
    -> half the full-vs-empty prompt gap) is cached after the first replicate;
    each replicate always pays the question part and the completion.
    """
    prefix = (tok["full"]["prompt"] - tok["empty"]["prompt"]) / 2.0
    q_part = tok["empty"]["prompt"]
    completion = (tok["empty"]["completion"] + tok["full"]["completion"]) / 2.0
    if cached:
        return 2 * (prefix + r * (q_part + completion))
    return 2 * r * (prefix + q_part + completion)


def main() -> None:
    all_rows = []
    token_stats = {}
    for model in MODELS:
        truth = exact_truth(model)
        token_stats[model] = measure_tokens(model)
        for r in REPLICATE_GRID:
            for budget in BUDGETS:
                for seed in range(SEEDS):
                    g = build_game(model, seed=4000 + seed)
                    res = cc_shapley(g, budget, replicates=r, delta=DELTA, rng=seed)
                    all_rows.append(
                        {
                            "model": model,
                            "replicates": r,
                            "budget": budget,
                            "seed": seed,
                            "pairs": res.calls // (2 * r),
                            "linf": linf_error(res.values, truth),
                            "kendall": kendall_tau(res.values, truth),
                            "all_covered": float(
                                ci_coverage(res.lower, res.upper, truth) == 1.0
                            ),
                            "mean_ci_width": mean_ci_width(res.lower, res.upper),
                        }
                    )
        print(f"{model}: done", flush=True)

    Path("results").mkdir(exist_ok=True)
    Path("results/replicate_allocation.json").write_text(
        json.dumps({"rows": all_rows, "token_stats": token_stats}, indent=1)
    )

    lines = [
        "# Replicate-allocation study: coalitions vs replicates (G2)",
        "",
        "CC-Bernstein on both exact T2 games, replayed from cache. At a fixed "
        "call budget B, replicates r give B/(2r) coalition pairs whose "
        "observations carry within-coalition noise reduced by 1/r — but each "
        "fresh pair also averages the between-coalition component, which "
        f"replicates cannot touch. {SEEDS} seeds per cell, delta = {DELTA}.",
        "",
    ]
    for model in MODELS:
        lines += [
            f"## {model}",
            "",
            "| budget | r | pairs | linf | kendall | P[all cover] | mean CI width |",
            "|---|---|---|---|---|---|---|",
        ]
        for budget in BUDGETS:
            for r in REPLICATE_GRID:
                sel = [
                    row
                    for row in all_rows
                    if row["model"] == model
                    and row["budget"] == budget
                    and row["replicates"] == r
                ]
                lines.append(
                    f"| {budget} | {r} | {sel[0]['pairs']} "
                    f"| {np.mean([x['linf'] for x in sel]):.4f} "
                    f"| {np.mean([x['kendall'] for x in sel]):.3f} "
                    f"| {np.mean([x['all_covered'] for x in sel]):.2f} "
                    f"| {np.mean([x['mean_ci_width'] for x in sel]):.4f} |"
                )
        lines.append("")

    # Token-cost projection at the largest budget.
    lines += [
        "## Idealized prefix-cache token model",
        "",
        "Uncached prefill+decode tokens per CC pair (prefix cached after the "
        "first replicate of a coalition; every replicate pays the question "
        "part and completion). Error is the measured linf at the same "
        "(budget, r); token cost = pairs x tokens/pair.",
        "",
        "| model | r | tokens/pair (cached) | tokens/pair (no cache) | linf @30k calls | Mtokens @30k calls (cached) |",
        "|---|---|---|---|---|---|",
    ]
    for model in MODELS:
        tok = token_stats[model]
        for r in REPLICATE_GRID:
            sel = [
                row
                for row in all_rows
                if row["model"] == model
                and row["budget"] == 30_000
                and row["replicates"] == r
            ]
            pairs = sel[0]["pairs"]
            tc = token_cost_per_pair(r, tok, cached=True)
            tn = token_cost_per_pair(r, tok, cached=False)
            lines.append(
                f"| {model.split('/')[-1]} | {r} | {tc:.0f} | {tn:.0f} "
                f"| {np.mean([x['linf'] for x in sel]):.4f} "
                f"| {pairs * tc / 1e6:.2f} |"
            )
    lines += [
        "",
        "Raw rows: `results/replicate_allocation.json`. "
        "Reproduce: `python3 experiments/run_replicate_allocation.py` "
        "(requires both T2 caches).",
        "",
    ]
    Path("REPLICATE_ALLOCATION.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
