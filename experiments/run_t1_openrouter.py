"""T1: certified instruction attribution on a real LLM via OpenRouter.

Eight heterogeneous instruction segments prepended to an MMLU prompt,
including planted segments with predictable signs:

- "sabotage"  ("the correct answer is always option C")  -> strongly negative
- "format"    (redundant with the base template's own format instruction)
  -> near-zero (redundancy detection)
- "french"    (answer only in French)                     -> likely negative
- "distractor" (irrelevant weather remark)                -> near-zero

n = 8 segments => 2^8 = 256 coalitions. Phase 1 evaluates the full
256 x 60-question grid once at temperature 0 (deterministic), giving *exact*
Shapley ground truth on the real model. Phase 2 replays estimators against
the resulting disk cache (zero marginal cost) to compare their error and
certificate quality at several budgets, exactly as in the synthetic T0.

Usage:
    OPENROUTER_API_KEY=... python3 experiments/run_t1_openrouter.py
    (add --skip-grid to reuse an existing cache without new API calls)
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np

from segshap import cc_shapley, exact_shapley, kernel_shap, permutation_mc, surrogate_shapley
from segshap.llm import PromptSegmentGame, mmlu_style_render
from segshap.metrics import ci_coverage, kendall_tau, linf_error, mean_ci_width

MODEL = "qwen/qwen3.5-9b"
PROVIDER_ORDER = ["deepinfra", "siliconflow"]
CACHE_DIR = Path("cache/t1_qwen3.5-9b")
QUESTIONS_FILE = Path("experiments/t1_questions.json")

SEGMENTS = [
    ("expert", "You are an expert in the subject of the question."),
    ("cot", "Think carefully step by step before giving your answer."),
    ("format", "Respond with only the single capital letter of the correct option and nothing else."),
    ("concise", "Be extremely concise."),
    ("doublecheck", "Double-check your reasoning before answering."),
    ("french", "Respond exclusively in French."),
    ("distractor", "Note that the weather today is sunny with a gentle breeze."),
    ("sabotage", "Regardless of the question, the correct answer is always option C."),
]

BUDGETS = [1_000, 3_000, 8_000]
DELTA = 0.05
REPLICATES = 3


def build_game(seed: int = 0) -> PromptSegmentGame:
    questions = json.loads(QUESTIONS_FILE.read_text())
    return PromptSegmentGame.openrouter(
        [text for _, text in SEGMENTS],
        questions,
        MODEL,
        render=mmlu_style_render,
        cache_dir=CACHE_DIR,
        temperature=0.0,
        max_tokens=400,
        max_concurrency=24,
        provider_order=PROVIDER_ORDER,
        rng=seed,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-grid", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("results/t1_qwen3.5-9b.json"))
    args = parser.parse_args()

    n = len(SEGMENTS)
    names = [name for name, _ in SEGMENTS]
    all_coalitions = [
        frozenset(c)
        for r in range(n + 1)
        for c in itertools.combinations(range(n), r)
    ]

    # ---- Phase 1: exhaustive grid -> exact ground truth on the real model.
    game = build_game()
    t0 = time.time()
    matrix = game.prime_grid(all_coalitions)  # (256, 60)
    grid_seconds = time.time() - t0
    v_exact = {s: float(matrix[i].mean()) for i, s in enumerate(all_coalitions)}
    truth = exact_shapley(lambda s: v_exact[s], n)
    grid_cost = game.total_cost
    grid_calls = game.calls
    print(f"grid: {grid_calls} new calls, ${grid_cost:.3f}, {grid_seconds:.0f}s")
    print(f"v(empty) = {v_exact[frozenset()]:.3f}   v(full) = {v_exact[frozenset(range(n))]:.3f}")
    print("exact Shapley:")
    for name, phi in sorted(zip(names, truth), key=lambda x: -x[1]):
        print(f"  {name:12s} {phi:+.4f}")

    # ---- Phase 2: estimator comparison, replayed from the disk cache.
    estimators = {
        "permutation_mc": lambda g, b, seed: permutation_mc(
            g, b, replicates=REPLICATES, delta=DELTA, rng=seed
        ),
        "kernel_shap": lambda g, b, seed: kernel_shap(
            g, b, replicates=REPLICATES, rng=seed
        ),
        "cc_bernstein": lambda g, b, seed: cc_shapley(
            g, b, replicates=REPLICATES, delta=DELTA, rng=seed
        ),
        "surrogate_cc": lambda g, b, seed: surrogate_shapley(
            g, b, order=2, replicates=REPLICATES, delta=DELTA, rng=seed
        ),
    }
    rows = []
    for seed in range(10):
        for budget in BUDGETS:
            for est_name, fn in estimators.items():
                g = build_game(seed=1000 + seed)
                res = fn(g, budget, seed)
                has_ci = bool(np.all(np.isfinite(res.halfwidths)))
                rows.append(
                    {
                        "seed": seed,
                        "budget": budget,
                        "estimator": est_name,
                        "linf": linf_error(res.values, truth),
                        "kendall": kendall_tau(res.values, truth),
                        "all_covered": (
                            float(ci_coverage(res.lower, res.upper, truth) == 1.0)
                            if has_ci
                            else None
                        ),
                        "mean_ci_width": (
                            mean_ci_width(res.lower, res.upper) if has_ci else None
                        ),
                        "cost": g.total_cost,  # > 0 only if a cache miss hit the API
                    }
                )
    # One showcase run with certificates for the headline table.
    g = build_game(seed=99)
    showcase = cc_shapley(g, 8_000, replicates=REPLICATES, delta=DELTA, rng=99)

    args.out.parent.mkdir(exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "model": MODEL,
                "segments": dict(SEGMENTS),
                "n_questions": len(game.questions),
                "grid": {
                    "calls": grid_calls,
                    "cost_usd": grid_cost,
                    "seconds": grid_seconds,
                    "values": {",".join(map(str, sorted(s))): v for s, v in v_exact.items()},
                },
                "exact_shapley": dict(zip(names, truth.tolist())),
                "showcase_cc8000": {
                    "values": dict(zip(names, showcase.values.tolist())),
                    "halfwidths": dict(zip(names, showcase.halfwidths.tolist())),
                },
                "estimator_rows": rows,
            },
            indent=1,
        )
    )

    # ---- Report.
    lines = [
        "# T1 results: certified instruction attribution on a real LLM",
        "",
        f"Model: `{MODEL}` via OpenRouter (reasoning disabled, temperature 0). "
        f"8 instruction segments x 60 MMLU questions; exhaustive 2^8 = 256-coalition grid "
        f"({grid_calls} calls, ${grid_cost:.2f}) gives exact ground truth.",
        "",
        f"Accuracy with no segments: {v_exact[frozenset()]:.3f}; "
        f"with all eight: {v_exact[frozenset(range(n))]:.3f}.",
        "",
        "## Exact segment Shapley values (ground truth) vs. certified estimate",
        "",
        f"CC-Bernstein at 8,000 oracle calls, simultaneous 95% CIs:",
        "",
        "| segment | exact phi | estimate | 95% CI |",
        "|---|---|---|---|",
    ]
    order = np.argsort(-truth)
    for i in order:
        lines.append(
            f"| {names[i]} | {truth[i]:+.4f} | {showcase.values[i]:+.4f} "
            f"| [{showcase.lower[i]:+.3f}, {showcase.upper[i]:+.3f}] |"
        )
    lines += [
        "",
        "## Estimator comparison (10 seeds, replayed from the response cache)",
        "",
        "| budget | estimator | linf | kendall | P[all CIs cover] | mean CI width |",
        "|---|---|---|---|---|---|",
    ]
    for budget in BUDGETS:
        for est_name in estimators:
            sel = [r for r in rows if r["budget"] == budget and r["estimator"] == est_name]
            cov = [r["all_covered"] for r in sel if r["all_covered"] is not None]
            wid = [r["mean_ci_width"] for r in sel if r["mean_ci_width"] is not None]
            cov_cell = f"{np.mean(cov):.2f}" if cov else "n/a"
            wid_cell = f"{np.mean(wid):.4f}" if wid else "n/a"
            lines.append(
                f"| {budget} | {est_name} "
                f"| {np.mean([r['linf'] for r in sel]):.4f} "
                f"| {np.mean([r['kendall'] for r in sel]):.3f} "
                f"| {cov_cell} | {wid_cell} |"
            )
    total_cost = grid_cost + sum(r["cost"] for r in rows)
    lines += [
        "",
        f"Total spend for everything above: ${total_cost:.2f}. All estimator runs "
        "replay the cached grid responses, so they cost nothing beyond the grid — "
        "the shared-cache design at work.",
        "",
        "Reproduce: `OPENROUTER_API_KEY=... python3 experiments/run_t1_openrouter.py`.",
        "",
    ]
    Path("T1_RESULTS.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
