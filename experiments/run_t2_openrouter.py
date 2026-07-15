"""T2: instruction attribution at n = 12 with hierarchy, on a real LLM.

Twelve instruction segments in four semantic groups of three, with planted
structure the methods should recover:

- Group "junk" (weather / trivia / date): three irrelevant remarks ->
  near-null group, the target for TreeSHAP-Elim's certified pruning.
- A planted CONFLICT pair across segments: "letter_only" (respond with only
  the letter) vs "verbose" (explain in full sentences before the letter) ->
  should appear as a large negative pairwise Mobius interaction.
- A planted mild-bias instruction: "bias_a" (if unsure choose option A).

2^12 = 4,096 coalitions x 30 MMLU questions, evaluated exhaustively at
temperature 0: exact Shapley values, exact Owen values, and exact pairwise
Mobius interactions on a real model — the ground truth against which the
estimators and the hierarchical triage are scored (replayed from the disk
cache at zero marginal cost).

Usage:
    OPENROUTER_API_KEY=... python3 experiments/run_t2_openrouter.py
    (resumable: rerun continues from whatever the cache already holds)
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from pathlib import Path

import numpy as np

from segshap import cc_shapley, exact_shapley, hierarchical_owen, kernel_shap, permutation_mc, surrogate_shapley
from segshap.llm import PromptSegmentGame, mmlu_style_render
from segshap.metrics import ci_coverage, kendall_tau, linf_error, mean_ci_width

DEFAULT_MODEL = "qwen/qwen3.5-9b"
# Provider preference per model slug (None -> OpenRouter auto-routing).
PROVIDER_ORDERS = {"qwen/qwen3.5-9b": ["deepinfra", "siliconflow"]}
QUESTIONS_FILE = Path("experiments/t1_questions.json")
N_QUESTIONS = 30

SEGMENTS = [
    # group 0: persona
    ("expert", "You are an expert in the subject of the question."),
    ("teacher", "You are a patient teacher who values accuracy above all."),
    ("bias_a", "If you are unsure, choose option A."),
    # group 1: format (letter_only vs verbose is a planted conflict)
    ("letter_only", "Respond with only the single capital letter of the correct option and nothing else."),
    ("verbose", "Explain your reasoning in full sentences before giving the letter."),
    ("brief", "Keep your answer as brief as possible."),
    # group 2: reasoning
    ("steps", "Think carefully step by step before giving your answer."),
    ("eliminate", "First eliminate the clearly wrong options, then choose among the rest."),
    ("doublecheck", "Double-check your reasoning before answering."),
    # group 3: junk (planted null group)
    ("weather", "Note that the weather today is sunny with a gentle breeze."),
    ("trivia", "Fun fact: bananas are botanically berries."),
    ("date", "For reference, today is a Tuesday."),
]
GROUPS = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]
GROUP_NAMES = ["persona", "format", "reasoning", "junk"]

BUDGETS = [3_000, 10_000, 30_000]
DELTA = 0.05
REPLICATES = 3
TAU = 0.08


def model_slug(model: str) -> str:
    return model.split("/")[-1]


def build_game(model: str = DEFAULT_MODEL, seed: int = 0) -> PromptSegmentGame:
    questions = json.loads(QUESTIONS_FILE.read_text())[:N_QUESTIONS]
    return PromptSegmentGame.openrouter(
        [text for _, text in SEGMENTS],
        questions,
        model,
        render=mmlu_style_render,
        cache_dir=Path(f"cache/t2_{model_slug(model)}"),
        temperature=0.0,
        max_tokens=400,
        max_concurrency=64,
        provider_order=PROVIDER_ORDERS.get(model),
        rng=seed,
    )


def exact_owen(v: dict, n: int, groups) -> np.ndarray:
    """Exact Owen values by direct enumeration of the two-level formula."""
    m = len(groups)
    owen = np.zeros(n)
    for g_idx, group in enumerate(groups):
        k = len(group)
        others = [j for j in range(m) if j != g_idx]
        for i in group:
            rest = [p for p in group if p != i]
            total = 0.0
            for r_size in range(len(others) + 1):
                for r_groups in itertools.combinations(others, r_size):
                    w_r = (
                        math.factorial(r_size)
                        * math.factorial(m - r_size - 1)
                        / math.factorial(m)
                    )
                    u = frozenset(p for j in r_groups for p in groups[j])
                    for t_size in range(len(rest) + 1):
                        for t in itertools.combinations(rest, t_size):
                            w_t = (
                                math.factorial(t_size)
                                * math.factorial(k - t_size - 1)
                                / math.factorial(k)
                            )
                            base = u | frozenset(t)
                            total += w_r * w_t * (v[base | {i}] - v[base])
            owen[i] = total
    return owen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    model = args.model
    slug = model_slug(model)
    out_path = args.out or Path(f"results/t2_{slug}.json")

    n = len(SEGMENTS)
    names = [name for name, _ in SEGMENTS]
    all_coalitions = [
        frozenset(c)
        for r in range(n + 1)
        for c in itertools.combinations(range(n), r)
    ]

    # ---- Phase 1: exhaustive 4096 x 30 grid.
    game = build_game(model)
    t0 = time.time()
    matrix = game.prime_grid(all_coalitions)
    grid_seconds = time.time() - t0
    v_exact = {s: float(matrix[i].mean()) for i, s in enumerate(all_coalitions)}
    grid_cost, grid_calls = game.total_cost, game.calls
    print(f"grid: {grid_calls} new calls, ${grid_cost:.2f}, {grid_seconds:.0f}s")

    truth = exact_shapley(lambda s: v_exact[s], n)
    owen_truth = exact_owen(v_exact, n, GROUPS)
    # Exact quotient-game (group-level) Shapley.
    v_groups = {}
    for r in range(len(GROUPS) + 1):
        for gs in itertools.combinations(range(len(GROUPS)), r):
            union = frozenset(p for j in gs for p in GROUPS[j])
            v_groups[frozenset(gs)] = v_exact[union]
    group_truth = exact_shapley(lambda s: v_groups[s], len(GROUPS))
    # Exact pairwise Mobius interactions.
    pair_mobius = {}
    for i, j in itertools.combinations(range(n), 2):
        pair_mobius[(i, j)] = (
            v_exact[frozenset({i, j})]
            - v_exact[frozenset({i})]
            - v_exact[frozenset({j})]
            + v_exact[frozenset()]
        )
    top_pairs = sorted(pair_mobius.items(), key=lambda kv: -abs(kv[1]))[:5]

    print("exact Shapley:")
    for name, phi in sorted(zip(names, truth), key=lambda x: -x[1]):
        print(f"  {name:12s} {phi:+.4f}")
    print("exact group Shapley:", dict(zip(GROUP_NAMES, np.round(group_truth, 4))))

    # ---- Phase 2: flat-estimator comparison, replayed from cache.
    estimators = {
        "permutation_mc": lambda g, b, seed: permutation_mc(g, b, replicates=REPLICATES, delta=DELTA, rng=seed),
        "kernel_shap": lambda g, b, seed: kernel_shap(g, b, replicates=REPLICATES, rng=seed),
        "cc_bernstein": lambda g, b, seed: cc_shapley(g, b, replicates=REPLICATES, delta=DELTA, rng=seed),
        "surrogate_cc": lambda g, b, seed: surrogate_shapley(g, b, order=2, replicates=REPLICATES, delta=DELTA, rng=seed),
    }
    rows = []
    for seed in range(8):
        for budget in BUDGETS:
            for est_name, fn in estimators.items():
                g = build_game(model, seed=2000 + seed)
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
                        "mean_ci_width": mean_ci_width(res.lower, res.upper) if has_ci else None,
                        "cost": g.total_cost,
                    }
                )
        print(f"estimator seeds done: {seed + 1}/8", flush=True)

    # ---- Phase 3: hierarchical triage on the real game (replayed).
    hier_rows = []
    for seed in range(8):
        g = build_game(model, seed=3000 + seed)
        res = hierarchical_owen(
            g, GROUPS, budget_calls=200_000, tau=TAU, replicates=2, delta=DELTA, rng=seed
        )
        member_err = {
            names[p]: abs(res.member_values[p] - owen_truth[p])
            <= res.member_halfwidths[p]
            for p in res.member_values
        }
        hier_rows.append(
            {
                "seed": seed,
                "certified": [GROUP_NAMES[g_] for g_ in res.certified_negligible],
                "expanded": [GROUP_NAMES[g_] for g_ in res.expanded],
                "group_values": dict(zip(GROUP_NAMES, res.group_values.tolist())),
                "group_halfwidths": dict(zip(GROUP_NAMES, res.group_halfwidths.tolist())),
                "member_cis_cover_owen": all(member_err.values()),
                "n_members_refined": len(res.member_values),
            }
        )
        print(f"hierarchical seeds done: {seed + 1}/8", flush=True)

    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "model": model,
                "segments": dict(SEGMENTS),
                "groups": {gn: [names[p] for p in g] for gn, g in zip(GROUP_NAMES, GROUPS)},
                "n_questions": N_QUESTIONS,
                "grid": {"calls": grid_calls, "cost_usd": grid_cost, "seconds": grid_seconds},
                "exact_shapley": dict(zip(names, truth.tolist())),
                "exact_owen": dict(zip(names, owen_truth.tolist())),
                "exact_group_shapley": dict(zip(GROUP_NAMES, group_truth.tolist())),
                "top_pair_mobius": [
                    {"pair": [names[i], names[j]], "mobius": val}
                    for (i, j), val in top_pairs
                ],
                "estimator_rows": rows,
                "hierarchical_rows": hier_rows,
                "tau": TAU,
            },
            indent=1,
        )
    )

    # ---- Report.
    lines = [
        "# T2 results: 12-instruction attribution with hierarchy on a real LLM",
        "",
        f"Model: `{model}` via OpenRouter (reasoning disabled, temperature 0). "
        f"12 instruction segments in 4 groups x {N_QUESTIONS} MMLU questions; exhaustive "
        f"2^12 = 4,096-coalition grid ({grid_calls} calls, ${grid_cost:.2f}, "
        f"{grid_seconds/60:.0f} min) gives exact Shapley, Owen, and interaction ground truth.",
        "",
        f"Accuracy with no segments: {v_exact[frozenset()]:.3f}; with all twelve: "
        f"{v_exact[frozenset(range(n))]:.3f}.",
        "",
        "## Exact segment Shapley values (ground truth)",
        "",
        "| segment | group | exact phi | exact Owen |",
        "|---|---|---|---|",
    ]
    group_of = {p: GROUP_NAMES[gi] for gi, g in enumerate(GROUPS) for p in g}
    for i in np.argsort(-truth):
        lines.append(
            f"| {names[i]} | {group_of[i]} | {truth[i]:+.4f} | {owen_truth[i]:+.4f} |"
        )
    lines += [
        "",
        "## Planted structure recovered from the exact grid",
        "",
        f"Group Shapley (quotient game): "
        + ", ".join(f"{gn} {gv:+.4f}" for gn, gv in zip(GROUP_NAMES, group_truth)),
        "",
        "Top-5 pairwise Mobius interactions:",
        "",
        "| pair | interaction |",
        "|---|---|",
    ]
    for (i, j), val in top_pairs:
        lines.append(f"| {names[i]} + {names[j]} | {val:+.4f} |")
    lines += [
        "",
        "## Flat-estimator comparison (8 seeds, replayed from cache)",
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
    cert_junk = np.mean([1.0 if "junk" in r["certified"] else 0.0 for r in hier_rows])
    false_elim = sum(
        1 for r in hier_rows for gn in r["certified"]
        if abs(dict(zip(GROUP_NAMES, group_truth))[gn]) > TAU
    )
    member_cov = np.mean([r["member_cis_cover_owen"] for r in hier_rows])
    lines += [
        "",
        f"## Hierarchical triage (TreeSHAP-Elim, tau = {TAU}, 200k replayed calls, 8 seeds)",
        "",
        f"- 'junk' group certified negligible in {cert_junk:.0%} of runs",
        f"- false eliminations (true |group value| > tau certified anyway): {false_elim}",
        f"- member Owen CIs cover exact Owen values: {member_cov:.0%} of runs",
        "",
        f"Total new spend for T2: ${grid_cost:.2f} (everything after the grid replays the cache).",
        "",
        "Reproduce: `OPENROUTER_API_KEY=... python3 experiments/run_t2_openrouter.py`.",
        "",
    ]
    Path(f"T2_RESULTS_{slug}.md" if model != DEFAULT_MODEL else "T2_RESULTS.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
