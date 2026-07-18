"""Job 1: PERM-BET on T2 and T4, strictly from the disk cache.

Runs the Perm-Bet estimator (permutation sampling + per-player betting
certificates) on the committed T2 (n=12) and T4 (n=8) games, for both model
families, replaying cached responses only. Any cache miss raises loudly and no
live API call is ever made; we additionally assert game.live_calls == 0.

Configs (match the committed permutation-MC baseline's call accounting):
  T2: n=12, 30k calls, 8 seeds, replicates=3, delta=0.05, vs exact truth.
  T4: n=8,  8k calls, 10 seeds, replicates=3, delta=0.05, vs the exact 2^8 grid.

Writes results/perm_bet.json and prints a Table-1-style summary per model.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "experiments")
import run_t2_openrouter as t2  # noqa: E402
import run_t4_safety as t4  # noqa: E402

from segshap.estimators.perm_bet import perm_bet  # noqa: E402
from segshap.llm import PromptSegmentGame, mmlu_style_render  # noqa: E402
from segshap.metrics import ci_coverage, kendall_tau, linf_error, mean_ci_width  # noqa: E402

DELTA = 0.05
REPLICATES = 3
MODELS = ["qwen/qwen3.5-9b", "google/gemma-4-26b-a4b-it"]


def slug(model: str) -> str:
    return model.split("/")[-1]


def t2_game(model: str, seed: int) -> PromptSegmentGame:
    qs = json.loads(Path(t2.QUESTIONS_FILE).read_text())[: t2.N_QUESTIONS]
    return PromptSegmentGame.openrouter(
        [text for _, text in t2.SEGMENTS], qs, model,
        render=mmlu_style_render, cache_dir=Path(f"cache/t2_{slug(model)}"),
        temperature=0.0, max_tokens=400,
        provider_order=t2.PROVIDER_ORDERS.get(model), offline=True, rng=seed,
    )


def t4_game(model: str, seed: int) -> PromptSegmentGame:
    qs = json.loads(Path(t4.QUESTIONS_FILE).read_text())[: t4.N_QUESTIONS]
    return PromptSegmentGame.openrouter(
        [text for _, text in t4.SEGMENTS], qs, model,
        render=t4.injection_render, utility=t4.resisted_utility,
        cache_dir=Path(f"cache/t4_{slug(model)}"),
        temperature=0.0, max_tokens=400,
        provider_order=t4.PROVIDER_ORDERS.get(model), offline=True, rng=seed,
    )


def exact_truth(path: str) -> np.ndarray:
    return np.array(list(json.load(open(path))["exact_shapley"].values()))


def score(res, truth) -> dict:
    certified_pos = res.lower > 0
    certified_neg = res.upper < 0
    certified = certified_pos | certified_neg
    # A "false" sign certification is one whose certified side disagrees with
    # the true sign (equivalently, a coverage failure on a certified player).
    false_sign = (certified_pos & (truth <= 0)) | (certified_neg & (truth >= 0))
    return {
        "linf": linf_error(res.values, truth),
        "kendall": kendall_tau(res.values, truth),
        "mean_ci_width": mean_ci_width(res.lower, res.upper),
        "all_covered": float(ci_coverage(res.lower, res.upper, truth) == 1.0),
        "n_certified": int(certified.sum()),
        "n_true_signs": int((certified & ~false_sign).sum()),
        "n_false_signs": int(false_sign.sum()),
    }


def run_config(name, builder, truth_path, budget, seeds, game_seed_base):
    truth = exact_truth(truth_path)
    per_seed = []
    live = 0
    for s in range(seeds):
        g = builder(game_seed_base + s)
        res = perm_bet(g, budget_calls=budget, replicates=REPLICATES, delta=DELTA, rng=s)
        assert g.live_calls == 0, f"{name}: live API call made!"
        assert g.calls > 0, f"{name}: no work done (empty replay)"
        live += g.live_calls
        per_seed.append(score(res, truth))
    return {"per_seed": per_seed, "live_calls": live, "budget": budget, "seeds": seeds}


def agg(rows, key):
    v = np.array([r[key] for r in rows], dtype=float)
    return float(v.mean()), float(v.std())


def fmt_row(label, rows):
    lm, ls = agg(rows, "linf")
    wm, ws = agg(rows, "mean_ci_width")
    cov = np.mean([r["all_covered"] for r in rows])
    tc = np.mean([r["n_true_signs"] for r in rows])
    fc = np.mean([r["n_false_signs"] for r in rows])
    return (f"| {label} | {lm:.4f} ± {ls:.4f} | {wm:.4f} ± {ws:.4f} "
            f"| {cov:.2f} | {tc:.1f} | {fc:.1f} |")


def main() -> None:
    out = {"delta": DELTA, "replicates": REPLICATES, "estimator": "perm_bet", "results": {}}
    lines = ["# Job 1: Perm-Bet (cache-replay only, zero live API calls)", ""]

    for tag, builder, truth_tmpl, budget, seeds, base in [
        ("T2", t2_game, "results/t2_{}.json", 30_000, 8, 2000),
        ("T4", t4_game, "results/t4_safety_{}.json", 8_000, 10, 1000),
    ]:
        lines += [
            f"## {tag}: Perm-Bet, {budget:,} calls, {seeds} seeds, delta={DELTA}",
            "",
            "| model | linf (mean±sd) | mean clipped CI width | P[all cover] | true signs cert. | false signs cert. |",
            "|---|---|---|---|---|---|",
        ]
        for model in MODELS:
            r = run_config(f"{tag}/{model}", (lambda s, m=model: builder(m, s)),
                           truth_tmpl.format(slug(model)), budget, seeds, base)
            out["results"][f"{tag}/{slug(model)}"] = r
            lines.append(fmt_row(slug(model), r["per_seed"]))
        lines.append("")

    total_live = sum(r["live_calls"] for r in out["results"].values())
    assert total_live == 0, f"NONZERO LIVE CALLS: {total_live}"
    out["total_live_calls"] = total_live
    lines += [f"Total live API calls across all configs: **{total_live}** "
              f"(hard requirement: 0).", ""]

    Path("results").mkdir(exist_ok=True)
    Path("results/perm_bet.json").write_text(json.dumps(out, indent=1))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
