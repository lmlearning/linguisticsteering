"""T3: beyond-enumeration scale — 20 instruction segments on a real LLM.

2^20 ~ 1M coalitions: exhaustive ground truth is impossible, which is the
regime the estimators exist for. The reference is therefore a *certified
gold standard*: a high-budget CC-Bernstein run whose simultaneous betting
CIs bound its own error — every comparison below is stated relative to gold
with gold's uncertainty carried along, not silently ignored.

Phases (all live API calls; the response cache makes same-seed reruns free):
1. Gold: CC-Bernstein, GOLD_BUDGET calls, r=1, delta=0.05.
2. Comparison at COMPARE_BUDGET calls: cc_bernstein, surrogate_cc,
   kernel_shap, leverage_shap, svarm.
3. Hierarchical triage over 5 groups of 4 at tau = TAU.

Usage:
    OPENROUTER_API_KEY=... python3 experiments/run_t3_scale.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from segshap import cc_shapley, hierarchical_owen, kernel_shap, surrogate_shapley
from segshap.estimators.external import leverage_shap, svarm_shapley
from segshap.llm import PromptSegmentGame, mmlu_style_render
from segshap.metrics import kendall_tau, mean_ci_width

MODEL = "qwen/qwen3.5-9b"
PROVIDER_ORDER = ["deepinfra", "siliconflow"]
CACHE_DIR = Path("cache/t3_qwen3.5-9b")
QUESTIONS_FILE = Path("experiments/t1_questions.json")
N_QUESTIONS = 30

SEGMENTS = [
    # group 0: persona
    ("expert", "You are an expert in the subject of the question."),
    ("teacher", "You are a patient teacher who values accuracy above all."),
    ("bias_a", "If you are unsure, choose option A."),
    ("confident", "Answer with confidence and commit to a single choice."),
    # group 1: format
    ("letter_only", "Respond with only the single capital letter of the correct option and nothing else."),
    ("verbose", "Explain your reasoning in full sentences before giving the letter."),
    ("brief", "Keep your answer as brief as possible."),
    ("nopunct", "Do not add any punctuation around your final answer."),
    # group 2: reasoning
    ("steps", "Think carefully step by step before giving your answer."),
    ("eliminate", "First eliminate the clearly wrong options, then choose among the rest."),
    ("doublecheck", "Double-check your reasoning before answering."),
    ("arithmetic", "Re-check any arithmetic before finalizing your answer."),
    # group 3: junk (planted null-ish group)
    ("weather", "Note that the weather today is sunny with a gentle breeze."),
    ("trivia", "Fun fact: bananas are botanically berries."),
    ("date", "For reference, today is a Tuesday."),
    ("sky", "Note that the sky is generally blue during the day."),
    # group 4: style
    ("polite", "Maintain a polite and professional tone."),
    ("formal", "Use formal academic language."),
    ("short", "Use short sentences."),
    ("noemoji", "Do not use emojis."),
]
GROUPS = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15], [16, 17, 18, 19]]
GROUP_NAMES = ["persona", "format", "reasoning", "junk", "style"]

GOLD_BUDGET = 120_000
COMPARE_BUDGET = 30_000
TRIAGE_BUDGET = 60_000
TAU = 0.10
DELTA = 0.05
COST_CEILING_USD = 30.0


def build_game(seed: int = 0) -> PromptSegmentGame:
    questions = json.loads(QUESTIONS_FILE.read_text())[:N_QUESTIONS]
    return PromptSegmentGame.openrouter(
        [text for _, text in SEGMENTS],
        questions,
        MODEL,
        render=mmlu_style_render,
        cache_dir=CACHE_DIR,
        temperature=0.0,
        max_tokens=400,
        max_concurrency=96,
        provider_order=PROVIDER_ORDER,
        rng=seed,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("results/t3_scale_qwen3.5-9b.json"))
    args = parser.parse_args()
    n = len(SEGMENTS)
    names = [name for name, _ in SEGMENTS]
    total_cost = 0.0

    # ---- Phase 1: certified gold standard.
    t0 = time.time()
    gold_game = build_game(seed=7)
    gold = cc_shapley(
        gold_game, GOLD_BUDGET, replicates=1, delta=DELTA, batch_pairs=64, rng=7
    )
    total_cost += gold_game.total_cost
    print(
        f"gold: {gold.calls} calls, ${gold_game.total_cost:.2f}, "
        f"{(time.time()-t0)/60:.0f} min, mean CI width {mean_ci_width(gold.lower, gold.upper):.4f}",
        flush=True,
    )
    if total_cost > COST_CEILING_USD:
        raise RuntimeError(f"cost ceiling exceeded: ${total_cost:.2f}")

    # ---- Phase 2: estimator comparison vs gold.
    estimators = {
        "cc_bernstein": lambda g, b: cc_shapley(
            g, b, replicates=1, delta=DELTA, batch_pairs=64, rng=11
        ),
        "surrogate_cc": lambda g, b: surrogate_shapley(
            g, b, order=2, replicates=1, delta=DELTA, rng=11
        ),
        "kernel_shap": lambda g, b: kernel_shap(g, b, replicates=1, rng=11),
        "leverage_shap": lambda g, b: leverage_shap(g, b, replicates=1, rng=11),
        "svarm": lambda g, b: svarm_shapley(g, b, replicates=1, rng=11),
    }
    rows = []
    for est_name, fn in estimators.items():
        t1 = time.time()
        g = build_game(seed=100)
        res = fn(g, COMPARE_BUDGET)
        total_cost += g.total_cost
        has_ci = bool(np.all(np.isfinite(res.halfwidths)))
        # "Consistent with gold": |est - gold| <= est halfwidth + gold halfwidth
        # for every segment (only checkable for estimators with certificates).
        consistent = (
            bool(
                np.all(
                    np.abs(res.values - gold.values)
                    <= res.halfwidths + gold.halfwidths + 1e-12
                )
            )
            if has_ci
            else None
        )
        rows.append(
            {
                "estimator": est_name,
                "budget": COMPARE_BUDGET,
                "calls": res.calls,
                "linf_vs_gold": float(np.max(np.abs(res.values - gold.values))),
                "kendall_vs_gold": kendall_tau(res.values, gold.values),
                "mean_ci_width": mean_ci_width(res.lower, res.upper) if has_ci else None,
                "consistent_with_gold": consistent,
                "cost_usd": g.total_cost,
                "minutes": (time.time() - t1) / 60.0,
            }
        )
        print(f"{est_name}: done (${g.total_cost:.2f}, {(time.time()-t1)/60:.0f} min)", flush=True)
        if total_cost > COST_CEILING_USD:
            raise RuntimeError(f"cost ceiling exceeded: ${total_cost:.2f}")

    # ---- Phase 3: hierarchical triage at n=20.
    t2 = time.time()
    tri_game = build_game(seed=200)
    triage = hierarchical_owen(
        tri_game, GROUPS, budget_calls=TRIAGE_BUDGET, tau=TAU, replicates=1,
        delta=DELTA, rng=200,
    )
    total_cost += tri_game.total_cost
    print(f"triage: done (${tri_game.total_cost:.2f}, {(time.time()-t2)/60:.0f} min)", flush=True)

    args.out.parent.mkdir(exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "model": MODEL,
                "n_segments": n,
                "segments": dict(SEGMENTS),
                "groups": {gn: [names[p] for p in g] for gn, g in zip(GROUP_NAMES, GROUPS)},
                "gold": {
                    "budget": GOLD_BUDGET,
                    "values": dict(zip(names, gold.values.tolist())),
                    "lower": dict(zip(names, gold.lower.tolist())),
                    "upper": dict(zip(names, gold.upper.tolist())),
                },
                "comparison": rows,
                "triage": {
                    "tau": TAU,
                    "budget": TRIAGE_BUDGET,
                    "group_values": dict(zip(GROUP_NAMES, triage.group_values.tolist())),
                    "group_halfwidths": dict(zip(GROUP_NAMES, triage.group_halfwidths.tolist())),
                    "certified_negligible": [GROUP_NAMES[g] for g in triage.certified_negligible],
                    "expanded": [GROUP_NAMES[g] for g in triage.expanded],
                    "member_values": {names[p]: v for p, v in triage.member_values.items()},
                    "member_halfwidths": {names[p]: v for p, v in triage.member_halfwidths.items()},
                },
                "total_cost_usd": total_cost,
            },
            indent=1,
        )
    )

    # ---- Report.
    order = np.argsort(-gold.values)
    lines = [
        "# T3 results: 20 segments — beyond exhaustive enumeration",
        "",
        f"Model: `{MODEL}`. n = {n} segments (2^20 ~ 1.05M coalitions: exact "
        "enumeration impossible). Reference = certified gold standard: "
        f"CC-Bernstein at {GOLD_BUDGET:,} calls whose simultaneous 95% betting "
        f"CIs (mean width {mean_ci_width(gold.lower, gold.upper):.4f}) bound its own error.",
        "",
        "## Certified gold attribution (top and bottom segments)",
        "",
        "| segment | group | gold phi | 95% CI |",
        "|---|---|---|---|",
    ]
    group_of = {p: GROUP_NAMES[gi] for gi, g in enumerate(GROUPS) for p in g}
    for i in list(order[:6]) + list(order[-4:]):
        lines.append(
            f"| {names[i]} | {group_of[i]} | {gold.values[i]:+.4f} "
            f"| [{gold.lower[i]:+.3f}, {gold.upper[i]:+.3f}] |"
        )
    lines += [
        "",
        f"## Estimator comparison at {COMPARE_BUDGET:,} calls (vs gold)",
        "",
        "| estimator | linf vs gold | kendall vs gold | mean CI width | consistent with gold |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        wid = f"{r['mean_ci_width']:.4f}" if r["mean_ci_width"] is not None else "n/a"
        cons = {True: "yes", False: "NO", None: "n/a"}[r["consistent_with_gold"]]
        lines.append(
            f"| {r['estimator']} | {r['linf_vs_gold']:.4f} "
            f"| {r['kendall_vs_gold']:.3f} | {wid} | {cons} |"
        )
    lines += [
        "",
        f"## Hierarchical triage (5 groups of 4, tau = {TAU}, {TRIAGE_BUDGET:,} calls)",
        "",
        "| group | value | halfwidth | verdict |",
        "|---|---|---|---|",
    ]
    for gi, gn in enumerate(GROUP_NAMES):
        verdict = "certified negligible" if gi in triage.certified_negligible else "expanded"
        lines.append(
            f"| {gn} | {triage.group_values[gi]:+.4f} "
            f"| {triage.group_halfwidths[gi]:.4f} | {verdict} |"
        )
    lines += [
        "",
        f"Total spend for T3: ${total_cost:.2f}. "
        "Reproduce: `OPENROUTER_API_KEY=... python3 experiments/run_t3_scale.py` "
        "(same seeds replay from the response cache).",
        "",
    ]
    Path("T3_RESULTS.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
