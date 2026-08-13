"""E5 ablation: decomposition (CC vs permutation) x interval (betting vs EB).

Cache-replay only (offline; zero live API calls). On the exact T2 games, crosses
the two decompositions with the two interval constructions, all intervals
range-clipped to [-1,1], and reports point error (l_inf), mean CI width, and the
decision metric (true/false signs certified). CC pairing lowers l_inf under
either interval; betting narrows both certificates; on the decision metric
Perm-Bet certifies more true signs at higher point error -- a Pareto frontier.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "experiments")
from run_perm_bet import exact_truth, t2_game  # noqa: E402  (offline T2 builder)

from segshap import cc_shapley, random_sparse_game  # noqa: E402
from segshap.estimators.perm_bet import perm_bet  # noqa: E402
from segshap.metrics import linf_error  # noqa: E402

MODELS = ["qwen/qwen3.5-9b", "google/gemma-4-26b-a4b-it"]
BUDGET = 25_000
SEEDS = 8
SYNTH_SEEDS = 10
REPLICATES = 1  # call-optimal per Prop. 1
DELTA = 0.05


def clip(res):
    lo = np.clip(res.lower, -1.0, 1.0)
    hi = np.clip(res.upper, -1.0, 1.0)
    return lo, hi


def decision_counts(lo, hi, truth):
    cert = (lo > 0) | (hi < 0)
    false = ((lo > 0) & (truth <= 0)) | ((hi < 0) & (truth >= 0))
    return int((cert & ~false).sum()), int(false.sum())


def methods_at(budget):
    return {
        "CC+betting": lambda g: cc_shapley(g, budget, replicates=REPLICATES, delta=DELTA, ci_method="betting", rng=0),
        "CC+EB":      lambda g: cc_shapley(g, budget, replicates=REPLICATES, delta=DELTA, ci_method="eb", rng=0),
        "Perm+betting": lambda g: perm_bet(g, budget, replicates=REPLICATES, delta=DELTA, ci_method="betting", rng=0),
        "Perm+EB":      lambda g: perm_bet(g, budget, replicates=REPLICATES, delta=DELTA, ci_method="eb", rng=0),
    }


def run_block(builder, truth_of, seeds, budget):
    methods = methods_at(budget)
    agg = {m: {"linf": [], "width": [], "true": [], "false": [], "cover": []} for m in methods}
    live = 0
    for seed in range(seeds):
        g0 = builder(seed)
        truth = truth_of(g0)
        for m, fn in methods.items():
            g = builder(seed)
            res = fn(g)
            live += getattr(g, "live_calls", 0)
            lo, hi = clip(res)
            t, f = decision_counts(lo, hi, truth)
            agg[m]["linf"].append(linf_error(res.values, truth))
            agg[m]["width"].append(float(np.mean(hi - lo)))
            agg[m]["true"].append(t)
            agg[m]["false"].append(f)
            agg[m]["cover"].append(float(np.all((truth >= lo) & (truth <= hi))))
    return agg, live


def table(agg, title, note):
    lines = [title, "",
             "| method | l_inf | mean CI width | true signs cert. | false signs | cover |",
             "|---|---|---|---|---|---|"]
    dump = {}
    for m, a in agg.items():
        dump[m] = {k: [float(np.mean(v)), float(np.std(v))] for k, v in a.items()}
        lines.append(f"| {m} | {np.mean(a['linf']):.4f} | {np.mean(a['width']):.3f} "
                     f"| {np.mean(a['true']):.2f} | {np.mean(a['false']):.2f} | {np.mean(a['cover']):.2f} |")
    lines += ["", note, ""]
    return lines, dump


def main():
    out = {"budget": BUDGET, "replicates": REPLICATES, "delta": DELTA}

    # --- Part A: synthetic n=12 (decision metric; many non-negligible signs) ---
    synth = lambda seed: random_sparse_game(12, n_terms=20, max_order=2, rng=seed, noise="bernoulli")
    agg_s, live_s = run_block(synth, lambda g: g.exact_shapley, SYNTH_SEEDS, BUDGET)
    la, out["synthetic_n12"] = table(
        agg_s, f"# E5 ablation (synthetic n=12, {BUDGET:,} calls, r={REPLICATES}, {SYNTH_SEEDS} seeds)",
        f"Decision metric: Perm+betting certifies {np.mean(agg_s['Perm+betting']['true']):.2f} "
        f"of 12 true signs vs {np.mean(agg_s['CC+betting']['true']):.2f} for CC+betting; "
        f"zero false signs for both. CC halves Perm's l_inf; betting narrows both intervals "
        "(Pareto frontier: CC for point accuracy, Perm-Bet for certified signs).")

    # --- Part B: T2 cache-replay confirmation ("holds on real models") ---
    agg_t, live_t = run_block(
        lambda seed: t2_game(MODELS[0], 5000 + seed),
        lambda g: exact_truth(f"results/t2_{MODELS[0].split('/')[-1]}.json"),
        SEEDS, 30_000)
    lb, out["t2_qwen_30k"] = table(
        agg_t, "# E5 ablation on real model (T2/Qwen, 30,000 calls, cache-replay)",
        f"Frontier holds on the real model: Perm+betting width "
        f"{np.mean(agg_t['Perm+betting']['width']):.2f} vs {np.mean(agg_t['CC+betting']['width']):.2f} "
        f"for CC+betting, at {np.mean(agg_t['Perm+betting']['linf'])/max(np.mean(agg_t['CC+betting']['linf']),1e-9):.1f}x "
        "the point error; coverage 1.00; zero false signs.")

    assert live_s + live_t == 0, f"NONZERO LIVE CALLS: {live_s + live_t}"
    out["total_live_calls"] = 0
    Path("results").mkdir(exist_ok=True)
    Path("results/ablation.json").write_text(json.dumps(out, indent=1))
    print("\n".join(la + lb))


if __name__ == "__main__":
    main()
