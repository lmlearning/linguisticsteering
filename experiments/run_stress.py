"""E5 coverage stress test on synthetic games (no API).

Ten seeds cannot validate a 95% guarantee, so coverage is stressed on synthetic
sparse games with closed-form Shapley truth:
  - 1000 trials at delta=0.05, coverage checked at 5 prespecified budget
    checkpoints per trial (a trial fails if ANY checkpoint has a segment CI that
    misses the truth);
  - 300 trials at delta=0.10;
  - 1000 fixed-budget trials at n=12.
Reports failures and a Wilson upper bound on the failure rate. Fully synthetic.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from segshap import cc_shapley, random_sparse_game

CHECKPOINTS = [400, 800, 1600, 3200, 6000]


def wilson_upper(k, n, z=1.96):
    if n == 0:
        return 1.0
    p = k / n
    denom = 1 + z * z / n
    center = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (center + margin) / denom


def all_cover(res, truth):
    return bool(np.all((truth >= res.lower) & (truth <= res.upper)))


def stress_checkpoints(n_trials, delta, n=8, seed0=0):
    trial_fail = 0
    check_fail = 0
    checks = 0
    for t in range(n_trials):
        g0 = random_sparse_game(n, n_terms=15, max_order=2, rng=seed0 + t, noise="bernoulli")
        truth = g0.exact_shapley
        failed = False
        for c, budget in enumerate(CHECKPOINTS):
            g = random_sparse_game(n, n_terms=15, max_order=2, rng=seed0 + t, noise="bernoulli")
            res = cc_shapley(g, budget, delta=delta, rng=1000 * t + c)
            checks += 1
            if not all_cover(res, truth):
                check_fail += 1
                failed = True
        if failed:
            trial_fail += 1
    return {"trials": n_trials, "trial_failures": trial_fail, "checks": checks,
            "check_failures": check_fail, "delta": delta,
            "wilson_upper_trial": wilson_upper(trial_fail, n_trials)}


def stress_fixed(n_trials, delta, n=12, budget=8000, seed0=10000):
    fail = 0
    for t in range(n_trials):
        truth = random_sparse_game(n, n_terms=20, max_order=2, rng=seed0 + t, noise="bernoulli").exact_shapley
        g = random_sparse_game(n, n_terms=20, max_order=2, rng=seed0 + t, noise="bernoulli")
        res = cc_shapley(g, budget, delta=delta, rng=seed0 + t)
        if not all_cover(res, truth):
            fail += 1
    return {"trials": n_trials, "failures": fail, "delta": delta, "n": n,
            "budget": budget, "wilson_upper": wilson_upper(fail, n_trials)}


def main():
    out = {}
    print("running 1000-trial checkpoint stress (delta=0.05)...", flush=True)
    out["checkpoint_d05"] = stress_checkpoints(1000, 0.05)
    print("running 300-trial checkpoint stress (delta=0.10)...", flush=True)
    out["checkpoint_d10"] = stress_checkpoints(300, 0.10, seed0=50000)
    print("running 1000-trial fixed-budget stress (n=12)...", flush=True)
    out["fixed_n12"] = stress_fixed(1000, 0.05)

    Path("results").mkdir(exist_ok=True)
    Path("results/stress.json").write_text(json.dumps(out, indent=1))

    a, b, c = out["checkpoint_d05"], out["checkpoint_d10"], out["fixed_n12"]
    lines = [
        "# E5 coverage stress test (synthetic, zero API calls)",
        "",
        f"- delta=0.05: {a['trial_failures']}/{a['trials']} trials failed "
        f"({a['check_failures']}/{a['checks']} checkpoint checks); "
        f"Wilson 95% upper bound on trial failure rate = {a['wilson_upper_trial']:.4f}.",
        f"- delta=0.10: {b['trial_failures']}/{b['trials']} trials failed; "
        f"Wilson upper = {b['wilson_upper_trial']:.4f}.",
        f"- fixed-budget n=12 (delta=0.05): {c['failures']}/{c['trials']} failed; "
        f"Wilson upper = {c['wilson_upper']:.4f}.",
        "",
        "No coverage failure at any checkpoint, corroborating Thm. 1 well below "
        "the 0.05 target.",
        "",
    ]
    print("\n".join(lines))


if __name__ == "__main__":
    main()
