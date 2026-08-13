"""T1 removal effects: exact Delta_i vs Shapley phi_i (from the committed grid).

Separates the two estimands on T1's exact 2^8 game. For each single segment i,
the removal effect Delta_i = v(N) - v(N\\{i}) is compared to its Shapley value
phi_i. They correlate but do not interchange (article Sec. Alignment): Kendall
~0.69 with a couple of near-zero sign flips, and the sabotage clause at
Delta = -0.50 vs phi = -0.43. Uses only the committed exact grid in
results/t1_qwen3.5-9b.json -- no oracle calls.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau

from segshap import exact_shapley

DATA = "results/t1_qwen3.5-9b.json"


def main():
    d = json.load(open(DATA))
    names = list(d["exact_shapley"].keys())
    n = len(names)
    grid = d["grid"]["values"]  # {"i,j,...": v}

    def v(s: frozenset) -> float:
        return grid[",".join(str(i) for i in sorted(s))]

    full = frozenset(range(n))
    phi = exact_shapley(v, n)
    delta = np.array([v(full) - v(full - {i}) for i in range(n)])

    tau, _ = kendalltau(delta, phi)
    sign_flips = int(np.sum(np.sign(delta) != np.sign(phi)))
    sab = names.index("sabotage")

    out = {
        "kendall_delta_vs_phi": float(tau),
        "sign_flips": sign_flips,
        "per_segment": {
            names[i]: {"phi": float(phi[i]), "delta": float(delta[i])} for i in range(n)
        },
        "sabotage": {"phi": float(phi[sab]), "delta": float(delta[sab])},
    }
    Path("results").mkdir(exist_ok=True)
    Path("results/t1_removal.json").write_text(json.dumps(out, indent=1))

    lines = [
        "# T1 removal effects: Delta_i (intervention) vs phi_i (average)",
        "",
        f"Kendall(Delta, phi) = {tau:.2f}; sign flips = {sign_flips} (near zero).",
        "",
        "| segment | phi (average) | Delta (removal effect) |",
        "|---|---|---|",
    ]
    for i in np.argsort(phi):
        lines.append(f"| {names[i]} | {phi[i]:+.4f} | {delta[i]:+.4f} |")
    lines += [
        "",
        f"Sabotage clause: phi = {phi[sab]:+.3f}, Delta = {delta[sab]:+.3f} "
        "-- averages and interventions correlate but do not interchange, which "
        "is why removals are gated on Delta rather than on Shapley/Owen values.",
        "",
    ]
    print("\n".join(lines))


if __name__ == "__main__":
    main()
