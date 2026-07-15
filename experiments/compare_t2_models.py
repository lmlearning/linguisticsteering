"""Cross-model robustness comparison of exact T2 attributions.

Reads two T2 result files (exact ground truth per model, from the exhaustive
grids) and reports how stable instruction attributions are across model
families: rank correlations, sign agreement, group-level comparison, and
overlap of the strongest pairwise interactions. This is metric 5 of the
shared experimental design — computed on *exact* values, so any disagreement
is a property of the models, not estimation noise.

Usage:
    python3 experiments/compare_t2_models.py results/t2_A.json results/t2_B.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau, spearmanr


def main() -> None:
    path_a, path_b = Path(sys.argv[1]), Path(sys.argv[2])
    a, b = json.loads(path_a.read_text()), json.loads(path_b.read_text())
    model_a, model_b = a["model"], b["model"]
    names = list(a["exact_shapley"].keys())
    assert names == list(b["exact_shapley"].keys()), "segment sets differ"

    phi_a = np.array([a["exact_shapley"][k] for k in names])
    phi_b = np.array([b["exact_shapley"][k] for k in names])
    owen_a = np.array([a["exact_owen"][k] for k in names])
    owen_b = np.array([b["exact_owen"][k] for k in names])

    rho, _ = spearmanr(phi_a, phi_b)
    tau, _ = kendalltau(phi_a, phi_b)
    rho_o, _ = spearmanr(owen_a, owen_b)
    sign_agree = float(np.mean(np.sign(phi_a) == np.sign(phi_b)))

    pairs_a = {tuple(sorted(p["pair"])): p["mobius"] for p in a["top_pair_mobius"]}
    pairs_b = {tuple(sorted(p["pair"])): p["mobius"] for p in b["top_pair_mobius"]}
    pair_overlap = set(pairs_a) & set(pairs_b)

    lines = [
        "# T2 cross-model robustness: exact instruction attributions",
        "",
        f"Models: `{model_a}` vs `{model_b}` — identical segments, questions, "
        "and protocol; both attribution vectors are exact (exhaustive grids), "
        "so every disagreement below is a real model difference.",
        "",
        "## Segment-level agreement",
        "",
        f"- Spearman rho (Shapley): **{rho:.3f}**;  Kendall tau: {tau:.3f}",
        f"- Spearman rho (Owen): {rho_o:.3f}",
        f"- Sign agreement: {sign_agree:.0%} of segments",
        "",
        "| segment | " + f"phi ({model_a.split('/')[-1]}) | phi ({model_b.split('/')[-1]}) |",
        "|---|---|---|",
    ]
    for i in np.argsort(-np.abs(phi_a)):
        lines.append(f"| {names[i]} | {phi_a[i]:+.4f} | {phi_b[i]:+.4f} |")
    ga = a["exact_group_shapley"]
    gb = b["exact_group_shapley"]
    lines += [
        "",
        "## Group-level (quotient game) comparison",
        "",
        "| group | " + f"{model_a.split('/')[-1]} | {model_b.split('/')[-1]} |",
        "|---|---|---|",
    ]
    for g in ga:
        lines.append(f"| {g} | {ga[g]:+.4f} | {gb[g]:+.4f} |")
    lines += [
        "",
        "## Top-5 pairwise interactions",
        "",
        f"Overlap of top-5 |Mobius| pairs: {len(pair_overlap)}/5 "
        f"({', '.join('+'.join(p) for p in sorted(pair_overlap)) or 'none'})",
        "",
        f"`{model_a}`: "
        + "; ".join(f"{'+'.join(p['pair'])} {p['mobius']:+.3f}" for p in a["top_pair_mobius"]),
        "",
        f"`{model_b}`: "
        + "; ".join(f"{'+'.join(p['pair'])} {p['mobius']:+.3f}" for p in b["top_pair_mobius"]),
        "",
    ]
    Path("T2_CROSSMODEL.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
