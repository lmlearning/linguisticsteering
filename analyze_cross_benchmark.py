"""
Cross-benchmark analysis: compares adjective impact rankings between
MMLU and ARC-Challenge results.

Usage:
    python analyze_cross_benchmark.py \
        --arc results/arc_phi3_results.json results/arc_llama8b_results.json \
        --mmlu results/shap_results_replicate.json results/results_llama.json \
        --labels "phi-3" "llama-3-8b" "phi-3" "llama-3-70b"
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def load_aggregated(path: Path) -> pd.DataFrame:
    with open(path) as f:
        data = json.load(f)
    return pd.DataFrame(data['aggregated_results']).T


def gini(x: np.ndarray) -> float:
    x = np.sort(x)
    n = len(x)
    return (2 * np.sum((np.arange(1, n + 1) * x)) - (n + 1) * np.sum(x)) / (n * np.sum(x))


def main(args):
    datasets = {}
    all_labels = args.labels if args.labels else []

    for i, path in enumerate(args.arc):
        label = all_labels[i] if i < len(all_labels) else Path(path).stem
        key = f"arc_{label}"
        datasets[key] = load_aggregated(path)

    for i, path in enumerate(args.mmlu):
        label = all_labels[len(args.arc) + i] if (len(args.arc) + i) < len(all_labels) else Path(path).stem
        key = f"mmlu_{label}"
        datasets[key] = load_aggregated(path)

    common = sorted(set.intersection(*[set(df.index) for df in datasets.values()]))
    names = list(datasets.keys())

    # --- Long-tail distribution ---
    print("=" * 60)
    print("LONG-TAIL DISTRIBUTION")
    print("=" * 60)
    print(f"{'Dataset':<25} {'Top-5':>8} {'Top-10':>8} {'Top-20':>8} {'Gini':>8}")
    for name, df in datasets.items():
        abs_vals = df['mean_abs_shapley']
        total = abs_vals.sum()
        s = abs_vals.sort_values(ascending=False)
        t5 = s.head(5).sum() / total * 100
        t10 = s.head(10).sum() / total * 100
        t20 = s.head(20).sum() / total * 100
        g = gini(abs_vals.values)
        print(f"{name:<25} {t5:>7.1f}% {t10:>7.1f}% {t20:>7.1f}% {g:>8.3f}")

    # --- Spearman rank correlation (absolute impact) ---
    print("\n" + "=" * 60)
    print("SPEARMAN RANK CORRELATION (mean_abs_shapley)")
    print("=" * 60)
    rows = []
    for n1 in names:
        row = {}
        for n2 in names:
            v1 = datasets[n1].loc[common, 'mean_abs_shapley']
            v2 = datasets[n2].loc[common, 'mean_abs_shapley']
            rho, p = spearmanr(v1, v2)
            row[n2] = f"{rho:+.3f}"
        rows.append(row)
    print(pd.DataFrame(rows, index=names).to_string())

    # --- Spearman rank correlation (directional) ---
    print("\n" + "=" * 60)
    print("SPEARMAN RANK CORRELATION (mean_shapley, directional)")
    print("=" * 60)
    rows = []
    for n1 in names:
        row = {}
        for n2 in names:
            v1 = datasets[n1].loc[common, 'mean_shapley']
            v2 = datasets[n2].loc[common, 'mean_shapley']
            rho, p = spearmanr(v1, v2)
            row[n2] = f"{rho:+.3f}"
        rows.append(row)
    print(pd.DataFrame(rows, index=names).to_string())

    # --- Top-10 overlap ---
    print("\n" + "=" * 60)
    print("TOP-10 ADJECTIVE OVERLAP")
    print("=" * 60)
    top10s = {name: set(df.sort_values('mean_abs_shapley', ascending=False).head(10).index)
              for name, df in datasets.items()}
    for i, n1 in enumerate(names):
        for n2 in names[i + 1:]:
            overlap = top10s[n1] & top10s[n2]
            print(f"{n1} vs {n2}: {len(overlap)}/10 — {sorted(overlap) if overlap else 'none'}")

    # --- Key pairwise findings ---
    print("\n" + "=" * 60)
    print("KEY PAIRWISE CORRELATIONS")
    print("=" * 60)
    for n1 in names:
        for n2 in names:
            if n1 < n2:
                v1 = datasets[n1].loc[common, 'mean_abs_shapley']
                v2 = datasets[n2].loc[common, 'mean_abs_shapley']
                rho, p = spearmanr(v1, v2)
                print(f"  {n1} vs {n2}: rho={rho:.3f} (p={p:.4f})")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Cross-benchmark Shapley analysis")
    parser.add_argument("--arc", nargs='+', type=Path, required=True,
                        help="ARC-Challenge result JSON files")
    parser.add_argument("--mmlu", nargs='+', type=Path, required=True,
                        help="MMLU result JSON files")
    parser.add_argument("--labels", nargs='+', type=str, default=None,
                        help="Labels for each file (arc files first, then mmlu)")
    args = parser.parse_args()
    main(args)
