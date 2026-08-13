"""Regenerate the article figures (fig_main.pdf, fig_t4.pdf) from results JSON.

Reads only the committed results/ files -- no oracle calls. Produces:
  figures/fig_main.pdf : (a,b) synthetic T0 under binary vs low-noise oracles,
                         (c) T2/Qwen point error vs budget, (d) T2 CI widths.
  figures/fig_t4.pdf   : certified injection-resistance audits, Qwen and Gemma.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("figures")
OUT.mkdir(exist_ok=True)
STYLE = {
    "cc_bernstein": ("CC-Bet", "#1b9e77", "-", "o"),
    "surrogate_cc": ("Surrogate", "#66a61e", "-", "s"),
    "permutation_mc": ("Permutation MC", "#7570b3", "--", "^"),
    "kernel_shap": ("KernelSHAP", "#e7298a", "--", "v"),
    "svarm": ("SVARM", "#d95f02", ":", "D"),
    "leverage_shap": ("Leverage SHAP", "#a6761d", ":", "P"),
}


def mean_by_budget(rows, est, key):
    budgets = sorted(set(r["budget"] for r in rows))
    xs, ys = [], []
    for b in budgets:
        vals = [r[key] for r in rows if r["budget"] == b and r["estimator"] == est]
        if vals:
            xs.append(b)
            ys.append(np.mean(vals))
    return xs, ys


def panel_linf(ax, rows, title):
    for est, (label, color, ls, mk) in STYLE.items():
        xs, ys = mean_by_budget(rows, est, "linf")
        if xs:
            ax.plot(xs, ys, ls, color=color, marker=mk, ms=4, label=label, lw=1.6)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("oracle calls"); ax.set_ylabel(r"$\ell_\infty$ error")
    ax.set_title(title, fontsize=9); ax.grid(alpha=0.3, which="both")


def fig_main():
    s = json.load(open("results/synthetic_results.json"))["estimation"]
    t2 = json.load(open("results/t2_qwen3.5-9b.json"))["estimator_rows"]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.0))
    panel_linf(axes[0], s["bernoulli"]["rows"], "(a) T0 binary (Bernoulli) oracle")
    panel_linf(axes[1], s["gauss(0.05)"]["rows"], "(b) T0 low-noise Gaussian oracle")
    panel_linf(axes[2], t2, "(c) T2 Qwen3.5-9B (exact truth)")

    ax = axes[3]
    for est, lab, col in [("cc_bernstein", "CC-Bet (betting)", "#1b9e77"),
                          ("permutation_mc", "Perm. MC (Hoeffding)", "#7570b3")]:
        xs, ys = mean_by_budget(t2, est, "mean_ci_width")
        ys = [min(w, 2.0) for w in ys]  # clip to feasible width 2
        ax.plot(xs, ys, "-o", color=col, ms=4, label=lab, lw=1.6)
    ax.axhline(2.0, color="gray", ls=":", lw=1, label="vacuous (width 2)")
    ax.set_xscale("log"); ax.set_xlabel("oracle calls")
    ax.set_ylabel("mean 95% CI width (clipped)")
    ax.set_title("(d) T2 certificate width", fontsize=9); ax.grid(alpha=0.3)
    ax.legend(fontsize=6, loc="upper right")
    axes[0].legend(fontsize=6, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / "fig_main.pdf"); plt.close(fig)


def t4_panel(ax, data, title):
    names = list(data["exact_shapley"].keys())
    exact = np.array([data["exact_shapley"][k] for k in names])
    lo = np.array([data["showcase_cc8000"]["lower"][k] for k in names])
    hi = np.array([data["showcase_cc8000"]["upper"][k] for k in names])
    mid = (lo + hi) / 2
    y = np.arange(len(names))
    for i in range(len(names)):
        cert = lo[i] > 0
        col = "#1b9e77" if cert else "#999999"
        ax.plot([lo[i], hi[i]], [y[i], y[i]], color=col, lw=3, alpha=0.8, solid_capstyle="round")
    ax.plot(exact, y, "x", color="black", ms=7, mew=2, label="exact", zorder=5)
    ax.axvline(0, color="red", ls="--", lw=1, alpha=0.6)
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel(r"resistance Shapley $\phi$"); ax.set_title(title, fontsize=9)
    ax.grid(alpha=0.3, axis="x")


def fig_t4():
    q = json.load(open("results/t4_safety_qwen3.5-9b.json"))
    g = json.load(open("results/t4_safety_gemma-4-26b-a4b-it.json"))
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.2))
    t4_panel(axes[0], q, "(a) Qwen3.5-9B: distributed")
    t4_panel(axes[1], g, "(b) Gemma-4-26B: concentrated")
    axes[0].legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "fig_t4.pdf"); plt.close(fig)


if __name__ == "__main__":
    fig_main()
    fig_t4()
    print(f"wrote {OUT/'fig_main.pdf'} and {OUT/'fig_t4.pdf'}")
