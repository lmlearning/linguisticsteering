"""Classical baselines: permutation Monte Carlo and KernelSHAP.

These are the reference points the proposal's experiments compare against.
Permutation MC (Castro et al. 2009) reuses one permutation chain for all n
players and carries anytime-valid Hoeffding intervals. KernelSHAP (Lundberg &
Lee 2016, with the paired-sampling improvement of Covert & Lee 2021) returns
point estimates only — it has no finite-sample certificate, which is exactly
the gap the proposal targets.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.special import binom

from segshap.bounds import StratumStats, hoeffding_halfwidth
from segshap.estimators.cc_bernstein import ShapleyResult
from segshap.games import BudgetExceeded, NoisyGame


def permutation_mc(
    game: NoisyGame,
    budget_calls: int,
    replicates: int = 1,
    delta: float = 0.05,
    rng: Optional[np.random.Generator | int] = None,
) -> ShapleyResult:
    """Castro-style permutation sampling with per-player Hoeffding CIs."""
    n = game.n
    rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    stats = [StratumStats() for _ in range(n)]
    start_calls = game.calls
    chain_cost = (n + 1) * replicates
    delta_player = delta / n

    while game.calls - start_calls + chain_cost <= budget_calls:
        perm = rng.permutation(n)
        try:
            prev = float(np.mean(game.evaluate(frozenset(), replicates)))
            coalition: set = set()
            for i in perm:
                coalition.add(int(i))
                cur = float(np.mean(game.evaluate(frozenset(coalition), replicates)))
                stats[int(i)].add(cur - prev)
                prev = cur
        except BudgetExceeded:
            break

    values = np.array([s.mean for s in stats])
    halfwidths = np.array(
        [
            hoeffding_halfwidth(s.count, 2.0 * game.range, delta_player)
            for s in stats
        ]
    )
    return ShapleyResult(
        values=values,
        halfwidths=halfwidths,
        calls=game.calls - start_calls,
        meta={"method": "permutation_mc", "delta": delta, "replicates": replicates},
    )


def kernel_shap(
    game: NoisyGame,
    budget_calls: int,
    replicates: int = 1,
    paired: bool = True,
    rng: Optional[np.random.Generator | int] = None,
) -> ShapleyResult:
    """KernelSHAP: Shapley-kernel-weighted least squares with the efficiency
    constraint eliminated in closed form. Point estimates only (no CIs)."""
    n = game.n
    rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    start_calls = game.calls

    v_empty = float(np.mean(game.evaluate(frozenset(), replicates)))
    v_full = float(np.mean(game.evaluate(frozenset(range(n)), replicates)))

    # Shapley kernel distribution over coalition sizes 1..n-1.
    sizes = np.arange(1, n)
    size_probs = (n - 1) / (sizes * (n - sizes))
    size_probs = size_probs / size_probs.sum()

    rows = []
    ys = []
    while game.calls - start_calls + replicates * (2 if paired else 1) <= budget_calls:
        s = int(rng.choice(sizes, p=size_probs))
        members = frozenset(rng.choice(n, size=s, replace=False).tolist())
        batch = [members, frozenset(range(n)) - members] if paired else [members]
        try:
            for coal in batch:
                z = np.zeros(n)
                z[list(coal)] = 1.0
                rows.append(z)
                ys.append(float(np.mean(game.evaluate(coal, replicates))))
        except BudgetExceeded:
            break

    if not rows:
        return ShapleyResult(
            values=np.full(n, np.nan),
            halfwidths=np.full(n, np.inf),
            calls=game.calls - start_calls,
            meta={"method": "kernel_shap", "error": "budget too small"},
        )

    z_mat = np.array(rows)
    y = np.array(ys) - v_empty
    total = v_full - v_empty
    # Eliminate the efficiency constraint sum(phi) = total by substituting
    # phi_n = total - sum(phi_1..n-1).
    a = z_mat[:, :-1] - z_mat[:, [-1]]
    b = y - z_mat[:, -1] * total
    coef, *_ = np.linalg.lstsq(a, b, rcond=None)
    values = np.append(coef, total - coef.sum())
    return ShapleyResult(
        values=values,
        halfwidths=np.full(n, np.nan),
        calls=game.calls - start_calls,
        meta={"method": "kernel_shap", "paired": paired, "replicates": replicates},
    )
