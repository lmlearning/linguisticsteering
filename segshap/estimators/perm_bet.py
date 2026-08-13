"""Perm-Bet: permutation-sampling Shapley with per-player betting certificates.

This is the permutation-Monte-Carlo estimator (Castro et al.), instrumented
with an anytime-valid betting confidence sequence per player instead of a
Hoeffding bound. Uniform random permutations are sampled; along each
permutation, player i is credited its marginal contribution
v(pre u {i}) - v(pre). Each player keeps one hedged betting confidence
sequence (segshap.bounds.betting_interval) at level delta/n, so the n intervals
hold jointly with probability >= 1 - delta by a union bound. Marginal
contributions of a game with values in [0,1] lie in [-1,1]; the final intervals
are clipped to [-1,1] (which can only tighten a valid interval).

Call accounting matches segshap.estimators.baselines.permutation_mc exactly:
each permutation costs (n+1)*replicates oracle calls (one empty coalition plus
n growing prefixes), and permutations are drawn while the next chain fits in the
remaining budget.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from segshap.bounds import betting_interval, eb_halfwidth
from segshap.estimators.cc_bernstein import ShapleyResult
from segshap.games import BudgetExceeded, NoisyGame


def perm_bet(
    game: NoisyGame,
    budget_calls: int,
    replicates: int = 1,
    delta: float = 0.05,
    ci_method: str = "betting",
    rng: Optional[np.random.Generator | int] = None,
) -> ShapleyResult:
    """Permutation-sampling Shapley with simultaneous certificates.

    ``ci_method="betting"`` (default) uses per-player hedged betting sequences;
    ``"eb"`` uses the time-uniform empirical-Bernstein comparator on the same
    marginal-contribution samples (for the decomposition x interval ablation).
    Both are simultaneous over players at level delta via a union bound.
    """
    n = game.n
    rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    samples: list[list[float]] = [[] for _ in range(n)]
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
                samples[int(i)].append(cur - prev)
                prev = cur
        except BudgetExceeded:
            break

    values = np.array([np.mean(s) if s else 0.0 for s in samples])
    lower = np.full(n, -1.0)
    upper = np.full(n, 1.0)
    for i in range(n):
        if ci_method == "betting":
            lo, hi = betting_interval(samples[i], -1.0, 1.0, delta_player)
        else:  # empirical-Bernstein comparator on the same samples
            xs = np.asarray(samples[i], dtype=float)
            var = float(np.var(xs, ddof=1)) if xs.size >= 2 else np.inf
            hw = eb_halfwidth(xs.size, var, 2.0, delta_player)
            lo, hi = values[i] - hw, values[i] + hw
        lower[i] = max(-1.0, lo)
        upper[i] = min(1.0, hi)

    return ShapleyResult(
        values=values,
        halfwidths=(upper - lower) / 2.0,
        calls=game.calls - start_calls,
        meta={
            "method": "perm_bet",
            "delta": delta,
            "replicates": replicates,
            "permutations": len(samples[0]) if n else 0,
        },
        lower_bounds=lower,
        upper_bounds=upper,
    )
