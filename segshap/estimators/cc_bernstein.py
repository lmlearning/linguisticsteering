"""Approach A — CC-Bernstein: certified Shapley from paired complementary coalitions.

Identity (Zhang et al., SIGMOD 2023): for every player i,

    phi_i = (1/n) * sum_{s=1..n} E[ CC(S) | i in S, |S| = s ],
    CC(S) = v(S) - v(N \\ S).

One evaluated pair (S, N\\S) therefore yields a valid sample for *every*
player: members of S get CC(S) in their size-|S| stratum, non-members get
-CC(S) in their size-(n-|S|) stratum. This is the maximal statistical reuse
available without structural assumptions, and it doubles as paired
(antithetic) sampling, a large variance reduction when the game is close to
additive.

Certificates: each (player, size) stratum keeps an anytime-valid
empirical-Bernstein interval (see segshap.bounds); the per-player interval is
their equally weighted sum, and all n intervals hold simultaneously with
probability >= 1 - delta. Sizes are allocated adaptively toward the strata
whose intervals dominate the current uncertainty (Burgess & Chapman-style
variance adaptivity); adaptivity cannot invalidate the intervals because they
are anytime-valid and samples within a stratum stay i.i.d.

Two-level noise: ``replicates`` controls the inner (within-coalition) sample
size r; each observed CC is a difference of two r-replicate means, so within-
coalition noise enters the stratum variance as sigma^2/r and is handled by
the same empirical bound — no separate analysis needed at run time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from segshap.bounds import StratifiedEstimate
from segshap.games import BudgetExceeded, NoisyGame


@dataclass
class ShapleyResult:
    values: np.ndarray
    halfwidths: np.ndarray
    calls: int
    meta: dict = field(default_factory=dict)
    # Explicit interval endpoints (betting CIs are not centered on the point
    # estimate). When absent, intervals are values +/- halfwidths.
    lower_bounds: Optional[np.ndarray] = None
    upper_bounds: Optional[np.ndarray] = None

    @property
    def lower(self) -> np.ndarray:
        if self.lower_bounds is not None:
            return self.lower_bounds
        return self.values - self.halfwidths

    @property
    def upper(self) -> np.ndarray:
        if self.upper_bounds is not None:
            return self.upper_bounds
        return self.values + self.halfwidths


def cc_shapley(
    game: NoisyGame,
    budget_calls: int,
    replicates: int = 1,
    delta: float = 0.05,
    target_eps: Optional[float] = None,
    min_pairs_per_size: int = 3,
    adaptive: bool = True,
    ci_method: str = "betting",
    batch_pairs: Optional[int] = None,
    rng: Optional[np.random.Generator | int] = None,
) -> ShapleyResult:
    """Estimate all Shapley values of ``game`` with simultaneous (eps, delta) CIs.

    Stops when ``budget_calls`` oracle calls (charged to this invocation) are
    spent, or earlier if every player's CI half-width falls below
    ``target_eps``.
    """
    n = game.n
    rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    # One CC observation is a difference of two means of values in the game's
    # range, so its range is 2 * game.range.
    est = StratifiedEstimate(
        n_players=n, n_strata=n, value_range=2.0 * game.range, delta=delta
    )
    players = np.arange(n)
    start_calls = game.calls
    pairs_per_size = np.zeros(n + 1, dtype=int)  # index by coalition size s

    def spent() -> int:
        return game.calls - start_calls

    def draw_pairs(sizes_batch: list) -> None:
        """Draw one pair per requested size, evaluated as one concurrent batch."""
        pairs = []
        for s in sizes_batch:
            members = rng.choice(players, size=s, replace=False)
            member_set = frozenset(int(i) for i in members)
            pairs.append((s, member_set, frozenset(range(n)) - member_set))
        coalitions = [c for _, m, comp in pairs for c in (m, comp)]
        values = game.evaluate_many(coalitions, replicates)
        for idx, (s, member_set, complement) in enumerate(pairs):
            v_s = float(np.mean(values[2 * idx]))
            v_c = float(np.mean(values[2 * idx + 1]))
            cc = v_s - v_c
            for i in member_set:
                est.add(i, s - 1, cc)
            for j in complement:
                est.add(j, (n - s) - 1, -cc)
            pairs_per_size[s] += 1

    pair_cost = 2 * replicates

    # Phase 1: seed every size so no stratum CI stays infinite.
    for _ in range(min_pairs_per_size):
        room = max(0, (budget_calls - spent()) // pair_cost)
        sizes_batch = list(range(1, n + 1))[:room]
        if not sizes_batch:
            break
        try:
            draw_pairs(sizes_batch)
        except BudgetExceeded:
            break

    # Phase 2: adaptive allocation across sizes, re-scored once per batch to
    # keep overhead negligible relative to oracle cost. Adaptivity across
    # batches never invalidates the CIs (they are anytime-valid and samples
    # within a stratum stay i.i.d.).
    batch_pairs = batch_pairs or max(8, n)
    while spent() + pair_cost <= budget_calls:
        if target_eps is not None and np.all(est.halfwidths() <= target_eps):
            break
        if adaptive:
            # Benefit proxy for drawing one more size-s pair: the CI mass it
            # would touch, discounted by how well-sampled those strata are.
            scores = np.zeros(n + 1)
            for s in range(1, n + 1):
                for i in range(n):
                    for str_idx, prob in ((s - 1, s / n), (n - s - 1, (n - s) / n)):
                        if prob <= 0:
                            continue
                        st = est.stats[i][str_idx]
                        w = est.stratum_halfwidth(i, str_idx)
                        if math.isinf(w):
                            w = 2.0 * game.range
                        scores[s] += prob * w / math.sqrt(st.count + 1)
            s_next = int(np.argmax(scores))
        else:
            s_next = int(rng.integers(1, n + 1))
        room = (budget_calls - spent()) // pair_cost
        k = int(min(batch_pairs, room))
        if k <= 0:
            break
        try:
            draw_pairs([s_next] * k)
        except BudgetExceeded:
            break

    lower, upper = est.intervals(method=ci_method)
    return ShapleyResult(
        values=est.values(),
        halfwidths=(upper - lower) / 2.0,
        calls=spent(),
        meta={
            "method": "cc_bernstein",
            "delta": delta,
            "replicates": replicates,
            "ci_method": ci_method,
            "pairs_per_size": pairs_per_size.tolist(),
        },
        lower_bounds=lower,
        upper_bounds=upper,
    )
