"""SOTA baseline wrappers: SVARM (shapiq) and Leverage SHAP.

These are the strongest published baselines in the query-efficiency lineage:

- SVARM (Kolpaczki et al., AAAI 2024): stratified coalition sampling where
  every evaluated coalition updates every player's estimate; unbiased with
  non-asymptotic error bounds. We wrap the authors' implementation from the
  ``shapiq`` package.
- Leverage SHAP (Musco & Witter, ICLR 2025): leverage-score sampling for the
  Shapley regression characterization with the first non-asymptotic
  guarantees in the KernelSHAP family. Implemented here from the paper's
  algorithm (see ``leverage_shap``).

Both consume our budgeted noisy-oracle games: `budget_calls` counts oracle
calls, and each coalition evaluation spends `replicates` calls (mean of r
question-draws), matching how the segshap estimators are budgeted.
"""

from __future__ import annotations

import itertools
from typing import Optional

import numpy as np
from scipy.special import binom

from segshap.estimators.cc_bernstein import ShapleyResult
from segshap.games import NoisyGame


def _game_fn(game: NoisyGame, replicates: int):
    def fn(coalitions: np.ndarray) -> np.ndarray:
        out = np.empty(coalitions.shape[0])
        for row_idx, row in enumerate(coalitions):
            members = frozenset(int(i) for i in np.where(row)[0])
            out[row_idx] = float(np.mean(game.evaluate(members, replicates)))
        return out

    return fn


def svarm_shapley(
    game: NoisyGame,
    budget_calls: int,
    replicates: int = 1,
    rng: Optional[int] = None,
) -> ShapleyResult:
    """SVARM via the authors' shapiq implementation. Point estimates only."""
    from shapiq import SVARM

    start_calls = game.calls
    seed = int(rng) if rng is not None else None
    # shapiq's border trick enumerates all 2^n coalitions once when the
    # evaluation budget allows and then stops, which would silently strand
    # most of a noisy game's call budget. Scale replicates up so the full
    # budget is spent averaging noise in that regime (fair full-budget SVARM).
    n_coalitions = 2 ** game.n
    if budget_calls // replicates >= n_coalitions:
        replicates = max(replicates, budget_calls // n_coalitions)
    approx = SVARM(n=game.n, random_state=seed)
    iv = approx.approximate(
        budget=budget_calls // replicates, game=_game_fn(game, replicates)
    )
    values = np.array([iv[(i,)] for i in range(game.n)])
    return ShapleyResult(
        values=values,
        halfwidths=np.full(game.n, np.nan),
        calls=game.calls - start_calls,
        meta={"method": "svarm", "replicates": replicates},
    )


def leverage_shap(
    game: NoisyGame,
    budget_calls: int,
    replicates: int = 1,
    rng: Optional[np.random.Generator | int] = None,
) -> ShapleyResult:
    """Leverage SHAP (Musco & Witter, ICLR 2025), deterministic-budget variant.

    The Shapley values solve the constrained weighted regression of Lemma 2.1
    (rows indexed by coalitions 0 < |S| < n with kernel weight
    w(s) = 1/(C(n,s) s (n-s))); the leverage score of every size-s row is
    exactly 1/C(n,s) (Lemma 3.2), so leverage sampling puts equal total mass
    on each size. Following the authors' reference implementation we use the
    deterministic per-size counts k_s = min(c, C(n,s)) with c solving
    sum_s k_s = m, paired sampling (each coalition drawn with its
    complement), inverse-inclusion-probability reweighting w(s)/p_s, and the
    projected min-norm solve of Lemma 3.1 with the efficiency constraint
    added back as (v(N) - v(empty))/n.
    """
    n = game.n
    rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    start_calls = game.calls

    # Deterministic leverage sampling saturates at full enumeration (2^n
    # rows); past that point spend the remaining budget on replicates so the
    # comparison against other estimators is full-budget fair on noisy games.
    if budget_calls // replicates >= 2**n:
        replicates = max(replicates, budget_calls // 2**n)
    m = budget_calls // replicates - 2
    sizes = np.arange(1, n)
    counts_per_size = np.array([binom(n, s) for s in sizes])
    m = int(min(m, counts_per_size.sum()))
    if m < n:
        raise ValueError("budget too small for leverage_shap (need >= n+2 evals)")

    # Solve sum_s min(c, C(n,s)) = m for c (piecewise linear, monotone).
    lo_c, hi_c = 0.0, float(counts_per_size.max())
    for _ in range(64):
        mid = (lo_c + hi_c) / 2.0
        if np.minimum(mid, counts_per_size).sum() < m:
            lo_c = mid
        else:
            hi_c = mid
    c = hi_c

    def sample_size_s(s: int, k: int) -> list:
        all_subsets = list(itertools.combinations(range(n), s))
        idx = rng.choice(len(all_subsets), size=k, replace=False)
        return [frozenset(all_subsets[i]) for i in idx]

    rows: list = []
    row_sizes: list = []
    inclusion: dict = {}
    for s in range(1, n // 2 + 1):
        comp = n - s
        c_ns = binom(n, s)
        if s < comp:
            k = int(round(min(c, c_ns)))
            if k == 0:
                continue
            picked = sample_size_s(s, k)
            for z in picked:
                rows.append(z)
                rows.append(frozenset(range(n)) - z)
                row_sizes += [s, comp]
            inclusion[s] = inclusion[comp] = k / c_ns
        else:  # middle size for even n: pin the last player to partition pairs
            k = int(round(min(c, c_ns)))
            k -= k % 2
            if k == 0:
                continue
            half = list(itertools.combinations(range(n - 1), s - 1))
            idx = rng.choice(len(half), size=k // 2, replace=False)
            for i in idx:
                z = frozenset(half[i]) | {n - 1}
                rows.append(z)
                rows.append(frozenset(range(n)) - z)
                row_sizes += [s, s]
            inclusion[s] = k / c_ns

    v_empty = float(np.mean(game.evaluate(frozenset(), replicates)))
    v_full = float(np.mean(game.evaluate(frozenset(range(n)), replicates)))
    total = v_full - v_empty
    y = np.array(
        [float(np.mean(game.evaluate(z, replicates))) - v_empty for z in rows]
    )

    z_mat = np.zeros((len(rows), n))
    for i, z in enumerate(rows):
        z_mat[i, list(z)] = 1.0
    s_arr = np.array(row_sizes, dtype=float)
    kernel_w = 1.0 / (np.array([binom(n, s) for s in row_sizes]) * s_arr * (n - s_arr))
    p_arr = np.array([inclusion[s] for s in row_sizes])
    sqrt_w = np.sqrt(kernel_w / p_arr)

    proj = np.eye(n) - np.ones((n, n)) / n
    a_mat = (sqrt_w[:, None] * z_mat) @ proj
    b_vec = sqrt_w * (y - total * s_arr / n)
    x, *_ = np.linalg.lstsq(a_mat, b_vec, rcond=None)
    values = proj @ x + total / n

    return ShapleyResult(
        values=values,
        halfwidths=np.full(n, np.nan),
        calls=game.calls - start_calls,
        meta={"method": "leverage_shap", "replicates": replicates, "c": c},
    )
