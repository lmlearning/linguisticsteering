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

from typing import Optional

import numpy as np

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
