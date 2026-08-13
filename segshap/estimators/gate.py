"""Intervention gate: a betting certificate for a removal effect Delta_P.

Screening (segshap.estimators.tree_elim) certifies *average* contributions
(Shapley/Owen values); it does not bound the effect of an actual removal
Delta_P = v(N) - v(N \\ P), because averages over contexts do not determine the
grand-coalition effect (Remark: unanimity game). The gate therefore certifies
the intervention directly (Thm. "Intervention gate"): draw paired observations
d_j = u(y_j, q_j) - u(y'_j, q_j) with the same task q_j, y_j from the full
prompt and y'_j from the pruned prompt, and maintain one betting confidence
sequence for their mean Delta_P. A removal is deployed only if the sequence
lies inside [-tau_delta, tau_delta].
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np

from segshap.bounds import betting_interval
from segshap.games import BudgetExceeded, NoisyGame


@dataclass
class GateResult:
    delta_hat: float
    lower: float
    upper: float
    approved: bool
    tau_delta: float
    calls: int
    meta: dict = field(default_factory=dict)


def intervention_gate(
    game: NoisyGame,
    remove: Iterable[int],
    budget_calls: int,
    tau_delta: float,
    replicates: int = 1,
    delta: float = 0.05,
    paired: bool = True,
    rng: Optional[np.random.Generator | int] = None,
) -> GateResult:
    """Certify Delta_P = v(N) - v(N\\P) with an anytime betting CS at level delta.

    Each step spends 2*replicates calls (full prompt and pruned prompt). The
    removal is approved iff the whole interval lies in [-tau_delta, tau_delta].
    """
    n = game.n
    rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    full = frozenset(range(n))
    pruned = full - frozenset(int(i) for i in remove)
    diffs: list[float] = []
    start_calls = game.calls
    step_cost = 2 * replicates

    while game.calls - start_calls + step_cost <= budget_calls:
        try:
            v_full = float(np.mean(game.evaluate(full, replicates)))
            v_pruned = float(np.mean(game.evaluate(pruned, replicates)))
        except BudgetExceeded:
            break
        diffs.append(v_full - v_pruned)

    lo, hi = betting_interval(diffs, -1.0, 1.0, delta)
    lo, hi = max(-1.0, lo), min(1.0, hi)
    approved = bool(lo >= -tau_delta and hi <= tau_delta)
    return GateResult(
        delta_hat=float(np.mean(diffs)) if diffs else 0.0,
        lower=lo,
        upper=hi,
        approved=approved,
        tau_delta=tau_delta,
        calls=game.calls - start_calls,
        meta={"method": "intervention_gate", "delta": delta,
              "removed": sorted(int(i) for i in remove), "paired": paired,
              "steps": len(diffs)},
    )
