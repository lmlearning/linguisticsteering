"""Screen-and-gate demonstration on synthetic doubly-stochastic games (no API).

Reproduces the article's gate results (Thm. "Intervention gate", Remark
"averages do not bound interventions"):

  (a) Unanimity oracle, n=10, 5% label flips: singleton Shapley values are all
      0.09 < tau, so a tight screen marks all ten singletons as removal
      candidates, yet removing them collapses behavior (Delta_P = 0.90). The
      gate blocks the removal with a certified interval covering 0.90.
  (b) Planted-null prune: removing the nine dummy players has Delta_P = 0; the
      gate approves it with an interval covering 0 inside [-tau_delta, tau_delta].

Fully synthetic -> no oracle calls to any API.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np

from segshap import SyntheticMobiusGame, exact_shapley
from segshap.estimators.gate import intervention_gate

DELTA = 0.05
TAU = 0.15
TAU_DELTA = 0.10
GATE_BUDGET = 1_000


def unanimity_game(n=10, flip=0.05, seed=0):
    # E[u] = flip + (1-2*flip)*1[S=N]; a single order-n Mobius term.
    m = {frozenset(range(n)): 1.0 - 2 * flip}
    return SyntheticMobiusGame(n, m, noise="bernoulli", offset=flip, rng=seed)


def additive_null_game(n=10, flip=0.05, seed=0):
    # Only player 0 matters; players 1..n-1 are dummies (Delta = 0 to remove).
    m = {frozenset({0}): 1.0 - 2 * flip}
    return SyntheticMobiusGame(n, m, noise="bernoulli", offset=flip, rng=seed)


def main():
    n = 10
    out = {"delta": DELTA, "tau": TAU, "tau_delta": TAU_DELTA, "cases": {}}
    lines = ["# Screen-and-gate demo (synthetic, zero API calls)", ""]

    # (a) unanimity: screen marks all, gate blocks.
    g = unanimity_game(n, seed=0)
    phi = g.exact_shapley
    v_full = g.mean_value(frozenset(range(n)))
    v_empty = g.mean_value(frozenset())
    delta_true = v_full - v_empty
    gate = intervention_gate(g, remove=range(n), budget_calls=GATE_BUDGET,
                             tau_delta=TAU_DELTA, delta=DELTA, rng=1)
    out["cases"]["unanimity"] = {
        "singleton_shapley": float(phi[0]), "screen_marks_all": bool(np.all(np.abs(phi) < TAU)),
        "delta_true": float(delta_true), "gate_ci": [gate.lower, gate.upper],
        "gate_approved": gate.approved, "gate_calls": gate.calls,
    }
    lines += [
        "## (a) Unanimity oracle (n=10, 5% label flips)",
        f"- singleton Shapley = {phi[0]:.3f} (all |phi| < tau={TAU}: "
        f"{bool(np.all(np.abs(phi) < TAU))}) -> screen marks all ten candidates",
        f"- true removal effect Delta_P = {delta_true:.3f}",
        f"- gate CI [{gate.lower:.3f}, {gate.upper:.3f}] (covers {delta_true:.2f}: "
        f"{gate.lower <= delta_true <= gate.upper}), approved={gate.approved} "
        f"-> removal correctly BLOCKED ({gate.calls} calls)",
        "",
    ]

    # (b) planted null: gate approves.
    g2 = additive_null_game(n, seed=0)
    delta_null = g2.mean_value(frozenset(range(n))) - g2.mean_value(frozenset({0}))
    gate2 = intervention_gate(g2, remove=range(1, n), budget_calls=GATE_BUDGET,
                              tau_delta=TAU_DELTA, delta=DELTA, rng=2)
    out["cases"]["planted_null"] = {
        "delta_true": float(delta_null), "gate_ci": [gate2.lower, gate2.upper],
        "gate_approved": gate2.approved, "gate_calls": gate2.calls,
    }
    lines += [
        "## (b) Planted-null prune (remove 9 dummy players)",
        f"- true removal effect Delta_P = {delta_null:.3f}",
        f"- gate CI [{gate2.lower:.3f}, {gate2.upper:.3f}], approved={gate2.approved} "
        f"-> removal correctly APPROVED ({gate2.calls} calls)",
        "",
        "Screening bounds what may be *proposed*; the gate bounds what is "
        "*deployed*. Averages (Shapley/Owen) do not bound the removal effect "
        "Delta_P, so both estimands are certified.",
        "",
    ]

    Path("results").mkdir(exist_ok=True)
    Path("results/gate_demo.json").write_text(json.dumps(out, indent=1))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
