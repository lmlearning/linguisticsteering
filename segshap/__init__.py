"""segshap: certified segment-level Shapley attribution with a budgeted noisy oracle.

Implements the three approaches of FOLLOW_UP_PROPOSAL.md against a single
coalition-oracle abstraction:

- estimators.cc_bernstein  (Approach A: paired complementary coalitions +
  anytime empirical-Bernstein certificates)
- estimators.tree_elim     (Approach B: hierarchical Owen values with
  certified adaptive pruning)
- estimators.surrogate     (Approach C: low-order Mobius surrogate used as a
  control variate, certified by a residual CC-Bernstein pass)

plus classical baselines (permutation Monte Carlo, KernelSHAP) and synthetic
testbeds with exact ground truth.
"""

from segshap.games import (
    NoisyGame,
    SyntheticMobiusGame,
    QuotientGame,
    BudgetExceeded,
    exact_shapley,
    random_sparse_game,
)
from segshap.estimators.cc_bernstein import cc_shapley
from segshap.estimators.tree_elim import hierarchical_owen
from segshap.estimators.surrogate import surrogate_shapley
from segshap.estimators.baselines import permutation_mc, kernel_shap

__all__ = [
    "NoisyGame",
    "SyntheticMobiusGame",
    "QuotientGame",
    "BudgetExceeded",
    "exact_shapley",
    "random_sparse_game",
    "cc_shapley",
    "hierarchical_owen",
    "surrogate_shapley",
    "permutation_mc",
    "kernel_shap",
]
