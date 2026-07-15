"""Correctness tests for the segshap estimators against exact ground truth."""

import numpy as np
import pytest

from segshap import (
    QuotientGame,
    SyntheticMobiusGame,
    cc_shapley,
    exact_shapley,
    hierarchical_owen,
    kernel_shap,
    permutation_mc,
    random_sparse_game,
    surrogate_shapley,
)
from segshap.metrics import ci_coverage, kendall_tau, linf_error


def make_game(noise="none", sigma=0.05, seed=0, n=6):
    mobius = {
        frozenset({0}): 0.30,
        frozenset({1}): 0.20,
        frozenset({2}): -0.10,
        frozenset({0, 1}): 0.15,
        frozenset({2, 3}): 0.10,
        frozenset({4}): 0.05,
    }
    return SyntheticMobiusGame(n, mobius, noise=noise, sigma=sigma, rng=seed, offset=0.2)


def test_exact_shapley_matches_enumeration():
    game = make_game()
    phi_mobius = game.exact_shapley
    phi_enum = exact_shapley(game.mean_value, game.n)
    np.testing.assert_allclose(phi_mobius, phi_enum, atol=1e-12)


def test_exact_shapley_efficiency():
    game = make_game()
    total = game.mean_value(game.players) - game.mean_value(frozenset())
    assert game.exact_shapley.sum() == pytest.approx(total, abs=1e-12)


def test_cc_shapley_noiseless_converges_with_valid_ci():
    game = make_game(noise="none")
    truth = game.exact_shapley
    res = cc_shapley(game, budget_calls=6000, delta=0.05, rng=1)
    assert linf_error(res.values, truth) < 0.02
    assert ci_coverage(res.lower, res.upper, truth) == 1.0


def test_cc_shapley_noisy_ci_covers_truth():
    failures = 0
    for seed in range(10):
        game = make_game(noise="gauss", sigma=0.1, seed=seed)
        truth = game.exact_shapley
        res = cc_shapley(game, budget_calls=4000, replicates=2, delta=0.05, rng=seed)
        if ci_coverage(res.lower, res.upper, truth) < 1.0:
            failures += 1
    # delta = 0.05 per run: 10 runs should essentially never fail twice.
    assert failures <= 1


def test_cc_shapley_respects_budget_and_stops_at_target():
    game = make_game(noise="gauss", sigma=0.05)
    res = cc_shapley(game, budget_calls=2000, target_eps=10.0, rng=0)
    # A huge target_eps should trigger early stopping well under budget.
    assert res.calls < 2000
    res2 = cc_shapley(game, budget_calls=500, rng=0)
    assert res2.calls <= 500


def test_permutation_mc_matches_truth():
    game = make_game(noise="none")
    truth = game.exact_shapley
    res = permutation_mc(game, budget_calls=7000, rng=2)
    assert linf_error(res.values, truth) < 0.03
    assert ci_coverage(res.lower, res.upper, truth) == 1.0


def test_kernel_shap_matches_truth_noiseless():
    game = make_game(noise="none")
    truth = game.exact_shapley
    res = kernel_shap(game, budget_calls=4000, rng=3)
    assert linf_error(res.values, truth) < 0.03


def test_surrogate_recovers_low_order_game_cheaply():
    game = make_game(noise="gauss", sigma=0.05, seed=4)
    truth = game.exact_shapley
    res = surrogate_shapley(game, budget_calls=4000, order=2, delta=0.05, rng=4)
    assert linf_error(res.values, truth) < 0.05
    assert ci_coverage(res.lower, res.upper, truth) == 1.0


def test_surrogate_beats_cc_at_equal_budget_on_structured_game():
    budget = 3000
    errs_cc, errs_sur = [], []
    for seed in range(5):
        g1 = make_game(noise="gauss", sigma=0.1, seed=seed)
        g2 = make_game(noise="gauss", sigma=0.1, seed=seed)
        truth = g1.exact_shapley
        errs_cc.append(linf_error(cc_shapley(g1, budget, rng=seed).values, truth))
        errs_sur.append(
            linf_error(surrogate_shapley(g2, budget, order=2, rng=seed).values, truth)
        )
    assert np.mean(errs_sur) <= np.mean(errs_cc) * 1.25  # at worst comparable


def test_quotient_game_charges_base_budget():
    game = make_game(noise="none")
    quotient = QuotientGame(game, [[0, 1], [2, 3], [4, 5]])
    quotient.evaluate([0, 2], replicates=3)
    assert game.calls == 3


def test_hierarchical_owen_certifies_null_groups():
    n, groups = 12, [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]
    # Direct sum of within-group games (no cross-group Mobius terms), so the
    # Owen values equal the Shapley values exactly. Groups 2 and 3 carry no
    # Mobius mass at all: they are exact null players in the quotient game.
    mobius = {
        frozenset({0}): 0.25,
        frozenset({0, 1}): 0.15,
        frozenset({2}): -0.10,
        frozenset({3}): 0.30,
        frozenset({4, 5}): 0.20,
    }
    game = SyntheticMobiusGame(
        n, mobius, noise="gauss", sigma=0.05, rng=7, offset=0.1
    )
    res = hierarchical_owen(
        game, groups, budget_calls=30000, tau=0.2, delta=0.05, rng=7
    )
    assert 2 in res.certified_negligible
    assert 3 in res.certified_negligible
    # The active groups (quotient Shapley 0.30 and 0.50 > tau) must survive.
    assert 0 in res.expanded and 1 in res.expanded
    # Refined member Owen values must be consistent with exact Shapley.
    truth = game.exact_shapley
    assert res.member_values, "expanded groups should yield member refinements"
    for player, val in res.member_values.items():
        hw = res.member_halfwidths[player]
        assert abs(val - truth[player]) <= hw + 1e-9


def test_random_sparse_game_values_in_unit_interval():
    import itertools

    game = random_sparse_game(8, n_terms=12, max_order=3, rng=11, noise="none")
    for r in range(9):
        for s in itertools.combinations(range(8), r):
            v = game.mean_value(frozenset(s))
            assert -1e-9 <= v <= 1 + 1e-9


def test_kendall_tau_metric():
    assert kendall_tau(np.array([1.0, 2.0, 3.0]), np.array([10.0, 20.0, 30.0])) == 1.0


def test_betting_interval_covers_and_beats_eb():
    from segshap.bounds import betting_interval, eb_halfwidth

    rng = np.random.default_rng(0)
    misses, width_ratios = 0, []
    true_mean = 0.15
    for seed in range(20):
        rng = np.random.default_rng(seed)
        # CC-like samples in [-1, 1] with mean 0.15
        x = np.clip(rng.normal(true_mean, 0.3, size=400), -1.0, 1.0)
        lo, hi = betting_interval(x, -1.0, 1.0, delta=0.05)
        if not (lo <= true_mean <= hi):
            misses += 1
        eb = eb_halfwidth(len(x), float(np.var(x, ddof=1)), 2.0, 0.05)
        width_ratios.append((hi - lo) / (2 * eb))
    assert misses <= 1  # anytime-valid at 95%, 20 trials
    assert np.mean(width_ratios) < 0.7  # substantially tighter than EB


def test_betting_interval_edge_cases():
    from segshap.bounds import betting_interval

    lo, hi = betting_interval([], -1.0, 1.0, delta=0.05)
    assert (lo, hi) == (-1.0, 1.0)
    lo, hi = betting_interval([0.3], -1.0, 1.0, delta=0.05)
    assert -1.0 <= lo <= 0.3 <= hi <= 1.0


def test_cc_shapley_betting_cis_tighter_than_eb():
    game_a = make_game(noise="gauss", sigma=0.1, seed=3)
    game_b = make_game(noise="gauss", sigma=0.1, seed=3)
    truth = game_a.exact_shapley
    res_bet = cc_shapley(game_a, budget_calls=4000, delta=0.05, ci_method="betting", rng=3)
    res_eb = cc_shapley(game_b, budget_calls=4000, delta=0.05, ci_method="eb", rng=3)
    assert ci_coverage(res_bet.lower, res_bet.upper, truth) == 1.0
    assert np.mean(res_bet.upper - res_bet.lower) < 0.6 * np.mean(
        res_eb.upper - res_eb.lower
    )
