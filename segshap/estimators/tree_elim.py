"""Approach B — TreeSHAP-Elim: hierarchical Owen values with certified pruning.

Stage 1 estimates the *quotient game* (players = groups) with the CC-Bernstein
estimator. Groups whose simultaneous CI lies inside [-tau, tau] are certified
negligible at group granularity and never expanded — the certificate is a
statement about the group's Shapley value in the quotient game, which is the
practically meaningful "can this whole block be triaged?" quantity.

Stage 2 refines each surviving group g: the Owen value of member i is

    owen_i = E_U [ phi_i( T -> v(U u T) ) ],

where U is the union of a uniformly-ordered random prefix of the *other*
groups (equivalently: |prefix| uniform on {0..m-1}, subset uniform given the
size) and the inner Shapley is over the within-group game w_U(T) = v(U u T).
We estimate the inner Shapley with the same CC identity, so each evaluated
pair (v(U u T), v(U u (g \\ T))) updates every member of g. Sampling U fresh
for every pair makes each stratum's samples i.i.d. draws of the *averaged*
quantity, so the empirical-Bernstein machinery applies unchanged and the CIs
account for both sources of randomness.

The total budget is split between stages; stage-2 budget is divided among
surviving groups proportionally to their remaining CI width (spend where the
uncertainty is).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from segshap.bounds import StratifiedEstimate
from segshap.estimators.cc_bernstein import ShapleyResult, cc_shapley
from segshap.games import BudgetExceeded, NoisyGame, QuotientGame


@dataclass
class HierarchicalResult:
    group_values: np.ndarray
    group_halfwidths: np.ndarray
    certified_negligible: list  # group indices certified in [-tau, tau]
    expanded: list  # group indices refined at member level
    member_values: dict  # player -> Owen value (players of expanded groups only)
    member_halfwidths: dict
    calls: int
    meta: dict = field(default_factory=dict)


def _owen_members(
    game: NoisyGame,
    groups: Sequence[Sequence[int]],
    g_idx: int,
    budget_calls: int,
    replicates: int,
    delta: float,
    rng: np.random.Generator,
    min_pairs_per_size: int = 3,
    ci_method: str = "betting",
) -> ShapleyResult:
    members = list(groups[g_idx])
    k = len(members)
    others = [j for j in range(len(groups)) if j != g_idx]
    est = StratifiedEstimate(
        n_players=k, n_strata=k, value_range=2.0 * game.range, delta=delta
    )
    start_calls = game.calls
    pair_cost = 2 * replicates

    def draw_pairs(t_batch: list) -> None:
        """One (T, g\\T) pair per requested size, one concurrent eval batch."""
        pairs = []
        for t in t_batch:
            n_pre = int(rng.integers(0, len(others) + 1)) if others else 0
            prefix = rng.choice(others, size=n_pre, replace=False) if n_pre else []
            u = frozenset(i for j in prefix for i in groups[int(j)])
            inside = rng.choice(k, size=t, replace=False)
            t_set = frozenset(members[int(i)] for i in inside)
            c_set = frozenset(members) - t_set
            pairs.append((t, t_set, u | t_set, u | c_set))
        flat = [c for _, _, a, b in pairs for c in (a, b)]
        values = game.evaluate_many(flat, replicates)
        for idx, (t, t_set, _, _) in enumerate(pairs):
            v_t = float(np.mean(values[2 * idx]))
            v_c = float(np.mean(values[2 * idx + 1]))
            cc = v_t - v_c
            for local, i in enumerate(members):
                if i in t_set:
                    est.add(local, t - 1, cc)
                else:
                    est.add(local, (k - t) - 1, -cc)

    def spent() -> int:
        return game.calls - start_calls

    for _ in range(min_pairs_per_size):
        room = max(0, (budget_calls - spent()) // pair_cost)
        t_batch = list(range(1, k + 1))[:room]
        if not t_batch:
            break
        try:
            draw_pairs(t_batch)
        except BudgetExceeded:
            break
    batch_pairs = max(16, k)
    while spent() + pair_cost <= budget_calls:
        hw = np.array(
            [
                sum(est.stratum_halfwidth(i, t) for i in range(k))
                for t in range(k)
            ]
        )
        hw[~np.isfinite(hw)] = 4.0 * game.range * k
        t_next = int(np.argmax(hw)) + 1
        room = (budget_calls - spent()) // pair_cost
        n_batch = int(min(batch_pairs, room))
        if n_batch <= 0:
            break
        try:
            draw_pairs([t_next] * n_batch)
        except BudgetExceeded:
            break

    lower, upper = est.intervals(method=ci_method)
    return ShapleyResult(
        values=est.values(),
        halfwidths=(upper - lower) / 2.0,
        calls=spent(),
        meta={"method": "owen_members", "group": g_idx, "ci_method": ci_method},
        lower_bounds=lower,
        upper_bounds=upper,
    )


def hierarchical_owen(
    game: NoisyGame,
    groups: Sequence[Sequence[int]],
    budget_calls: int,
    tau: float,
    replicates: int = 1,
    delta: float = 0.05,
    stage1_frac: float = 0.35,
    ci_method: str = "betting",
    rng: Optional[np.random.Generator | int] = None,
) -> HierarchicalResult:
    """Certified triage of a segment hierarchy under a total call budget.

    The overall failure probability is split evenly between the two stages, so
    all reported statements (group certificates and member CIs) hold jointly
    with probability >= 1 - delta.
    """
    rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    m = len(groups)
    start_calls = game.calls

    quotient = QuotientGame(game, groups)
    stage1 = cc_shapley(
        quotient,
        budget_calls=int(budget_calls * stage1_frac),
        replicates=replicates,
        delta=delta / 2.0,
        ci_method=ci_method,
        rng=rng,
    )

    lo, hi = stage1.lower, stage1.upper
    certified = [g for g in range(m) if lo[g] >= -tau and hi[g] <= tau]
    expanded = [g for g in range(m) if g not in certified]

    member_values: dict = {}
    member_halfwidths: dict = {}
    remaining = budget_calls - (game.calls - start_calls)
    if expanded and remaining > 0:
        widths = np.array([stage1.halfwidths[g] + tau for g in expanded])
        shares = widths / widths.sum()
        per_group_delta = (delta / 2.0) / len(expanded)
        for g, share in zip(expanded, shares):
            sub_budget = int(remaining * share)
            if sub_budget <= 2 * replicates:
                continue
            res = _owen_members(
                game, groups, g, sub_budget, replicates, per_group_delta, rng,
                ci_method=ci_method,
            )
            for local, player in enumerate(groups[g]):
                member_values[player] = float(res.values[local])
                member_halfwidths[player] = float(res.halfwidths[local])

    return HierarchicalResult(
        group_values=stage1.values,
        group_halfwidths=stage1.halfwidths,
        certified_negligible=certified,
        expanded=expanded,
        member_values=member_values,
        member_halfwidths=member_halfwidths,
        calls=game.calls - start_calls,
        meta={"tau": tau, "delta": delta, "stage1_calls": stage1.calls},
    )
