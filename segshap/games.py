"""Cooperative games observed through a stochastic, pay-per-call oracle.

A *game* here is a set function v: 2^N -> R that we can only query through
``evaluate(coalition, replicates)``, which returns ``replicates`` i.i.d. noisy
samples of v(S) and charges ``replicates`` oracle calls against the budget.
This mirrors the LLM setting, where v(S) is the expected task utility of the
prompt containing exactly the segments in S and each call is one
question-draw + generation.
"""

from __future__ import annotations

import itertools
import math
from abc import ABC, abstractmethod
from typing import Callable, Dict, FrozenSet, Iterable, Optional, Sequence

import numpy as np

Coalition = FrozenSet[int]


class BudgetExceeded(RuntimeError):
    """Raised when an evaluate() call would exceed the game's call budget."""


class NoisyGame(ABC):
    """Base class: budgeted access to noisy samples of a set function."""

    def __init__(
        self,
        n: int,
        rng: Optional[np.random.Generator | int] = None,
        budget: Optional[int] = None,
        value_range: tuple[float, float] = (0.0, 1.0),
    ):
        self.n = n
        self.rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
        self.budget = budget
        self.calls = 0
        self.value_range = value_range

    @property
    def players(self) -> FrozenSet[int]:
        return frozenset(range(self.n))

    @property
    def range(self) -> float:
        return self.value_range[1] - self.value_range[0]

    def evaluate(self, coalition: Iterable[int], replicates: int = 1) -> np.ndarray:
        """Return ``replicates`` i.i.d. noisy samples of v(coalition)."""
        if self.budget is not None and self.calls + replicates > self.budget:
            raise BudgetExceeded(
                f"budget {self.budget} would be exceeded ({self.calls} used, {replicates} requested)"
            )
        self.calls += replicates
        return self._sample(frozenset(coalition), replicates)

    def evaluate_many(
        self, coalitions: Sequence[Iterable[int]], replicates: int = 1
    ) -> list:
        """Evaluate several coalitions; subclasses may parallelize.

        Statistically identical to sequential evaluate() calls — batching
        exists so expensive oracles (LLM APIs) can fire requests concurrently.
        """
        return [self.evaluate(c, replicates) for c in coalitions]

    @abstractmethod
    def _sample(self, coalition: Coalition, replicates: int) -> np.ndarray:
        ...


class SyntheticMobiusGame(NoisyGame):
    """A game defined by its Mobius (Harsanyi dividend) coefficients.

    mean v(S) = sum_{T subseteq S, T != empty} mobius[T] + offset.

    Noise models:
      - ``"gauss"``: v(S) + N(0, sigma^2)
      - ``"bernoulli"``: Bernoulli(clip(v(S), 0, 1)) — matches a binary
        correctness utility, the noisiest realistic LLM regime.
      - ``"none"``: deterministic.
    """

    def __init__(
        self,
        n: int,
        mobius: Dict[Coalition, float],
        noise: str = "gauss",
        sigma: float = 0.1,
        offset: float = 0.0,
        **kwargs,
    ):
        super().__init__(n, **kwargs)
        self.mobius = {frozenset(t): float(c) for t, c in mobius.items() if len(t) > 0}
        self.noise = noise
        self.sigma = sigma
        self.offset = offset

    def mean_value(self, coalition: Iterable[int]) -> float:
        s = frozenset(coalition)
        return self.offset + sum(c for t, c in self.mobius.items() if t <= s)

    def _sample(self, coalition: Coalition, replicates: int) -> np.ndarray:
        mu = self.mean_value(coalition)
        if self.noise == "none":
            return np.full(replicates, mu)
        if self.noise == "gauss":
            return mu + self.rng.normal(0.0, self.sigma, size=replicates)
        if self.noise == "bernoulli":
            p = min(max(mu, 0.0), 1.0)
            return self.rng.binomial(1, p, size=replicates).astype(float)
        raise ValueError(f"unknown noise model {self.noise!r}")

    @property
    def exact_shapley(self) -> np.ndarray:
        """phi_i = sum_{T ni i} m(T) / |T| (exact, from the Mobius basis)."""
        phi = np.zeros(self.n)
        for t, c in self.mobius.items():
            share = c / len(t)
            for i in t:
                phi[i] += share
        return phi


class QuotientGame(NoisyGame):
    """View a base game through a partition: players are groups of base players.

    v_Q(C) = v(union of the groups in C). Calls are charged to the *base*
    game's budget so that hierarchical methods account cost honestly.
    """

    def __init__(self, base: NoisyGame, groups: Sequence[Sequence[int]]):
        super().__init__(n=len(groups), rng=base.rng, budget=None, value_range=base.value_range)
        self.base = base
        self.groups = [tuple(g) for g in groups]
        flat = [i for g in self.groups for i in g]
        if sorted(flat) != list(range(base.n)):
            raise ValueError("groups must partition the base players 0..n-1")

    def evaluate(self, coalition: Iterable[int], replicates: int = 1) -> np.ndarray:
        union = frozenset(i for g in coalition for i in self.groups[g])
        # Mirror the spend on this wrapper so estimators tracking `calls` on
        # the quotient view (e.g. cc_shapley's budget loop) see real progress.
        self.calls += replicates
        return self.base.evaluate(union, replicates)

    def evaluate_many(
        self, coalitions: Sequence[Iterable[int]], replicates: int = 1
    ) -> list:
        unions = [
            frozenset(i for g in c for i in self.groups[g]) for c in coalitions
        ]
        self.calls += replicates * len(unions)
        return self.base.evaluate_many(unions, replicates)

    def _sample(self, coalition: Coalition, replicates: int) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError("QuotientGame delegates to the base game")


class TransformedGame(NoisyGame):
    """Base game with a deterministic function of the coalition subtracted.

    Used to run an estimator on the *residual* game h(S) = v(S) - g(S) when a
    surrogate g is used as a control variate (Approach C).
    """

    def __init__(self, base: NoisyGame, shift_fn: Callable[[Coalition], float], extra_range: float):
        super().__init__(
            n=base.n,
            rng=base.rng,
            budget=None,
            value_range=(base.value_range[0] - extra_range, base.value_range[1]),
        )
        self.base = base
        self.shift_fn = shift_fn
        self._range = base.range + extra_range

    @property
    def range(self) -> float:
        return self._range

    def evaluate(self, coalition: Iterable[int], replicates: int = 1) -> np.ndarray:
        s = frozenset(coalition)
        self.calls += replicates  # keep the wrapper's spend observable
        return self.base.evaluate(s, replicates) - self.shift_fn(s)

    def evaluate_many(
        self, coalitions: Sequence[Iterable[int]], replicates: int = 1
    ) -> list:
        sets = [frozenset(c) for c in coalitions]
        self.calls += replicates * len(sets)
        raw = self.base.evaluate_many(sets, replicates)
        return [r - self.shift_fn(s) for r, s in zip(raw, sets)]

    def _sample(self, coalition: Coalition, replicates: int) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError("TransformedGame delegates to the base game")


def exact_shapley(mean_fn: Callable[[Coalition], float], n: int) -> np.ndarray:
    """Exact Shapley values by full enumeration (2^n evaluations of mean_fn)."""
    values = {}
    players = list(range(n))
    for r in range(n + 1):
        for s in itertools.combinations(players, r):
            values[frozenset(s)] = mean_fn(frozenset(s))
    phi = np.zeros(n)
    fact = math.factorial
    for s, vs in values.items():
        k = len(s)
        for i in players:
            if i in s:
                continue
            w = fact(k) * fact(n - k - 1) / fact(n)
            phi[i] += w * (values[s | {i}] - vs)
    return phi


def random_sparse_game(
    n: int,
    n_terms: int,
    max_order: int = 2,
    rng: Optional[np.random.Generator | int] = None,
    noise: str = "bernoulli",
    sigma: float = 0.1,
    support: Optional[Sequence[int]] = None,
    **kwargs,
) -> SyntheticMobiusGame:
    """A random sparse low-order game rescaled so mean values lie in [0, 1].

    ``support`` restricts the players that carry nonzero Mobius mass (the rest
    are exact null players) — used to plant negligible groups for Approach B.
    """
    rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    pool = list(support) if support is not None else list(range(n))
    mobius: Dict[Coalition, float] = {}
    while len(mobius) < n_terms:
        order = int(rng.integers(1, max_order + 1))
        order = min(order, len(pool))
        t = frozenset(rng.choice(pool, size=order, replace=False).tolist())
        if t not in mobius:
            mobius[t] = float(rng.normal(0.0, 1.0))

    raw = SyntheticMobiusGame(n, mobius, noise="none")
    all_vals = [
        raw.mean_value(frozenset(s))
        for r in range(n + 1)
        for s in itertools.combinations(range(n), r)
    ]
    lo, hi = min(all_vals), max(all_vals)
    scale = (hi - lo) if hi > lo else 1.0
    rescaled = {t: c / scale for t, c in mobius.items()}
    return SyntheticMobiusGame(
        n, rescaled, noise=noise, sigma=sigma, offset=-lo / scale, rng=rng, **kwargs
    )
