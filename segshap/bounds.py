"""Anytime-valid empirical-Bernstein confidence intervals and accumulators.

The base bound is Maurer & Pontil (2009): for m i.i.d. samples of a random
variable with range R, with probability >= 1 - d,

    |mean_hat - mu| <= sqrt(2 * Var_hat * ln(2/d) / m) + 7 R ln(2/d) / (3 (m-1)).

Our estimators choose *how many* samples each stratum gets adaptively, based
on the data seen so far, so a fixed-m bound is not directly valid. We make the
bound anytime-valid by a union bound over sample counts (stake d_m =
d / (m (m+1)) at count m, sum_m d_m <= d), i.e. we replace ln(2/d) with
ln(2 m (m+1) / d). Within a stratum, samples remain i.i.d. regardless of the
allocation policy, which is all the union bound needs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class StratumStats:
    """Welford accumulator for one stratum's i.i.d. samples."""

    count: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def add(self, x: float) -> None:
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (x - self.mean)

    @property
    def variance(self) -> float:
        if self.count < 2:
            return math.inf
        return self.m2 / (self.count - 1)


def eb_halfwidth(count: int, variance: float, value_range: float, delta: float) -> float:
    """Anytime-valid empirical-Bernstein confidence half-width."""
    if count < 2 or not math.isfinite(variance):
        return math.inf
    log_term = math.log(2.0 * count * (count + 1) / delta)
    return math.sqrt(2.0 * variance * log_term / count) + (
        7.0 * value_range * log_term / (3.0 * (count - 1))
    )


def hoeffding_halfwidth(count: int, value_range: float, delta: float) -> float:
    """Anytime-valid Hoeffding half-width (variance-free, for baselines)."""
    if count < 1:
        return math.inf
    log_term = math.log(2.0 * count * (count + 1) / delta)
    return value_range * math.sqrt(log_term / (2.0 * count))


@dataclass
class StratifiedEstimate:
    """A per-player estimate assembled from equally weighted strata.

    Used by both the CC identity (phi_i = (1/n) sum_s mu_{i,s}) and its
    within-group Owen analogue. The player's confidence interval is the sum of
    per-stratum intervals, each run at delta / (n_players * n_strata) so that
    all intervals hold simultaneously with probability >= 1 - delta.
    """

    n_players: int
    n_strata: int
    value_range: float
    delta: float
    stats: list = field(default_factory=list)

    def __post_init__(self):
        self.stats = [
            [StratumStats() for _ in range(self.n_strata)] for _ in range(self.n_players)
        ]
        self._delta_stratum = self.delta / (self.n_players * self.n_strata)

    def add(self, player: int, stratum: int, x: float) -> None:
        self.stats[player][stratum].add(x)

    def values(self) -> np.ndarray:
        out = np.zeros(self.n_players)
        for i in range(self.n_players):
            out[i] = sum(st.mean for st in self.stats[i]) / self.n_strata
        return out

    def halfwidths(self) -> np.ndarray:
        out = np.zeros(self.n_players)
        for i in range(self.n_players):
            out[i] = (
                sum(
                    eb_halfwidth(st.count, st.variance, self.value_range, self._delta_stratum)
                    for st in self.stats[i]
                )
                / self.n_strata
            )
        return out

    def stratum_halfwidth(self, player: int, stratum: int) -> float:
        st = self.stats[player][stratum]
        return eb_halfwidth(st.count, st.variance, self.value_range, self._delta_stratum)
