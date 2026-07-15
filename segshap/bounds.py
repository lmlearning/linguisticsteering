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
    """Welford accumulator for one stratum's i.i.d. samples.

    Also retains the raw sample sequence so that betting confidence sequences
    (which need the ordered samples) can be evaluated at reporting time.
    """

    count: int = 0
    mean: float = 0.0
    m2: float = 0.0
    samples: list = field(default_factory=list)

    def add(self, x: float) -> None:
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (x - self.mean)
        self.samples.append(x)

    @property
    def variance(self) -> float:
        if self.count < 2:
            return math.inf
        return self.m2 / (self.count - 1)


def betting_interval(
    samples,
    lower_bound: float,
    upper_bound: float,
    delta: float,
    grid_size: int = 401,
) -> tuple[float, float]:
    """Hedged-capital betting confidence interval for a bounded mean.

    Waudby-Smith & Ramdas (2023), "Estimating means of bounded random
    variables by betting": for each candidate mean m, a gambler bets against
    m with predictable plug-in bet sizes; the capital process
    K_t(m) = prod_i (1 + lambda_i(m) (x_i - m)) is a nonnegative martingale
    under the true mean, so by Ville's inequality the set
    {m : K_t(m) < 1/delta} is an anytime-valid confidence sequence. The
    hedged version bets both directions and takes the average capital, which
    makes the confidence set an interval. This is uniformly (often several
    times) tighter than empirical-Bernstein at the same delta, at the cost of
    needing the raw samples.

    Returns a conservative outer hull of the grid solution (one grid step of
    padding on each side), so validity is not affected by grid resolution.
    """
    x = np.asarray(samples, dtype=float)
    t = x.size
    if t == 0:
        return lower_bound, upper_bound
    width = upper_bound - lower_bound
    x = np.clip((x - lower_bound) / width, 0.0, 1.0)

    # Predictable plug-in estimates (each uses only samples strictly before i).
    counts = np.arange(1, t + 1)
    prior_sums = 0.5 + np.concatenate(([0.0], np.cumsum(x)[:-1]))
    prior_means = prior_sums / counts
    dev = x - prior_means  # deviation from the predictable running mean
    prior_sq = 0.25 + np.concatenate(([0.0], np.cumsum(dev * dev)[:-1]))
    prior_vars = prior_sq / counts
    lam = np.sqrt(
        2.0 * np.log(2.0 / delta) / (prior_vars * counts * np.log(counts + 1.0))
    )

    grid = np.linspace(0.0, 1.0, grid_size)
    lam_plus = np.minimum(lam[:, None], 0.5 / np.maximum(grid[None, :], 1e-12))
    lam_minus = np.minimum(lam[:, None], 0.5 / np.maximum(1.0 - grid[None, :], 1e-12))
    diff = x[:, None] - grid[None, :]
    log_k_plus = np.sum(np.log1p(lam_plus * diff), axis=0)
    log_k_minus = np.sum(np.log1p(-lam_minus * diff), axis=0)
    log_k = np.logaddexp(log_k_plus, log_k_minus) - math.log(2.0)

    accepted = np.where(log_k < math.log(1.0 / delta))[0]
    if accepted.size == 0:
        j = int(np.argmin(log_k))
        m_lo = m_hi = grid[j]
    else:
        m_lo, m_hi = grid[accepted[0]], grid[accepted[-1]]
    step = 1.0 / (grid_size - 1)
    m_lo, m_hi = max(0.0, m_lo - step), min(1.0, m_hi + step)
    return lower_bound + m_lo * width, lower_bound + m_hi * width


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

    def intervals(self, method: str = "betting") -> tuple[np.ndarray, np.ndarray]:
        """Simultaneous per-player confidence intervals.

        ``method="betting"`` uses the hedged betting confidence sequence per
        stratum (anytime-valid, typically much tighter); ``"eb"`` reproduces
        the symmetric empirical-Bernstein intervals. Both hold jointly with
        probability >= 1 - delta by the same union bound over strata.
        """
        lower = np.zeros(self.n_players)
        upper = np.zeros(self.n_players)
        half_range = self.value_range / 2.0
        for i in range(self.n_players):
            for st in self.stats[i]:
                if method == "betting":
                    lo, hi = betting_interval(
                        st.samples, -half_range, half_range, self._delta_stratum
                    )
                else:
                    hw = eb_halfwidth(
                        st.count, st.variance, self.value_range, self._delta_stratum
                    )
                    lo, hi = st.mean - hw, st.mean + hw
                lower[i] += lo
                upper[i] += hi
            lower[i] /= self.n_strata
            upper[i] /= self.n_strata
        return lower, upper
