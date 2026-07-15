"""Approach C — SurroSHAP-Cert: a low-order surrogate as a certified control variate.

Phase 1 fits a k-additive surrogate g on sampled coalitions: a Lasso over the
Mobius indicator basis {1[T subseteq S] : 1 <= |T| <= k}. The Shapley values
of g are closed-form: phi_i(g) = sum_{T ni i} m_g(T) / |T|.

Phase 2 is the certificate. Shapley values are linear in the game, so

    phi(v) = phi(g) + phi(v - g)               (exactly),

and the residual game h = v - g is estimable by the assumption-free
CC-Bernstein pass (Approach A) at the *residual's* variance. When the game is
close to k-additive the residual is tiny, the CIs collapse quickly, and the
combined estimate is far cheaper than running Approach A on v directly; when
the structural bet fails, the certificate does not lie — the residual CIs
stay wide and the method has gracefully *become* Approach A (all phase-2
samples are ordinary CC samples of h, and phi(g) is just a shift). Nothing
about the guarantee depends on the surrogate being right.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from sklearn.linear_model import Lasso

from segshap.estimators.cc_bernstein import ShapleyResult, cc_shapley
from segshap.games import NoisyGame, TransformedGame


@dataclass
class MobiusSurrogate:
    n: int
    order: int
    terms: list  # list of frozensets, aligned with coefs
    coefs: np.ndarray
    intercept: float

    def predict(self, coalition: frozenset) -> float:
        s = frozenset(coalition)
        return self.intercept + sum(
            c for t, c in zip(self.terms, self.coefs) if c != 0.0 and t <= s
        )

    @property
    def shapley(self) -> np.ndarray:
        phi = np.zeros(self.n)
        for t, c in zip(self.terms, self.coefs):
            if c == 0.0:
                continue
            share = c / len(t)
            for i in t:
                phi[i] += share
        return phi

    @property
    def spread_bound(self) -> float:
        """Upper bound on max_S g(S) - min_S g(S).

        Exact by enumeration over the players carrying nonzero Mobius mass
        when that support is small (the expected sparse case); otherwise the
        safe bound sum |m(T)|. A tight spread matters: it sets the range term
        of the residual game's empirical-Bernstein certificate.
        """
        active = [(t, c) for t, c in zip(self.terms, self.coefs) if c != 0.0]
        if not active:
            return 0.0
        support = sorted({i for t, _ in active for i in t})
        if len(support) <= 16:
            pos = {p: b for b, p in enumerate(support)}
            masks = np.arange(1 << len(support), dtype=np.int64)
            vals = np.zeros(masks.shape[0])
            for t, c in active:
                t_mask = sum(1 << pos[i] for i in t)
                vals += c * ((masks & t_mask) == t_mask)
            return float(vals.max() - vals.min())
        return float(sum(abs(c) for _, c in active))


def fit_mobius_surrogate(
    game: NoisyGame,
    order: int,
    budget_calls: int,
    replicates: int = 1,
    alpha: float = 1e-3,
    rng: Optional[np.random.Generator | int] = None,
) -> MobiusSurrogate:
    """Fit a k-additive surrogate from uniform-size random coalitions."""
    n = game.n
    rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    terms = [
        frozenset(t)
        for k in range(1, order + 1)
        for t in itertools.combinations(range(n), k)
    ]
    n_coalitions = max(budget_calls // replicates, 2)

    coalitions = []
    ys = []
    # Anchor with the empty and grand coalitions, then uniform-size sampling.
    fixed = [frozenset(), frozenset(range(n))]
    for idx in range(n_coalitions):
        if idx < len(fixed):
            s = fixed[idx]
        else:
            size = int(rng.integers(0, n + 1))
            s = frozenset(rng.choice(n, size=size, replace=False).tolist())
        coalitions.append(s)
        ys.append(float(np.mean(game.evaluate(s, replicates))))

    x = np.zeros((len(coalitions), len(terms)))
    for row, s in enumerate(coalitions):
        for col, t in enumerate(terms):
            if t <= s:
                x[row, col] = 1.0
    model = Lasso(alpha=alpha, fit_intercept=True, max_iter=50_000)
    model.fit(x, np.array(ys))
    return MobiusSurrogate(
        n=n,
        order=order,
        terms=terms,
        coefs=model.coef_.copy(),
        intercept=float(model.intercept_),
    )


def surrogate_shapley(
    game: NoisyGame,
    budget_calls: int,
    order: int = 2,
    fit_frac: float = 0.3,
    replicates: int = 1,
    delta: float = 0.05,
    alpha: float = 1e-3,
    ci_method: str = "betting",
    rng: Optional[np.random.Generator | int] = None,
) -> ShapleyResult:
    """phi(v) = phi(g) + CC-Bernstein estimate of phi(v - g), with valid CIs."""
    rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    start_calls = game.calls

    surrogate = fit_mobius_surrogate(
        game,
        order=order,
        budget_calls=int(budget_calls * fit_frac),
        replicates=replicates,
        alpha=alpha,
        rng=rng,
    )
    residual_game = TransformedGame(
        game, shift_fn=surrogate.predict, extra_range=surrogate.spread_bound
    )
    remaining = budget_calls - (game.calls - start_calls)
    res = cc_shapley(
        residual_game,
        budget_calls=remaining,
        replicates=replicates,
        delta=delta,
        ci_method=ci_method,
        rng=rng,
    )
    return ShapleyResult(
        values=surrogate.shapley + res.values,
        halfwidths=res.halfwidths,
        calls=game.calls - start_calls,
        meta={
            "method": "surrogate_cc",
            "order": order,
            "nonzero_terms": int(np.count_nonzero(surrogate.coefs)),
            "surrogate_shapley": surrogate.shapley.tolist(),
            "residual_meta": res.meta,
        },
        lower_bounds=surrogate.shapley + res.lower,
        upper_bounds=surrogate.shapley + res.upper,
    )
