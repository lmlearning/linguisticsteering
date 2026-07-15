"""Shared evaluation metrics for the common experimental design."""

from __future__ import annotations

import numpy as np
from scipy.stats import kendalltau


def linf_error(estimate: np.ndarray, truth: np.ndarray) -> float:
    return float(np.max(np.abs(estimate - truth)))


def l2_error(estimate: np.ndarray, truth: np.ndarray) -> float:
    return float(np.linalg.norm(estimate - truth))


def kendall_tau(estimate: np.ndarray, truth: np.ndarray) -> float:
    tau, _ = kendalltau(estimate, truth)
    return float(tau)


def ci_coverage(lower: np.ndarray, upper: np.ndarray, truth: np.ndarray) -> float:
    """Fraction of players whose true value lies in the reported interval.

    The simultaneous guarantee is that coverage == 1.0 with prob >= 1 - delta,
    so the quantity to check across seeds is P[coverage == 1].
    """
    return float(np.mean((truth >= lower) & (truth <= upper)))


def mean_ci_width(lower: np.ndarray, upper: np.ndarray) -> float:
    width = upper - lower
    finite = width[np.isfinite(width)]
    return float(np.mean(finite)) if finite.size else float("inf")


def topk_precision(estimate: np.ndarray, truth: np.ndarray, k: int) -> float:
    est_top = set(np.argsort(-np.abs(estimate))[:k].tolist())
    true_top = set(np.argsort(-np.abs(truth))[:k].tolist())
    return len(est_top & true_top) / k
