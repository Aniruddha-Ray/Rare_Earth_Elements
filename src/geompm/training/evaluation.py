"""Evaluation for sparse-label prospectivity maps.

With only a handful of known occurrences, point classification metrics on a random split are
unstable and leak through spatial autocorrelation. The primary metrics here are map-based:

* ``area_rank``          fraction of the study area scored >= the score at a (held-out) deposit.
                         0.02 means the deposit falls in the top 2 % of the map. Lower is better;
                         a random map gives 0.5 on average.
* ``success_rate_curve`` captured-deposit fraction vs. area fraction (the standard MPM curve) and its AUC.
* ``roc_auc`` / ``pr_auc`` deposit cells vs. background cells, for completeness.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def area_rank(prob_map: np.ndarray, valid: np.ndarray, rc: np.ndarray) -> np.ndarray:
    scores = prob_map[valid]
    return np.array([(scores >= prob_map[r, c]).mean() for r, c in np.atleast_2d(rc)])


def success_rate_curve(prob_map: np.ndarray, valid: np.ndarray, rc: np.ndarray, n: int = 101):
    """Returns (area_fraction, captured_fraction, auc)."""
    ranks = np.sort(area_rank(prob_map, valid, rc))
    area = np.linspace(0, 1, n)
    captured = np.array([(ranks <= a).mean() for a in area])
    return area, captured, float(np.trapezoid(captured, area))


def captured_at(prob_map: np.ndarray, valid: np.ndarray, rc: np.ndarray, area_fraction: float) -> float:
    return float((area_rank(prob_map, valid, rc) <= area_fraction).mean())


def cell_auc(prob_map: np.ndarray, valid: np.ndarray, rc: np.ndarray) -> dict:
    y = np.zeros(prob_map.shape, dtype=bool)
    y[rc[:, 0], rc[:, 1]] = True
    yt, ys = y[valid], prob_map[valid]
    return {"roc_auc": float(roc_auc_score(yt, ys)), "pr_auc": float(average_precision_score(yt, ys))}
