"""Geochemical prior ``g``: a mineralisation-related principal component mapped onto the grid.

Pipeline: centred log-ratio (CLR) transform of the multi-element composition -> standardise -> PCA
-> choose the component whose loadings are dominated by the indicator elements (e.g. La, Y, Th for
REE) -> orient it so high values mean "more mineralisation-like" -> inverse-distance interpolate to
the grid -> robustly rescale to [0, 1].

Fixes relative to the original code: ``g`` was a random unit vector of length 16384 multiplied
element-wise with fully-connected features (so it carried no spatial meaning and shrank every
activation by ~100x). Here ``g`` is a map aligned with the input grid; models consume it per sample
(classifier) or per token (ViT FFN). The PCA uses geochemistry only - no deposit labels - so it
cannot leak the target.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def clr(X: np.ndarray) -> np.ndarray:
    """Centred log-ratio transform (rows are compositions, all values must be > 0)."""
    X = np.asarray(X, dtype=float)
    if np.any(X <= 0) or not np.all(np.isfinite(X)):
        raise ValueError("CLR requires strictly positive, finite values; impute below-detection values first")
    L = np.log(X)
    return L - L.mean(axis=1, keepdims=True)


@dataclass
class GeochemPCA:
    elements: list[str]
    scaler: StandardScaler
    pca: PCA
    component: int = 0
    sign: float = 1.0
    indicator_scores: dict[str, float] = field(default_factory=dict)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Score of the selected component for compositions X (n, n_elements)."""
        Z = self.scaler.transform(clr(X))
        return self.sign * self.pca.transform(Z)[:, self.component]

    def loadings(self) -> dict[str, float]:
        return dict(zip(self.elements, (self.sign * self.pca.components_[self.component]).round(3).tolist()))

    def summary(self) -> dict:
        return {
            "component": int(self.component),
            "sign": float(self.sign),
            "explained_variance_ratio": float(self.pca.explained_variance_ratio_[self.component]),
            "loadings": self.loadings(),
            "indicator_scores": self.indicator_scores,
        }


def fit_geochem_pca(
    X: np.ndarray,
    elements: list[str],
    indicators: tuple[str, ...] | list[str],
    n_components: int = 8,
) -> GeochemPCA:
    """Fit CLR-PCA and pick the component most associated with the indicator elements.

    The selection score of a component is the sum of the squared indicator loadings weighted by the
    component's explained variance; the sign is chosen so the indicators load positively.
    """
    missing = [e for e in indicators if e not in elements]
    if missing:
        raise KeyError(f"Indicator elements {missing} not in {elements}")
    scaler = StandardScaler()
    Z = scaler.fit_transform(clr(X))
    pca = PCA(n_components=min(n_components, Z.shape[1])).fit(Z)
    ind = [elements.index(e) for e in indicators]
    load = pca.components_[:, ind]
    scores = (load**2).sum(1) * pca.explained_variance_ratio_
    comp = int(np.argmax(scores))
    sign = 1.0 if load[comp].sum() >= 0 else -1.0
    return GeochemPCA(
        list(elements), scaler, pca, comp, sign, {f"PC{i + 1}": float(s) for i, s in enumerate(scores)}
    )


def idw_interpolate(
    xy_src: np.ndarray, values: np.ndarray, xy_query: np.ndarray, k: int = 12, power: float = 2.0
) -> np.ndarray:
    """Inverse-distance-weighted interpolation (coordinates must be in a metric space, e.g. km)."""
    tree = cKDTree(xy_src)
    k = min(k, len(xy_src))
    dist, idx = tree.query(xy_query, k=k)
    if k == 1:
        dist, idx = dist[:, None], idx[:, None]
    exact = dist[:, 0] < 1e-12
    wts = 1.0 / np.maximum(dist, 1e-12) ** power
    out = (wts * values[idx]).sum(1) / wts.sum(1)
    out[exact] = values[idx[exact, 0]]
    return out


def robust_unit_scale(x: np.ndarray, lo_q: float = 2.0, hi_q: float = 98.0, mask: np.ndarray | None = None) -> np.ndarray:
    """Rescale to [0, 1] using percentiles (robust to outliers)."""
    ref = x[mask] if mask is not None else x[np.isfinite(x)]
    lo, hi = np.percentile(ref, [lo_q, hi_q])
    return np.clip((x - lo) / max(hi - lo, 1e-12), 0.0, 1.0).astype(np.float32)
