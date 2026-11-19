"""Spatial geological prior ``w`` from a distance-decay (power-law) relationship.

Mineralisation density ``m`` at distance ``d`` from an ore-controlling factor is modelled as

    m(d) = N * d**k          (log m = log N + k log d)

``N`` and ``k`` are *fitted* by least squares on log-log binned densities (``k`` is expected to be
negative for a controlling factor). With few deposits most distance bins are empty, so the default
estimator is unit-level Poisson maximum likelihood (no binning) rather than least squares on binned
log density (which has to discard the empty bins). The prior is then

    w = N * d**k / m_max     with m_max = max over the map, so w is in (0, 1].

Fixes relative to the original notebook: ``k`` was hard-coded to +0.95 (density *increasing* with
distance) and distances were measured in raw degrees. Here ``k`` is estimated from data, distances
are in km, and the fit only ever sees the deposits it is given (pass training deposits only to keep
cross-validation leak-free).
"""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class PowerLawFit:
    N: float
    k: float
    d_min: float
    bin_centers: np.ndarray
    density: np.ndarray
    n_positive: int

    def to_dict(self) -> dict:
        d = asdict(self)
        d["bin_centers"] = self.bin_centers.tolist()
        d["density"] = self.density.tolist()
        return d


def _poisson_fit(d: np.ndarray, is_positive: np.ndarray, k_bounds=(-5.0, 5.0)) -> tuple[float, float]:
    """Unit-level maximum likelihood for y_i ~ Poisson(N d_i^k) -> (k, log N). No binning.

    For fixed k the MLE of N is n_pos / sum_i d_i^k, so the likelihood is profiled to one dimension:
        l(k) = k * sum_pos log d_i - n_pos * log(sum_i d_i^k)   (+ const)
    """
    from scipy.optimize import minimize_scalar

    log_d = np.log(d)
    n_pos = is_positive.sum()
    s_pos = log_d[is_positive].sum()

    def neg_profile(k):
        z = k * log_d
        zmax = z.max()
        return -(k * s_pos - n_pos * (zmax + np.log(np.exp(z - zmax).sum())))

    k = minimize_scalar(neg_profile, bounds=k_bounds, method="bounded").x
    z = k * log_d
    zmax = z.max()
    logN = np.log(n_pos) - (zmax + np.log(np.exp(z - zmax).sum()))
    return float(k), float(logN)


def fit_power_law(
    distances: np.ndarray,
    is_positive: np.ndarray,
    n_bins: int = 6,
    d_min: float | None = None,
    method: str = "poisson",
) -> PowerLawFit:
    """Fit ``log m = log N + k log d`` from distances of all units (cells or samples).

    Parameters
    ----------
    distances : distance (km) of every unit to the controlling factor.
    is_positive : boolean, True where the unit is mineralised (deposit cell / anomalous sample).
    n_bins : number of distance bins (cells on the factor, d = 0, form their own bin; the rest are
        equal-population quantile bins). Density in a bin is the fraction of positive units, i.e.
        deposits per cell for a regular grid.
    d_min : floor applied to distances (avoids log(0) on the factor itself). Defaults to half the
        smallest positive distance in the data, or 0.05 km.
    method : ``poisson`` (default) - unit-level maximum likelihood of the same power-law model; no
        binning, so it uses every cell including the many with no deposit; ``lsq`` - least squares on
        binned log density (the textbook log-log fit), which must drop empty bins and is biased when
        deposits are few.
    """
    distances = np.asarray(distances, dtype=float).ravel()
    is_positive = np.asarray(is_positive, dtype=bool).ravel()
    ok = np.isfinite(distances)
    distances, is_positive = distances[ok], is_positive[ok]
    n_pos = int(is_positive.sum())
    if n_pos == 0:
        raise ValueError("fit_power_law needs at least one positive unit")

    nonzero = distances[distances > 0]
    if d_min is None:
        d_min = float(0.5 * nonzero.min()) if nonzero.size else 0.05
    d = np.maximum(distances, d_min)

    # Cells *on* the factor (d = 0, e.g. inside an intrusion) form their own bin; quantile bins on the
    # rest. Without this, zero-inflated distances collapse the quantile edges into very few bins.
    on_factor = distances <= 0
    groups = [on_factor] if on_factor.any() else []
    n_rest = n_bins - len(groups)
    if nonzero.size and n_rest > 0:
        edges = np.unique(np.quantile(nonzero, np.linspace(0, 1, n_rest + 1)))
        idx = np.clip(np.searchsorted(edges, distances, side="right") - 1, 0, len(edges) - 2)
        groups += [(~on_factor) & (idx == b) for b in range(len(edges) - 1)]
    centers, dens = [], []  # binned densities: used by 'lsq' and kept for reporting / plots
    for in_bin in groups:
        if in_bin.sum() == 0:
            continue
        centers.append(np.exp(np.log(d[in_bin]).mean()))  # geometric mean distance
        dens.append(is_positive[in_bin].mean())
    centers, dens = np.asarray(centers), np.asarray(dens)

    use = dens > 0
    if method == "poisson":
        k, logN = _poisson_fit(d, is_positive)
    elif method != "lsq":
        raise ValueError(f"Unknown method '{method}' (poisson | lsq)")
    elif use.sum() >= 2:
        k, logN = np.polyfit(np.log(centers[use]), np.log(dens[use]), 1)
    else:
        # All positives fall in one bin: decay cannot be estimated. Use k = -1 anchored at that bin.
        warnings.warn("Only one non-empty density bin; falling back to k = -1 (inverse distance).")
        k = -1.0
        c, m = centers[use][0], dens[use][0]
        logN = np.log(m) - k * np.log(c)
    if k >= 0:
        warnings.warn(
            f"Fitted k = {k:.3f} >= 0: mineralisation does not decay with distance to this factor; "
            "it may not be ore-controlling at this scale."
        )
    return PowerLawFit(float(np.exp(logN)), float(k), float(d_min), centers, dens, n_pos)


def power_law_prior(distance: np.ndarray, fit: PowerLawFit, valid: np.ndarray | None = None) -> np.ndarray:
    """Evaluate ``w = N d^k / m_max`` on a distance array; output is in (0, 1]."""
    d = np.maximum(np.asarray(distance, dtype=float), fit.d_min)
    m = fit.N * d**fit.k
    ref = m[valid] if valid is not None else m[np.isfinite(m)]
    w = m / np.nanmax(ref)
    w = np.clip(w, 0.0, 1.0)
    if valid is not None:
        w = np.where(valid, w, 0.0)
    return w.astype(np.float32)


def combine_priors(priors: list[np.ndarray], method: str = "mean") -> np.ndarray:
    """Combine per-factor priors (each in [0, 1]) into a single w map."""
    if not priors:
        raise ValueError("No priors to combine")
    stack = np.stack(priors)
    if method == "mean":
        out = stack.mean(0)
    elif method == "max":
        out = stack.max(0)
    elif method == "product":
        out = stack.prod(0)
    else:
        raise ValueError(f"Unknown combine method '{method}'")
    return out.astype(np.float32)


def build_spatial_prior(
    distance_maps: dict[str, np.ndarray],
    deposit_rc: np.ndarray,
    valid: np.ndarray,
    n_bins: int = 6,
    combine: str = "mean",
    drop_nonnegative_k: bool = True,
) -> tuple[np.ndarray, dict[str, PowerLawFit]]:
    """Fit one power law per controlling factor from the given deposits and build the w map.

    Factors whose fitted k >= 0 (no distance decay) are left out of w when ``drop_nonnegative_k``;
    if none remain, w is uniform (= 1 inside the study area), i.e. the hard spatial constraint is
    neutral. Check ``fit.k`` in the returned fits to see which factors were used.
    """
    deposit_mask = np.zeros(valid.shape, dtype=bool)
    deposit_mask[deposit_rc[:, 0], deposit_rc[:, 1]] = True
    fits, priors = {}, []
    for name, dist in distance_maps.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = fit_power_law(dist[valid], deposit_mask[valid], n_bins=n_bins)
        fits[name] = fit
        if drop_nonnegative_k and fit.k >= 0:
            continue
        priors.append(power_law_prior(dist, fit, valid))
    if not priors:
        warnings.warn("No controlling factor shows distance decay; using a uniform w prior.")
        return valid.astype(np.float32), fits
    return combine_priors(priors, combine), fits
