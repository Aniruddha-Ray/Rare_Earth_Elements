import numpy as np
import pytest

from geompm.geology import build_spatial_prior, clr, fit_geochem_pca, fit_power_law, idw_interpolate, power_law_prior


@pytest.mark.parametrize("method", ["poisson", "lsq"])
def test_power_law_recovers_negative_k(method):
    rng = np.random.default_rng(0)
    d = rng.uniform(0.1, 20, 200_000)
    p = 0.05 * d**-1.5
    pos = rng.random(d.size) < np.clip(p, 0, 1)
    fit = fit_power_law(d, pos, n_bins=8, method=method)
    assert fit.k == pytest.approx(-1.5, abs=0.25)


def test_poisson_uses_empty_bins():
    """With few positives, far bins are empty; the Poisson fit must still see the decay."""
    rng = np.random.default_rng(3)
    d = np.concatenate([np.zeros(5000), rng.uniform(0.05, 5, 15000)])
    pos = np.zeros(d.size, bool)
    pos[rng.choice(np.nonzero(d < 0.6)[0], 6, replace=False)] = True
    assert fit_power_law(d, pos, n_bins=6, method="poisson").k < 0


def test_prior_is_unit_scaled_and_decays():
    rng = np.random.default_rng(1)
    d = rng.uniform(0.1, 20, 50_000)
    fit = fit_power_law(d, rng.random(d.size) < 0.02 * d**-1)
    grid = np.linspace(0, 20, 50)
    w = power_law_prior(grid, fit)
    assert w.max() == pytest.approx(1.0) and w.min() >= 0
    assert np.all(np.diff(w) <= 1e-7)


def test_build_spatial_prior_shapes():
    H, W = 30, 40
    yy, xx = np.mgrid[:H, :W]
    dist = {"fault": np.abs(xx - 20).astype(np.float32) * 0.2}
    valid = np.ones((H, W), bool)
    deposits = np.array([[5, 20], [15, 21], [25, 19], [10, 23]])
    w, fits = build_spatial_prior(dist, deposits, valid, n_bins=4)
    assert w.shape == (H, W) and 0 <= w.min() and w.max() <= 1
    assert fits["fault"].k < 0
    assert w[0, 20] > w[0, 0]


def test_clr_rows_sum_to_zero():
    X = np.random.default_rng(0).uniform(1, 100, (20, 5))
    assert np.allclose(clr(X).sum(1), 0)
    with pytest.raises(ValueError):
        clr(np.array([[1.0, 0.0]]))


def test_pca_selects_and_orients_indicator_component():
    rng = np.random.default_rng(0)
    n = 500
    ree = rng.lognormal(0, 1, n)
    X = np.column_stack([ree * 50, ree * 30, rng.lognormal(3, 0.3, n), rng.lognormal(2, 0.3, n), rng.lognormal(4, 0.3, n)])
    els = ["La", "Y", "Zn", "Cu", "Zr"]
    gp = fit_geochem_pca(X, els, ("La", "Y"))
    s = gp.transform(X)
    assert np.corrcoef(s, np.log(ree))[0, 1] > 0.8


def test_idw_exact_at_sources():
    src = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    vals = np.array([1.0, 2.0, 3.0])
    assert np.allclose(idw_interpolate(src, vals, src, k=3), vals)
    mid = idw_interpolate(src, vals, np.array([[0.3, 0.3]]), k=3)
    assert 1.0 < mid[0] < 3.0
