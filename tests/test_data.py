from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from geompm.config import CurnamonaConfig
from geompm.data.brs import BRSFeatures, parse_censored, parse_dms, robust_anomaly_threshold, spatial_blocks
from geompm.data.patches import PatchDataset, build_training_set, negative_pool, positive_centers


def test_parse_censored():
    assert parse_censored("<50") == (25.0, "<")
    assert parse_censored("<50", policy="dl") == (50.0, "<")
    assert parse_censored(">100") == (100.0, ">")
    assert parse_censored(163.0) == (163.0, "")
    assert np.isnan(parse_censored("NA")[0])


def test_parse_dms():
    assert parse_dms('N27°40\' 56.7"') == pytest.approx(27.682417, abs=1e-6)
    assert parse_dms('S27°40\' 56.7"') < 0
    assert np.isnan(parse_dms(None))


def test_patch_alignment_under_augmentation():
    """x, w and g must undergo the same rotation/flip."""
    H = W = 20
    w = np.random.default_rng(0).random((H, W)).astype(np.float32)
    stack = w[None].copy()
    ds = PatchDataset(stack, w, w, np.array([[10, 10], [3, 15]]), np.array([1, 0]), patch_size=8, augment=True, seed=1)
    for _ in range(10):
        for i in range(len(ds)):
            s = ds[i]
            assert torch.equal(s["x"], s["w"]) and torch.equal(s["w"], s["g_patch"])


def test_patch_is_centred():
    H = W = 16
    arr = np.zeros((H, W), np.float32)
    arr[7, 9] = 1.0
    ds = PatchDataset(arr[None], arr, arr, np.array([[7, 9]]), None, patch_size=8)
    s = ds[0]
    assert s["x"][0, 4, 4] == 1.0 and s["w_center"] == 1.0


def test_sampling_respects_buffer():
    valid = np.ones((50, 50), bool)
    deps = np.array([[25, 25]])
    pos = positive_centers(deps, valid, 1)
    assert len(pos) == 5
    neg = negative_pool(deps, valid, (0.2, 0.2), buffer_km=2.0)
    assert np.all(np.hypot(neg[:, 0] - 25, neg[:, 1] - 25) * 0.2 > 2.0)
    ds, counts = build_training_set(np.zeros((2, 50, 50)), np.zeros((50, 50)), np.zeros((50, 50)), valid, deps, (0.2, 0.2), 8, neg_per_pos=3)
    assert counts["n_neg"] == 3 * counts["n_pos"]


@pytest.mark.filterwarnings("ignore:Fitted k")  # random synthetic data has no distance decay
def test_brs_features_exclude_target_and_fit_on_train_only():
    rng = np.random.default_rng(0)
    n = 300
    df = pd.DataFrame({
        "latitude": 27.68 + rng.random(n) * 0.04, "longitude": 75.90 + rng.random(n) * 0.03,
        "Cu": rng.lognormal(3.5, 1.2, n), "Zn": rng.lognormal(4, 0.5, n),
        "Ni": rng.lognormal(3.7, 0.3, n), "Co": rng.lognormal(3.2, 0.2, n),
    })
    thr = robust_anomaly_threshold(df["Cu"])
    feats = BRSFeatures("Cu", thr, ["Zn", "Ni", "Co"]).fit(df.iloc[:200])
    out = feats.transform(df.iloc[200:])
    assert set(out.columns) == {"w_proximity", "g_pc1", "distance_km"}
    assert out["w_proximity"].between(0, 1).all()
    assert len(np.unique(spatial_blocks(df.latitude, df.longitude, 0.5))) > 4


DATA = Path(CurnamonaConfig().root)


@pytest.mark.skipif(not DATA.exists(), reason="Curnamona data not downloaded")
def test_prepare_curnamona():
    from geompm.data.curnamona import prepare_curnamona

    gg = prepare_curnamona(verbose=False)
    C, H, W = gg.stack.shape
    assert (H, W) == gg.valid.shape == gg.g_map.shape
    assert C == len(gg.channel_names) == 35  # 12 mag + 3 grav + 4 rad + 15 ASTER + DEM
    assert len(gg.deposits_rc) == 7
    assert gg.valid[gg.deposits_rc[:, 0], gg.deposits_rc[:, 1]].all()
    assert np.isfinite(gg.stack).all() and 0 <= gg.g_map.min() and gg.g_map.max() <= 1
    for d in gg.distances.values():
        assert d.min() == 0 and np.isfinite(d).all()
