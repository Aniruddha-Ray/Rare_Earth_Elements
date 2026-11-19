"""Patch sampling and datasets for patch-based prospectivity classification.

Each sample is a P x P window centred on a grid cell:
    x (C, P, P), w (1, P, P), g_patch (1, P, P), g_center (), w_center (), y ()

Labelling follows the usual positive-unlabelled convention in MPM (and DEEP-SEAM): positives are cells
at / next to known occurrences; "negatives" are random cells farther than a buffer from the *training*
occurrences. They are unlabelled cells assumed barren, not confirmed barren ground.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


def disk_offsets(radius: int) -> np.ndarray:
    r = np.arange(-radius, radius + 1)
    dy, dx = np.meshgrid(r, r, indexing="ij")
    keep = dy**2 + dx**2 <= radius**2
    return np.column_stack([dy[keep], dx[keep]])


def positive_centers(deposits_rc: np.ndarray, valid: np.ndarray, radius: int = 1) -> np.ndarray:
    cells = (deposits_rc[:, None, :] + disk_offsets(radius)[None]).reshape(-1, 2)
    H, W = valid.shape
    ok = (cells[:, 0] >= 0) & (cells[:, 0] < H) & (cells[:, 1] >= 0) & (cells[:, 1] < W)
    cells = cells[ok]
    cells = cells[valid[cells[:, 0], cells[:, 1]]]
    return np.unique(cells, axis=0)


def negative_pool(deposits_rc: np.ndarray, valid: np.ndarray, pixel_km: tuple[float, float], buffer_km: float) -> np.ndarray:
    """All valid cells farther than ``buffer_km`` from every given deposit."""
    rows, cols = np.nonzero(valid)
    far = np.ones(rows.shape, dtype=bool)
    for r, c in deposits_rc:
        d = np.hypot((rows - r) * pixel_km[0], (cols - c) * pixel_km[1])
        far &= d > buffer_km
    return np.column_stack([rows[far], cols[far]])


class PatchDataset(Dataset):
    """Crops patches on the fly from padded full-grid arrays (memory = one copy of the grid)."""

    def __init__(
        self,
        stack: np.ndarray,
        w_map: np.ndarray,
        g_map: np.ndarray,
        centers: np.ndarray,
        labels: np.ndarray | None,
        patch_size: int = 32,
        augment: bool = False,
        seed: int = 0,
    ):
        if patch_size % 2:
            raise ValueError("patch_size must be even")
        self.P, self.h = patch_size, patch_size // 2
        pad = ((self.h, self.h), (self.h, self.h))
        # Outside the grid: evidence = 0 (the standardised median), priors = 0 (no support).
        self.stack = torch.from_numpy(np.pad(stack, ((0, 0), *pad)).astype(np.float32))
        self.w = torch.from_numpy(np.pad(w_map, pad).astype(np.float32))[None]
        self.g = torch.from_numpy(np.pad(g_map, pad).astype(np.float32))[None]
        self.w_raw, self.g_raw = w_map, g_map
        self.centers = np.asarray(centers, dtype=int)
        self.labels = None if labels is None else np.asarray(labels, dtype=np.float32)
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.centers)

    def _crop(self, arr: torch.Tensor, r: int, c: int) -> torch.Tensor:
        # padded index of cell (r, c) is (r + h, c + h); window [r, r + P) is centred on it
        return arr[:, r : r + self.P, c : c + self.P]

    def __getitem__(self, i: int) -> dict:
        r, c = self.centers[i]
        x, w, g = self._crop(self.stack, r, c), self._crop(self.w, r, c), self._crop(self.g, r, c)
        if self.augment:  # random dihedral transform, identical for x, w and g
            k = int(self.rng.integers(4))
            flip = bool(self.rng.integers(2))
            x, w, g = (torch.rot90(t, k, (1, 2)) for t in (x, w, g))
            if flip:
                x, w, g = (torch.flip(t, (2,)) for t in (x, w, g))
        out = {
            "x": x.contiguous(),
            "w": w.contiguous(),
            "g_patch": g.contiguous(),
            "g_center": torch.tensor(float(self.g_raw[r, c])),
            "w_center": torch.tensor(float(self.w_raw[r, c])),
        }
        if self.labels is not None:
            out["y"] = torch.tensor(self.labels[i])
        return out


def build_training_set(
    stack, w_map, g_map, valid, deposits_rc, pixel_km, patch_size, pos_radius=1, neg_buffer_km=5.0,
    neg_per_pos=10, augment=True, seed=0,
) -> tuple[PatchDataset, dict]:
    rng = np.random.default_rng(seed)
    pos = positive_centers(deposits_rc, valid, pos_radius)
    pool = negative_pool(deposits_rc, valid, pixel_km, neg_buffer_km)
    n_neg = min(len(pool), neg_per_pos * len(pos))
    neg = pool[rng.choice(len(pool), n_neg, replace=False)]
    centers = np.concatenate([pos, neg])
    labels = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    ds = PatchDataset(stack, w_map, g_map, centers, labels, patch_size, augment, seed)
    return ds, {"n_pos": int(len(pos)), "n_neg": int(len(neg)), "neg_pool": int(len(pool))}
