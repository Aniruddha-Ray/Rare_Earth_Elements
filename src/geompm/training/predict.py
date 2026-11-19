"""Dense prospectivity map by sliding-window patch classification over every valid cell."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..data.patches import PatchDataset
from .trainer import forward_batch, resolve_device


@torch.no_grad()
def predict_map(
    model: torch.nn.Module,
    stack: np.ndarray,
    w_map: np.ndarray,
    g_map: np.ndarray,
    valid: np.ndarray,
    patch_size: int,
    batch_size: int = 512,
    device: str = "auto",
) -> np.ndarray:
    """Probability map (H, W); NaN outside the study area."""
    dev = resolve_device(device)
    model.to(dev).eval()
    centers = np.column_stack(np.nonzero(valid))
    ds = PatchDataset(stack, w_map, g_map, centers, None, patch_size, augment=False)
    probs = []
    for batch in DataLoader(ds, batch_size=batch_size, shuffle=False):
        probs.append(torch.sigmoid(forward_batch(model, batch, dev)).float().cpu().numpy())
    out = np.full(valid.shape, np.nan, dtype=np.float32)
    out[centers[:, 0], centers[:, 1]] = np.concatenate(probs)
    return out
