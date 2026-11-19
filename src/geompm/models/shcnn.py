"""Soft + Hard constrained CNN (SHCNN).

    x -> [Conv -> BN -> ReLU -> * w] x n -> MaxPool -> Flatten -> FC1 -> ReLU -> * g -> FC2 -> logit

* ``w`` (B, 1, P, P): spatial prior patch aligned with the input patch (hard spatial constraint).
* ``g_center`` (B,): geochemical prior at the sample location (hard classifier constraint).
* Returns **logits**; apply ``torch.sigmoid`` for probabilities. The soft constraint lives in GeoLoss.

Fixes relative to the notebook: a single random 32x32 ``w`` was shared by every sample (now one
patch per sample); ``g[:768]`` sliced a random 16384-vector (now a per-sample scalar); the flatten
size was hard-coded to 64*16*16 (now derived from the patch size); the model returned sigmoid
outputs that were then passed to a with-logits loss.

``use_w=False, use_g=False`` gives the plain CNN baseline with an identical architecture.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .modulation import modulate_spatial, modulate_vector


class SHCNN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        patch_size: int = 32,
        channels: tuple[int, ...] = (32, 64),
        hidden: int = 128,
        use_w: bool = True,
        use_g: bool = True,
        prior_floor: float = 0.1,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.use_w, self.use_g, self.prior_floor = use_w, use_g, prior_floor
        self.convs = nn.ModuleList()
        c_in = in_channels
        for c_out in channels:
            self.convs.append(
                nn.Sequential(nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True))
            )
            c_in = c_out
        self.pool = nn.MaxPool2d(2)
        flat = c_in * (patch_size // 2) ** 2
        self.fc1 = nn.Linear(flat, hidden)
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor, w: torch.Tensor, g_patch: torch.Tensor, g_center: torch.Tensor) -> torch.Tensor:
        """``g_patch`` is accepted for a uniform model interface; SHCNN only uses ``g_center``."""
        g = g_center
        for conv in self.convs:
            x = conv(x)
            if self.use_w:
                x = modulate_spatial(x, w, self.prior_floor)
        x = self.pool(x).flatten(1)
        h = torch.relu(self.fc1(x))
        if self.use_g:
            h = modulate_vector(h, g, self.prior_floor)
        return self.fc2(self.drop(h)).squeeze(-1)
