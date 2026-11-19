"""Soft + Hard constrained hybrid CNN-ViT (SHVCNN).

    x -> for each stage s:
           CNN block (Conv-BN-ReLU x2, MaxPool/2)  -> * w            (hard spatial constraint)
           GeoViTStage: attention + alpha*M(w), FFN * g             (geology-aware global context)
      -> global average pool -> FC1 -> ReLU -> * g(center) -> FC2 -> logit

Patch 32 with 3 stages gives token grids 16x16 -> 8x8 -> 4x4 (local precision early, global context late).

Fixes relative to the notebook: ``forward`` called the *global* ``vit_model`` instead of
``self.vit_model``, so the optimiser (which held ``model_shvcnn.parameters()``) only ever updated the
final 768->1 layer; the same ViT was applied twice with its 3072-d "classification" output reshaped
into an image; the head multiplied by a random ``g_data[:768]`` and applied sigmoid before a
with-logits loss. Every component is now a registered submodule and gradients reach all parameters
(see tests/test_models.py).

Flags give the ablation grid: ``use_w_conv``, ``affinity`` (none/product/learned), ``use_g_ffn``,
``use_g_head``; all off = plain CNN+ViT baseline.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .geology_vit import GeoViTStage
from .modulation import modulate_spatial, modulate_vector


def cnn_block(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class SHVCNN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        patch_size: int = 32,
        stage_channels: tuple[int, ...] = (32, 64, 128),
        stage_depths: tuple[int, ...] = (1, 1, 1),
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1,
        affinity: str = "product",
        use_w_conv: bool = True,
        use_g_ffn: bool = True,
        use_g_head: bool = True,
        prior_floor: float = 0.1,
        hidden: int = 128,
    ):
        super().__init__()
        if len(stage_channels) != len(stage_depths):
            raise ValueError("stage_channels and stage_depths must have the same length")
        if patch_size % (2 ** len(stage_channels)):
            raise ValueError(f"patch_size {patch_size} must be divisible by 2**{len(stage_channels)}")
        self.use_w_conv, self.use_g_head, self.prior_floor = use_w_conv, use_g_head, prior_floor
        self.cnn, self.vit = nn.ModuleList(), nn.ModuleList()
        c_in, size = in_channels, patch_size
        for c, depth in zip(stage_channels, stage_depths):
            size //= 2
            self.cnn.append(cnn_block(c_in, c))
            self.vit.append(
                GeoViTStage(c, c, (size, size), depth, num_heads, mlp_ratio, dropout, affinity, use_g_ffn, prior_floor)
            )
            c_in = c
        self.fc1, self.fc2 = nn.Linear(c_in, hidden), nn.Linear(hidden, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, w: torch.Tensor, g_patch: torch.Tensor, g_center: torch.Tensor) -> torch.Tensor:
        for cnn, vit in zip(self.cnn, self.vit):
            x = cnn(x)
            if self.use_w_conv:
                x = modulate_spatial(x, w, self.prior_floor)
            x = x + vit(x, w, g_patch)  # residual keeps the local CNN features
        h = torch.relu(self.fc1(x.mean((2, 3))))
        if self.use_g_head:
            h = modulate_vector(h, g_center, self.prior_floor)
        return self.fc2(self.drop(h)).squeeze(-1)
