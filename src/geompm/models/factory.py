"""Build any model of the comparison grid from a ModelConfig.

All models share one interface:  ``logits = model(x, w, g_patch, g_center)``
    x        (B, C, P, P) evidence-layer patch
    w        (B, 1, P, P) spatial prior patch
    g_patch  (B, 1, P, P) geochemical prior patch
    g_center (B,)         geochemical prior at the patch centre

| name     | architecture | hard w (conv) | geo attention | g (FFN) | g (head) |
|----------|--------------|---------------|---------------|---------|----------|
| cnn      | CNN          | -             | -             | -       | -        |
| shcnn    | CNN          | yes           | -             | -       | yes      |
| vit      | ViT          | -             | none          | -       | -        |
| geovit   | ViT          | -             | cfg.affinity  | yes     | yes      |
| cnn_vit  | CNN+ViT      | -             | none          | -       | -        |
| shvcnn   | CNN+ViT      | yes           | cfg.affinity  | yes     | yes      |

The soft constraint is independent of the architecture: set ``TrainConfig.geo_lambda`` (0 = off).
"""

from __future__ import annotations

import torch.nn as nn

from ..config import ModelConfig
from .geology_vit import GeoViT
from .shcnn import SHCNN
from .shvcnn import SHVCNN

MODEL_NAMES = ("cnn", "shcnn", "vit", "geovit", "cnn_vit", "shvcnn")


def build_model(cfg: ModelConfig, in_channels: int) -> nn.Module:
    n = cfg.name
    if n in ("cnn", "shcnn"):
        hard = n == "shcnn"
        return SHCNN(in_channels, cfg.patch_size, cfg.cnn_channels, cfg.hidden, hard, hard, cfg.prior_floor, cfg.dropout)
    if n in ("vit", "geovit"):
        geo = n == "geovit"
        return GeoViT(
            in_channels, cfg.patch_size, cfg.vit_patch, cfg.stage_channels[-1], sum(cfg.stage_depths),
            cfg.num_heads, cfg.mlp_ratio, cfg.dropout, cfg.affinity if geo else "none", geo,
            cfg.prior_floor, cfg.hidden,
        )
    if n in ("cnn_vit", "shvcnn"):
        geo = n == "shvcnn"
        return SHVCNN(
            in_channels, cfg.patch_size, cfg.stage_channels, cfg.stage_depths, cfg.num_heads, cfg.mlp_ratio,
            cfg.dropout, cfg.affinity if geo else "none", geo, geo, geo, cfg.prior_floor, cfg.hidden,
        )
    raise ValueError(f"Unknown model '{n}'. Choose from {MODEL_NAMES}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
