"""Hard-constraint helpers: multiplicative modulation of features by a geological prior."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def soften(prior: torch.Tensor, floor: float) -> torch.Tensor:
    """Map a prior in [0, 1] to [floor, 1].

    A pure multiplication by a prior that reaches 0 kills both the activation and its gradient. The
    floor keeps unfavourable regions suppressed but still trainable. ``floor=0`` recovers F * w.
    """
    return floor + (1.0 - floor) * prior


def resize_prior(prior: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """Resize a (B, 1, H, W) prior to a feature map's spatial size (area averaging when shrinking)."""
    if prior.shape[-2:] == tuple(size):
        return prior
    if prior.shape[-2] >= size[0] and prior.shape[-1] >= size[1]:
        return F.adaptive_avg_pool2d(prior, size)
    return F.interpolate(prior, size=size, mode="bilinear", align_corners=False)


def modulate_spatial(features: torch.Tensor, prior: torch.Tensor, floor: float) -> torch.Tensor:
    """Hard spatial constraint F_g(x) = w * F(x), with w resized to the feature map (B, C, H, W)."""
    return features * soften(resize_prior(prior, features.shape[-2:]), floor)


def modulate_vector(features: torch.Tensor, prior: torch.Tensor, floor: float) -> torch.Tensor:
    """Hard classifier constraint C_g(x) = g * l(x) with one scalar prior per sample (B,) or token (B, N)."""
    while prior.dim() < features.dim():
        prior = prior.unsqueeze(-1)
    return features * soften(prior, floor)
