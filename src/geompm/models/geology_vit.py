"""Geology-aware Vision Transformer components.

* ``GeoFFN``            FFN whose hidden activations are modulated by the token's geochemical prior g.
* ``GeoTransformerBlock`` pre-norm block: geological attention (w) + GeoFFN (g).
* ``GeoViTStage``       turns a feature map into tokens, runs blocks, and maps tokens back to a feature map,
                        so it can sit between CNN stages (hybrid) or after a patch embedding (pure ViT).
* ``GeoViT``            pure ViT baseline: patch embedding -> GeoViTStage -> pooled g-constrained head.

Fixes relative to the notebook: the FFN hidden size was forced to 16384 to match the random g vector
(now ``mlp_ratio * dim``); g is a per-token scalar taken from the g map under each token; the ViT
head output was reshaped into a fake 3x32x32 "image" and fed through the same ViT again.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .geology_attention import GeoMultiHeadAttention, build_affinity
from .modulation import modulate_vector, resize_prior


def token_prior(prior: torch.Tensor, hw: tuple[int, int]) -> torch.Tensor:
    """Average a (B, 1, P, P) prior under each token of an (h, w) token grid -> (B, h*w)."""
    return resize_prior(prior, hw).flatten(1)


class GeoFFN(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 2.0, dropout: float = 0.0, use_g: bool = True, prior_floor: float = 0.1):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1, self.fc2 = nn.Linear(dim, hidden), nn.Linear(hidden, dim)
        self.drop = nn.Dropout(dropout)
        self.use_g, self.prior_floor = use_g, prior_floor

    def forward(self, x: torch.Tensor, g_tok: torch.Tensor | None) -> torch.Tensor:
        h = F.gelu(self.fc1(x))
        if self.use_g and g_tok is not None:
            h = modulate_vector(h, g_tok, self.prior_floor)
        return self.drop(self.fc2(self.drop(h)))


class GeoTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=4, mlp_ratio=2.0, dropout=0.0, use_g=True, prior_floor=0.1):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = GeoMultiHeadAttention(dim, num_heads, dropout)
        self.ffn = GeoFFN(dim, mlp_ratio, dropout, use_g, prior_floor)

    def forward(self, x, M=None, g_tok=None):
        x = x + self.attn(self.norm1(x), M)
        return x + self.ffn(self.norm2(x), g_tok)


class GeoViTStage(nn.Module):
    """Feature map (B, C, h, w) -> tokens (B, h*w, D) -> blocks -> feature map (B, D, h, w)."""

    def __init__(
        self,
        in_channels: int,
        dim: int,
        grid_hw: tuple[int, int],
        depth: int = 1,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
        affinity: str = "product",
        use_g: bool = True,
        prior_floor: float = 0.1,
    ):
        super().__init__()
        self.grid_hw = tuple(grid_hw)
        self.proj = nn.Conv2d(in_channels, dim, 1)
        self.pos = nn.Parameter(torch.zeros(1, grid_hw[0] * grid_hw[1], dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.affinity = build_affinity(affinity, num_heads)
        self.blocks = nn.ModuleList(
            GeoTransformerBlock(dim, num_heads, mlp_ratio, dropout, use_g, prior_floor) for _ in range(depth)
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, feat: torch.Tensor, w: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        B, _, h, wd = feat.shape
        if (h, wd) != self.grid_hw:
            raise ValueError(f"Stage built for grid {self.grid_hw}, got {(h, wd)}")
        x = self.proj(feat).flatten(2).transpose(1, 2) + self.pos
        M = self.affinity(token_prior(w, (h, wd))) if self.affinity is not None else None
        g_tok = token_prior(g, (h, wd))
        for blk in self.blocks:
            x = blk(x, M, g_tok)
        return self.norm(x).transpose(1, 2).reshape(B, -1, h, wd)


class GeoViT(nn.Module):
    """Pure (geology-aware) ViT baseline. With affinity='none', use_g=False it is a plain ViT."""

    def __init__(
        self,
        in_channels: int,
        patch_size: int = 32,
        vit_patch: int = 4,
        dim: int = 64,
        depth: int = 3,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1,
        affinity: str = "product",
        use_g: bool = True,
        prior_floor: float = 0.1,
        hidden: int = 128,
    ):
        super().__init__()
        n = patch_size // vit_patch
        self.embed = nn.Conv2d(in_channels, dim, vit_patch, stride=vit_patch)
        self.stage = GeoViTStage(dim, dim, (n, n), depth, num_heads, mlp_ratio, dropout, affinity, use_g, prior_floor)
        self.use_g, self.prior_floor = use_g, prior_floor
        self.fc1, self.fc2 = nn.Linear(dim, hidden), nn.Linear(hidden, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, w, g_patch, g_center):
        f = self.stage(self.embed(x), w, g_patch)
        h = torch.relu(self.fc1(f.mean((2, 3))))
        if self.use_g:
            h = modulate_vector(h, g_center, self.prior_floor)
        return self.fc2(self.drop(h)).squeeze(-1)
