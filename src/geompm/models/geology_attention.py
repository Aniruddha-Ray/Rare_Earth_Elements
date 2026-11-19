"""Geological self-attention: softmax(QK^T / sqrt(d) + alpha * M) V.

The affinity matrix ``M`` is built per sample from token-level spatial priors ``w_i``:

* ``product``  M_ij = w_i * w_j
* ``learned``  M_ij = MLP(w_i, w_j, |w_i - w_j|), one output per head
* ``none``     plain self-attention (baseline)

Fixes relative to the notebook: ``M`` used to be built as ``ones @ (w*w) @ ones``, so every
patch-patch entry was the same global sum of w^2 (a per-row constant that softmax cancels - geology
had no effect on which patches attend to which), while the CLS row/column were 1, so the bias mainly
starved attention to the CLS token. ``M`` is now built per token from local priors. ``alpha`` is per head.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ProductAffinity(nn.Module):
    def forward(self, w_tok: torch.Tensor) -> torch.Tensor:  # (B, N) -> (B, 1, N, N)
        return (w_tok.unsqueeze(-1) * w_tok.unsqueeze(-2)).unsqueeze(1)


class LearnedAffinity(nn.Module):
    def __init__(self, num_heads: int, hidden: int = 16):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(3, hidden), nn.GELU(), nn.Linear(hidden, num_heads))

    def forward(self, w_tok: torch.Tensor) -> torch.Tensor:  # (B, N) -> (B, heads, N, N)
        wi = w_tok.unsqueeze(-1).expand(-1, -1, w_tok.shape[1])
        wj = w_tok.unsqueeze(-2).expand(-1, w_tok.shape[1], -1)
        feats = torch.stack([wi, wj, (wi - wj).abs()], dim=-1)
        return self.mlp(feats).permute(0, 3, 1, 2)


def build_affinity(kind: str, num_heads: int) -> nn.Module | None:
    if kind == "none":
        return None
    if kind == "product":
        return ProductAffinity()
    if kind == "learned":
        return LearnedAffinity(num_heads)
    raise ValueError(f"Unknown affinity '{kind}' (none | product | learned)")


class GeoMultiHeadAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.0, alpha_init: float = 1.0):
        super().__init__()
        if dim % num_heads:
            raise ValueError(f"dim {dim} not divisible by num_heads {num_heads}")
        self.num_heads, self.head_dim = num_heads, dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.att_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)
        self.alpha = nn.Parameter(torch.full((num_heads,), float(alpha_init)))
        self.last_attention: torch.Tensor | None = None  # kept for explainability plots

    def forward(self, x: torch.Tensor, M: torch.Tensor | None = None) -> torch.Tensor:
        B, N, D = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        att = (q @ k.transpose(-2, -1)) * self.head_dim**-0.5
        if M is not None:
            att = att + self.alpha.view(1, -1, 1, 1) * M
        att = att.softmax(dim=-1)
        self.last_attention = att.detach()
        out = (self.att_drop(att) @ v).transpose(1, 2).reshape(B, N, D)
        return self.proj_drop(self.proj(out))
