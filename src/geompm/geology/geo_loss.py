"""Soft geological constraint.

    L_total = L_data + lambda * L_geo
    L_geo   = || p - sigmoid(a * w + b) ||_2 / sqrt(B)

where ``p = sigmoid(logit)`` is the predicted mineralisation probability of each sample and ``w`` the
spatial prior at that sample's location.

Bugs fixed relative to the notebook version:
* models now return *logits*; previously sigmoid outputs were fed to ``binary_cross_entropy_with_logits``
  (a double sigmoid that pins the loss near a constant - the SHCNN loss sat at 1.7580 from epoch 2);
* ``p`` was a softmax across the *batch* and was compared with a single scalar (mean of the whole w map);
  it is now a per-sample comparison with the prior at each sample;
* ``lambda`` was an ``nn.Parameter`` (the optimiser could drive it to 0 or below, switching the
  constraint off); it is now a fixed hyper-parameter;
* ``a`` and ``b`` were never passed to the optimiser; the trainer now optimises ``loss.parameters()``.
  ``a`` is kept positive (softplus) so the prior keeps its meaning "higher w -> more prospective".
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class GeoLoss(nn.Module):
    def __init__(self, lam: float = 0.1, pos_weight: float | None = None, learn_affine: bool = True):
        super().__init__()
        if lam < 0:
            raise ValueError("lambda must be >= 0")
        self.lam = float(lam)
        self.register_buffer(
            "pos_weight", torch.tensor(float(pos_weight)) if pos_weight is not None else torch.tensor(1.0)
        )
        # softplus(0.5413) == 1.0
        self.raw_a = nn.Parameter(torch.tensor(0.5413), requires_grad=learn_affine)
        self.b = nn.Parameter(torch.tensor(0.0), requires_grad=learn_affine)

    @property
    def a(self) -> torch.Tensor:
        return F.softplus(self.raw_a)

    def forward(self, logits: torch.Tensor, target: torch.Tensor, w_sample: torch.Tensor):
        logits, target, w_sample = logits.view(-1), target.view(-1).float(), w_sample.view(-1)
        l_data = F.binary_cross_entropy_with_logits(logits, target, pos_weight=self.pos_weight)
        if self.lam == 0:
            return l_data, {"data": l_data.detach(), "geo": torch.zeros(())}
        p = torch.sigmoid(logits)
        prior = torch.sigmoid(self.a * w_sample + self.b)
        l_geo = torch.linalg.vector_norm(p - prior) / math.sqrt(p.numel())
        return l_data + self.lam * l_geo, {"data": l_data.detach(), "geo": l_geo.detach()}
