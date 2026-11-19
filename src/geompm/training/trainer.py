"""Training loop shared by every model of the comparison grid."""

from __future__ import annotations

import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..config import TrainConfig
from ..geology.geo_loss import GeoLoss


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device: str = "auto") -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def forward_batch(model, batch, device):
    return model(
        batch["x"].to(device, non_blocking=True),
        batch["w"].to(device, non_blocking=True),
        batch["g_patch"].to(device, non_blocking=True),
        batch["g_center"].to(device, non_blocking=True),
    )


def train_model(model: torch.nn.Module, train_ds, cfg: TrainConfig, log_every: int = 10, verbose: bool = True) -> dict:
    """Train with BCE (class-balanced) + lambda * GeoLoss; returns the per-epoch history.

    The optimiser covers the model *and* the loss's learnable prior affine (a, b) - previously the
    loss parameters were never optimised.
    """
    device = resolve_device(cfg.device)
    model.to(device)
    labels = train_ds.labels
    pos_weight = float((labels == 0).sum() / max((labels == 1).sum(), 1))
    criterion = GeoLoss(cfg.geo_lambda, pos_weight=pos_weight).to(device)
    params = list(model.parameters()) + [p for p in criterion.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=len(train_ds) > cfg.batch_size)

    history = []
    for epoch in range(cfg.epochs):
        model.train()
        tot = data = geo = 0.0
        n = 0
        for batch in loader:
            logits = forward_batch(model, batch, device)
            loss, parts = criterion(logits, batch["y"].to(device), batch["w_center"].to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            b = len(logits)
            tot, data, geo, n = tot + loss.item() * b, data + parts["data"].item() * b, geo + float(parts["geo"]) * b, n + b
        sched.step()
        rec = {"epoch": epoch + 1, "loss": tot / n, "data": data / n, "geo": geo / n, "a": criterion.a.item(), "b": criterion.b.item()}
        history.append(rec)
        if verbose and ((epoch + 1) % log_every == 0 or epoch == 0):
            print(f"  epoch {rec['epoch']:3d} | loss {rec['loss']:.4f} (data {rec['data']:.4f}, geo {rec['geo']:.4f})")
    return {"history": history, "pos_weight": pos_weight}
