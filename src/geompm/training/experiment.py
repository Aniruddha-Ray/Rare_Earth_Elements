"""Leave-one-deposit-out (LODO) cross-validation and full-data training on a GeoGrid.

For each fold the held-out deposit is invisible to *everything* fitted: the power-law spatial prior
(N, k per controlling factor), positive patches and the negative-sampling buffer. The geochemical
prior and the evidence-layer normalisation are unsupervised and are shared across folds.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from ..config import ExperimentConfig
from ..data.curnamona import GeoGrid
from ..data.patches import build_training_set
from ..geology.spatial_prior import build_spatial_prior
from ..models.factory import build_model, count_parameters
from .evaluation import area_rank, captured_at, success_rate_curve
from .predict import predict_map
from .trainer import set_seed, train_model


def fit_and_predict(gg: GeoGrid, train_rc: np.ndarray, cfg: ExperimentConfig, seed: int, verbose: bool = True):
    set_seed(seed)
    w_map, fits = build_spatial_prior(
        gg.distances, train_rc, gg.valid, cfg.prior.n_bins, cfg.prior.combine, cfg.prior.drop_nonnegative_k
    )
    t = cfg.train
    ds, counts = build_training_set(
        gg.stack, w_map, gg.g_map, gg.valid, train_rc, gg.grid.pixel_km(), cfg.model.patch_size,
        t.pos_radius_cells, t.neg_buffer_km, t.neg_per_pos, t.augment, seed,
    )
    model = build_model(cfg.model, gg.stack.shape[0])
    info = train_model(model, ds, t, verbose=verbose)
    prob = predict_map(model, gg.stack, w_map, gg.g_map, gg.valid, cfg.model.patch_size, device=t.device)
    info.update(counts=counts, priors={k: v.to_dict() for k, v in fits.items()}, n_params=count_parameters(model))
    return model, prob, w_map, info


def run_lodo(gg: GeoGrid, cfg: ExperimentConfig, verbose: bool = True) -> dict:
    n = len(gg.deposits_rc)
    folds, maps = [], []
    for i in range(n):
        t0 = time.time()
        train_rc = np.delete(gg.deposits_rc, i, axis=0)
        if verbose:
            print(f"[{cfg.name}] fold {i + 1}/{n}: hold out {gg.deposit_names[i]}")
        _, prob, _, info = fit_and_predict(gg, train_rc, cfg, cfg.train.seed + i, verbose)
        rank = float(area_rank(prob, gg.valid, gg.deposits_rc[i : i + 1])[0])
        folds.append({
            "held_out": gg.deposit_names[i], "area_rank": rank,
            "k": {kk: vv["k"] for kk, vv in info["priors"].items()},
            "seconds": round(time.time() - t0, 1), "final_loss": info["history"][-1],
        })
        maps.append(prob)
        if verbose:
            print(f"    held-out deposit ranks in top {100 * rank:.2f}% of area")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    ranks = np.array([f["area_rank"] for f in folds])
    return {
        "name": cfg.name,
        "folds": folds,
        "mean_area_rank": float(ranks.mean()),
        "median_area_rank": float(np.median(ranks)),
        "captured_at_5pct": float((ranks <= 0.05).mean()),
        "captured_at_10pct": float((ranks <= 0.10).mean()),
        "_maps": maps,
    }


def run_full(gg: GeoGrid, cfg: ExperimentConfig, verbose: bool = True) -> dict:
    """Train on all deposits (for the final map). Its scores on those deposits are *fit*, not skill."""
    model, prob, w_map, info = fit_and_predict(gg, gg.deposits_rc, cfg, cfg.train.seed, verbose)
    _, _, sr_auc = success_rate_curve(prob, gg.valid, gg.deposits_rc)
    return {
        "model": model, "prob": prob, "w_map": w_map, "info": info,
        "train_success_rate_auc": sr_auc,
        "train_captured_at_5pct": captured_at(prob, gg.valid, gg.deposits_rc, 0.05),
    }


def save_results(results: dict, cfg: ExperimentConfig, out_dir: str | Path) -> Path:
    out = Path(out_dir) / cfg.name
    out.mkdir(parents=True, exist_ok=True)
    cfg.save(out / "config.json")
    clean = {k: v for k, v in results.items() if not k.startswith("_")}
    (out / "results.json").write_text(json.dumps(clean, indent=2, default=float))
    return out
