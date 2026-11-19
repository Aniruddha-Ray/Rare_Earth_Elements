"""Run one experiment config: leave-one-deposit-out CV and/or a full-data map.

    python scripts/run_experiment.py configs/shvcnn.json            # LODO + full map
    python scripts/run_experiment.py configs/shcnn.json --mode lodo
    python scripts/run_experiment.py configs/cnn.json --epochs 5    # quick smoke run

Outputs go to <output_dir>/<name>/: config.json, results.json, prospectivity.tif, w_prior.tif.
"""

from __future__ import annotations

import argparse
import json

import torch

from geompm.config import ExperimentConfig
from geompm.data.curnamona import load_or_prepare
from geompm.data.raster import write_geotiff
from geompm.training.experiment import run_full, run_lodo, save_results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--mode", choices=["lodo", "full", "both"], default="both")
    ap.add_argument("--epochs", type=int, help="override train.epochs")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    cfg = ExperimentConfig.load(args.config)
    if args.epochs:
        cfg.train.epochs = args.epochs
    gg = load_or_prepare(cfg.data, verbose=not args.quiet)
    results: dict = {"name": cfg.name, "n_deposits": len(gg.deposits_rc)}

    if args.mode in ("lodo", "both"):
        lodo = run_lodo(gg, cfg, verbose=not args.quiet)
        results["lodo"] = {k: v for k, v in lodo.items() if not k.startswith("_")}
        print(
            f"[{cfg.name}] LODO mean area rank {100 * lodo['mean_area_rank']:.2f}% | "
            f"captured@5% {lodo['captured_at_5pct']:.2f} | captured@10% {lodo['captured_at_10pct']:.2f}"
        )
    out = save_results(results, cfg, cfg.output_dir)

    if args.mode in ("full", "both"):
        full = run_full(gg, cfg, verbose=not args.quiet)
        write_geotiff(out / "prospectivity.tif", full["prob"], gg.grid)
        write_geotiff(out / "w_prior.tif", full["w_map"], gg.grid)
        torch.save(full["model"].state_dict(), out / "model.pt")
        results["full"] = {
            "train_success_rate_auc": full["train_success_rate_auc"],
            "train_captured_at_5pct": full["train_captured_at_5pct"],
            "priors": full["info"]["priors"],
            "counts": full["info"]["counts"],
            "n_params": full["info"]["n_params"],
        }
    save_results(results, cfg, cfg.output_dir)
    print(f"Saved to {out}")
    print(json.dumps({k: v for k, v in results.items() if k != "lodo"}, indent=1, default=float)[:1500])


if __name__ == "__main__":
    main()
