"""Collect experiments/results/*/results.json into one comparison table (+ prior-only references).

    python scripts/summarize_results.py [--results experiments/results] [--out experiments/results/summary.csv]

Reference rows use no learning:
  prior_w_only  - the w prior refitted per fold without the held-out deposit (leak-free)
  prior_g_only  - the g prior (unsupervised, identical in every fold)
  random        - expected area rank of a random map (0.5)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from geompm.config import CurnamonaConfig, PriorConfig
from geompm.data.curnamona import load_or_prepare
from geompm.geology.spatial_prior import build_spatial_prior
from geompm.training.evaluation import area_rank


def summarise(ranks: np.ndarray) -> dict:
    return {
        "mean_area_rank": ranks.mean(),
        "median_area_rank": np.median(ranks),
        "captured@5%": (ranks <= 0.05).mean(),
        "captured@10%": (ranks <= 0.10).mean(),
        "captured@20%": (ranks <= 0.20).mean(),
    }


def prior_references(names: list[str]) -> list[dict]:
    gg = load_or_prepare(CurnamonaConfig(), verbose=False)
    pc = PriorConfig()
    w_ranks = []
    for i in range(len(gg.deposits_rc)):
        w, _ = build_spatial_prior(
            gg.distances, np.delete(gg.deposits_rc, i, 0), gg.valid, pc.n_bins, pc.combine, pc.drop_nonnegative_k
        )
        w_ranks.append(area_rank(w, gg.valid, gg.deposits_rc[i : i + 1])[0])
    g_ranks = area_rank(gg.g_map, gg.valid, gg.deposits_rc)
    rows = []
    for name, ranks in (("prior_w_only", np.array(w_ranks)), ("prior_g_only", g_ranks)):
        rows.append({"config": name, **summarise(ranks), **dict(zip(names, ranks))})
    rows.append({"config": "random", "mean_area_rank": 0.5, "median_area_rank": 0.5,
                 "captured@5%": 0.05, "captured@10%": 0.10, "captured@20%": 0.20})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="experiments/results")
    ap.add_argument("--out", default="experiments/results/summary.csv")
    args = ap.parse_args()

    rows, names = [], None
    for f in sorted(Path(args.results).glob("*/results.json")):
        r = json.loads(f.read_text())
        if "lodo" not in r:
            continue
        folds = r["lodo"]["folds"]
        names = [fd["held_out"] for fd in folds]
        ranks = np.array([fd["area_rank"] for fd in folds])
        row = {"config": r["name"], **summarise(ranks), **dict(zip(names, ranks))}
        row["minutes"] = sum(fd["seconds"] for fd in folds) / 60
        if "full" in r:
            row["train_sr_auc"] = r["full"]["train_success_rate_auc"]
            row["n_params"] = r["full"]["n_params"]
        rows.append(row)
    if not rows:
        raise SystemExit(f"No results with LODO folds under {args.results}")
    rows += prior_references(names)
    df = pd.DataFrame(rows).set_index("config").sort_values("mean_area_rank")
    df.to_csv(args.out)
    pd.set_option("display.width", 200)
    print(df.round(3).to_string())
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
