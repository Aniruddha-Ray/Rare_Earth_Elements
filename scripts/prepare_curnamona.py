"""Build the model-ready Curnamona grid (evidence stack, distance maps, g prior, deposits) and cache it.

    python scripts/prepare_curnamona.py [--rebuild] [--export-tifs]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from geompm.config import CurnamonaConfig, PriorConfig
from geompm.data.curnamona import load_or_prepare
from geompm.data.raster import write_geotiff
from geompm.geology.spatial_prior import build_spatial_prior


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--export-tifs", action="store_true", help="write g and all-deposit w priors as GeoTIFF")
    args = ap.parse_args()

    cfg = CurnamonaConfig()
    gg = load_or_prepare(cfg, rebuild=args.rebuild)
    print(f"stack {gg.stack.shape}, valid cells {int(gg.valid.sum())}, deposits {len(gg.deposits_rc)}")
    print("geochem PCA:", json.dumps(gg.meta["geochem_pca"]["loadings"]))

    pc = PriorConfig()
    w_map, fits = build_spatial_prior(gg.distances, gg.deposits_rc, gg.valid, pc.n_bins, pc.combine, pc.drop_nonnegative_k)
    for name, fit in fits.items():
        used = "used" if fit.k < 0 or not pc.drop_nonnegative_k else "EXCLUDED (k >= 0)"
        print(
            f"w[{name}]: N={fit.N:.4g}, k={fit.k:.3f} {used}\n"
            f"    bins {fit.bin_centers.round(2).tolist()} km, density {fit.density.round(5).tolist()}"
        )
    if args.export_tifs:
        out = Path("data/processed")
        write_geotiff(out / "g_prior.tif", gg.g_map, gg.grid)
        write_geotiff(out / "w_prior_all_deposits.tif", w_map, gg.grid)
        print(f"Wrote priors to {out}")


if __name__ == "__main__":
    main()
