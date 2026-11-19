"""Northern Curnamona Province REE dataset (Luo et al. 2026, DEEP-SEAM; data from SARIG).

``prepare_curnamona`` turns the raw layers into a ``GeoGrid``:

* ``stack``      (C, H, W) robust-standardised evidence layers (magnetics, gravity, radiometrics, ASTER, DEM)
* ``valid``      (H, W) study-area mask (Clip_frame polygon)
* ``distances``  {factor: (H, W) km} distance to each ore-controlling factor (input to the w prior)
* ``g_map``      (H, W) in [0, 1] geochemical prior (CLR-PCA REE component, IDW-gridded)
* ``deposits_rc`` (n, 2) grid cells of the known REE occurrences (labels)

The spatial prior w is *not* stored: it depends on which deposits are used for fitting and is
therefore rebuilt per cross-validation fold from training deposits only.

Download the data with ``python scripts/download_curnamona.py``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from affine import Affine

from ..config import CurnamonaConfig
from ..geology.geochem_prior import fit_geochem_pca, idw_interpolate, robust_unit_scale
from .raster import GridSpec, distance_km, rasterize_geometries, read_aligned, robust_standardize


@dataclass
class GeoGrid:
    stack: np.ndarray
    channel_names: list[str]
    grid: GridSpec
    valid: np.ndarray
    distances: dict[str, np.ndarray]
    g_map: np.ndarray
    deposits_rc: np.ndarray
    deposit_names: list[str]
    meta: dict = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int]:
        return self.grid.shape

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "channel_names": self.channel_names,
            "transform": list(self.grid.transform)[:6],
            "shape": list(self.grid.shape),
            "crs_wkt": self.grid.crs_wkt,
            "distance_names": list(self.distances),
            "deposit_names": self.deposit_names,
            "meta": self.meta,
        }
        np.savez_compressed(
            path,
            stack=self.stack,
            valid=self.valid,
            distances=np.stack(list(self.distances.values())),
            g_map=self.g_map,
            deposits_rc=self.deposits_rc,
            meta=np.array(json.dumps(meta)),
        )

    @classmethod
    def load(cls, path: str | Path) -> "GeoGrid":
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
            grid = GridSpec(Affine(*meta["transform"]), tuple(meta["shape"]), meta["crs_wkt"])
            return cls(
                stack=z["stack"],
                channel_names=meta["channel_names"],
                grid=grid,
                valid=z["valid"],
                distances=dict(zip(meta["distance_names"], z["distances"])),
                g_map=z["g_map"],
                deposits_rc=z["deposits_rc"],
                deposit_names=meta["deposit_names"],
                meta=meta["meta"],
            )


def _raster_paths(root: Path, groups) -> list[tuple[str, Path]]:
    out = []
    for group in groups:
        files = sorted((root / group).glob("*.tif"))
        if not files:
            raise FileNotFoundError(f"No .tif files in {root / group}")
        out += [(group, f) for f in files]
    return out


def prepare_curnamona(cfg: CurnamonaConfig | None = None, verbose: bool = True) -> GeoGrid:
    cfg = cfg or CurnamonaConfig()
    root = Path(cfg.root)
    if not root.exists():
        raise FileNotFoundError(f"{root} not found. Run `python scripts/download_curnamona.py` first.")
    grid = GridSpec.from_raster(root / cfg.reference_raster)
    log = print if verbose else (lambda *a, **k: None)
    log(f"Reference grid {grid.shape}, cell {grid.pixel_km()[0]:.3f} x {grid.pixel_km()[1]:.3f} km")

    # Study area
    frame = gpd.read_file(root / cfg.clip_frame)
    valid = rasterize_geometries(frame, grid, all_touched=False)

    # Evidence layers
    names, bands = [], []
    for group, path in _raster_paths(root, cfg.raster_groups):
        bands.append(read_aligned(path, grid, zero_is_nodata=group in cfg.zero_is_nodata_groups))
        names.append(f"{group}/{path.stem}")
    raw = np.stack(bands)
    nan_frac = {n: float(np.isnan(b[valid]).mean()) for n, b in zip(names, raw)}
    stack, norm_stats = robust_standardize(raw, valid)
    log(f"Stacked {len(names)} evidence layers")

    # Ore-controlling factors -> distance maps
    distances = {}
    for name, rel in cfg.controlling_factors.items():
        mask = rasterize_geometries(gpd.read_file(root / rel), grid)
        distances[name] = distance_km(mask, grid)
        log(f"Distance to {name}: max {distances[name][valid].max():.1f} km")

    # Deposits
    dep = gpd.read_file(root / cfg.deposits).to_crs(grid.crs_wkt)
    deposits_rc = grid.rowcol(dep.geometry.x.values, dep.geometry.y.values)
    inside = (
        (deposits_rc[:, 0] >= 0) & (deposits_rc[:, 0] < grid.shape[0])
        & (deposits_rc[:, 1] >= 0) & (deposits_rc[:, 1] < grid.shape[1])
    )
    inside[inside] &= valid[deposits_rc[inside, 0], deposits_rc[inside, 1]]
    if not inside.all():
        log(f"Dropping {int((~inside).sum())} deposits outside the study area")
    deposits_rc = deposits_rc[inside]
    name_col = "DEP_NAME" if "DEP_NAME" in dep.columns else dep.columns[0]
    deposit_names = [str(n) for n in dep.loc[inside, name_col]]
    log(f"{len(deposits_rc)} REE occurrences: {deposit_names}")

    # Geochemical prior g
    gc = pd.read_csv(root / cfg.geochem_csv)
    lon_c, lat_c = cfg.geochem_coords
    elements = [c for c in gc.columns if c not in (lon_c, lat_c)]
    gc = gc.dropna(subset=elements)
    gpca = fit_geochem_pca(gc[elements].to_numpy(float), elements, cfg.geochem_indicators, cfg.geochem_n_components)
    scores = gpca.transform(gc[elements].to_numpy(float))
    lon, lat = grid.cell_centers_lonlat()
    g_raw = idw_interpolate(
        grid.lonlat_to_km(gc[lon_c], gc[lat_c]), scores,
        grid.lonlat_to_km(lon.ravel(), lat.ravel()), cfg.idw_neighbours, cfg.idw_power,
    ).reshape(grid.shape)
    g_map = robust_unit_scale(g_raw, mask=valid)
    g_map[~valid] = 0.0
    info = gpca.summary()
    log(f"g = PC{info['component'] + 1} ({100 * info['explained_variance_ratio']:.1f}% var) from {len(gc)} samples")

    return GeoGrid(
        stack=stack,
        channel_names=names,
        grid=grid,
        valid=valid,
        distances=distances,
        g_map=g_map.astype(np.float32),
        deposits_rc=deposits_rc,
        deposit_names=deposit_names,
        meta={"nan_fraction": nan_frac, "normalisation": norm_stats, "geochem_pca": info, "config": cfg.__dict__},
    )


def load_or_prepare(cfg: CurnamonaConfig | None = None, rebuild: bool = False, verbose: bool = True) -> GeoGrid:
    cfg = cfg or CurnamonaConfig()
    cache = Path(cfg.cache)
    if cache.exists() and not rebuild:
        return GeoGrid.load(cache)
    gg = prepare_curnamona(cfg, verbose)
    gg.save(cache)
    return gg
