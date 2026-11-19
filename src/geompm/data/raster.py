"""Raster / vector utilities: grid alignment, rasterisation, metric distance maps, normalisation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import rasterio
from affine import Affine
from rasterio.features import rasterize
from rasterio.warp import Resampling, reproject
from scipy.ndimage import distance_transform_edt

KM_PER_DEG_LAT = 110.574


def km_per_deg_lon(lat_deg: float) -> float:
    return 111.320 * np.cos(np.radians(lat_deg))


@dataclass
class GridSpec:
    """Reference grid: every layer is resampled onto it."""

    transform: Affine
    shape: tuple[int, int]
    crs_wkt: str

    @classmethod
    def from_raster(cls, path) -> "GridSpec":
        with rasterio.open(path) as src:
            return cls(src.transform, (src.height, src.width), src.crs.to_wkt())

    @property
    def center_lat(self) -> float:
        return float(self.transform.f + self.transform.e * self.shape[0] / 2)

    def pixel_km(self) -> tuple[float, float]:
        """(dy_km, dx_km) of one cell, assuming a geographic (degree) CRS."""
        return abs(self.transform.e) * KM_PER_DEG_LAT, abs(self.transform.a) * km_per_deg_lon(self.center_lat)

    def lonlat_to_km(self, lon, lat) -> np.ndarray:
        """Local equirectangular projection (km) centred on the grid - adequate at this extent."""
        lon0 = self.transform.c + self.transform.a * self.shape[1] / 2
        lat0 = self.center_lat
        return np.column_stack(
            [(np.asarray(lon) - lon0) * km_per_deg_lon(lat0), (np.asarray(lat) - lat0) * KM_PER_DEG_LAT]
        )

    def cell_centers_lonlat(self) -> tuple[np.ndarray, np.ndarray]:
        rows, cols = np.mgrid[0 : self.shape[0], 0 : self.shape[1]]
        lon = self.transform.c + (cols + 0.5) * self.transform.a
        lat = self.transform.f + (rows + 0.5) * self.transform.e
        return lon, lat

    def rowcol(self, lon, lat) -> np.ndarray:
        r, c = rasterio.transform.rowcol(self.transform, np.asarray(lon), np.asarray(lat))
        return np.column_stack([np.asarray(r), np.asarray(c)]).astype(int)


def read_aligned(path, grid: GridSpec, zero_is_nodata: bool = False, resampling=Resampling.bilinear) -> np.ndarray:
    """Read band 1 of a raster and resample it onto ``grid``; nodata becomes NaN."""
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)
        nodata = src.nodata
        mask = ~np.isfinite(data)
        if nodata is not None:
            mask |= data == nodata
        if zero_is_nodata:
            mask |= data == 0
        data[mask] = np.nan
        out = np.full(grid.shape, np.nan, dtype=np.float32)
        reproject(
            data, out,
            src_transform=src.transform, src_crs=src.crs, src_nodata=np.nan,
            dst_transform=grid.transform, dst_crs=grid.crs_wkt, dst_nodata=np.nan,
            resampling=resampling,
        )
    return out


def rasterize_geometries(gdf, grid: GridSpec, all_touched: bool = True) -> np.ndarray:
    """Boolean mask of cells touched by the geometries (lines -> traces, polygons -> interiors)."""
    gdf = gdf.to_crs(grid.crs_wkt) if gdf.crs is not None else gdf
    shapes = [(geom, 1) for geom in gdf.geometry if geom is not None and not geom.is_empty]
    return rasterize(shapes, out_shape=grid.shape, transform=grid.transform, fill=0, all_touched=all_touched, dtype="uint8").astype(bool)


def distance_km(feature_mask: np.ndarray, grid: GridSpec) -> np.ndarray:
    """Euclidean distance (km) from every cell to the nearest feature cell (0 on the feature)."""
    if not feature_mask.any():
        raise ValueError("Feature mask is empty - nothing to measure distance to")
    return distance_transform_edt(~feature_mask, sampling=grid.pixel_km()).astype(np.float32)


def robust_standardize(stack: np.ndarray, valid: np.ndarray, clip: float = 5.0) -> tuple[np.ndarray, dict]:
    """Per-channel (x - median) / IQR inside the study area, clipped to +-clip; NaN -> 0 (= median).

    Statistics are unsupervised (no labels), so computing them on the whole map does not leak targets.
    """
    out = np.zeros_like(stack, dtype=np.float32)
    stats = {"median": [], "iqr": []}
    for i, band in enumerate(stack):
        vals = band[valid & np.isfinite(band)]
        if vals.size == 0:
            med, iqr = 0.0, 1.0
        else:
            q1, med, q3 = np.percentile(vals, [25, 50, 75])
            iqr = q3 - q1 if q3 > q1 else (vals.std() or 1.0)
        z = np.clip((band - med) / iqr, -clip, clip)
        out[i] = np.where(np.isfinite(z), z, 0.0)
        stats["median"].append(float(med))
        stats["iqr"].append(float(iqr))
    return out, stats


def write_geotiff(path, array: np.ndarray, grid: GridSpec, nodata: float = np.nan) -> None:
    arr = np.asarray(array, dtype=np.float32)
    with rasterio.open(
        path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1, dtype="float32",
        crs=grid.crs_wkt, transform=grid.transform, nodata=nodata, compress="deflate",
    ) as dst:
        dst.write(arr, 1)
