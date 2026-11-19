"""Baleshwar-Dariba (BRS) soil geochemistry, Khetri Copper Belt, Rajasthan (local GSI-style survey).

Fixes relative to ``brs_report_data_prep.ipynb``:

* Censored values: "<50" was turned into 49 (and "<10" into 9), so the target "Cu >= 10 ppm" meant
  "Cu above detection limit", not "anomalous Cu". Censored values now use the standard DL/2
  substitution and keep a flag column; the anomaly threshold is a parameter (``robust_anomaly_threshold``
  gives a data-driven default: median + 2 MAD of log10 values).
* Target leakage: PC1 was computed from Zn, Ni, Co **and Cu** while predicting Cu presence. The
  geochemical feature now excludes the target element.
* The proximity "centre" (highest-Cu sample) was chosen using validation rows, distance was in raw
  degrees and k = +0.95 was hand-set (w *increased* with distance). Features are now fitted on the
  training fold only (``fit_features`` / ``transform``), distance is haversine km and k is fitted.
* Random splits overstate skill on spatially autocorrelated samples: use ``spatial_blocks`` with
  GroupKFold.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from ..geology.spatial_prior import PowerLawFit, fit_power_law

ELEMENTS = ("Cu", "Pb", "Zn", "Ni", "Co", "Ag", "Cd")


def parse_censored(val, policy: str = "half") -> tuple[float, str]:
    """Parse '<50', '>100', '163', 'NA' -> (value, flag) where flag is '<', '>' or ''.

    policy for '<DL': 'half' -> DL/2 (common geochemical convention), 'dl' -> DL, 'nan' -> NaN.
    """
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return np.nan, ""
    s = str(val).strip()
    if s.upper() in {"", "NA", "N/A", "-", "ND"}:
        return np.nan, ""
    flag = s[0] if s[0] in "<>" else ""
    m = re.search(r"\d+(?:\.\d+)?", s)
    if not m:
        return np.nan, ""
    num = float(m.group())
    if flag == "<":
        num = {"half": num / 2, "dl": num, "nan": np.nan}[policy]
    return num, flag


def parse_dms(coord) -> float:
    """'N27°40' 56.7"' -> 27.682417 (S/W negative)."""
    if coord is None or (isinstance(coord, float) and np.isnan(coord)):
        return np.nan
    s = str(coord).strip()
    nums = re.findall(r"\d+(?:\.\d+)?", s)
    if not nums:
        return np.nan
    deg, minute, sec = (float(x) for x in (nums + ["0", "0"])[:3])
    dd = deg + minute / 60 + sec / 3600
    hemi = re.search(r"[NSEWnsew]", s)
    return -dd if hemi and hemi.group().upper() in "SW" else dd


def load_brs(path, sheet: str = "BRS", header_row: int = 1, policy: str = "half") -> pd.DataFrame:
    """Load the BRS sheet -> tidy frame with float element columns, censoring flags and decimal lat/lon."""
    raw = pd.read_excel(path, sheet_name=sheet, header=header_row, engine="xlrd")
    df = pd.DataFrame({"sample": raw["Sam.No"], "location": raw.get("Location")})
    for col in raw.columns:
        m = re.match(r"([A-Z][a-z]?)\s*\(ppm\)", str(col))
        if m:
            parsed = raw[col].map(lambda v: parse_censored(v, policy))
            df[m.group(1)] = parsed.map(lambda t: t[0]).astype(float)
            df[f"{m.group(1)}_censored"] = parsed.map(lambda t: t[1])
    df["latitude"] = raw["Latitude"].map(parse_dms)
    df["longitude"] = raw["Longitude"].map(parse_dms)
    return df


def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(a, dtype=float)) for a in (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.0088 * np.arcsin(np.sqrt(a))


def robust_anomaly_threshold(values, n_mad: float = 2.0) -> float:
    """Anomaly threshold = 10 ** (median + n_mad * 1.4826 * MAD) of log10 values."""
    v = np.log10(np.asarray(values, dtype=float)[np.isfinite(values)])
    med = np.median(v)
    mad = 1.4826 * np.median(np.abs(v - med))
    return float(10 ** (med + n_mad * mad))


def spatial_blocks(lat, lon, block_km: float = 0.5) -> np.ndarray:
    """Integer block id of a regular block_km grid, for GroupKFold spatial cross-validation."""
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    y = (lat - np.nanmin(lat)) * 110.574
    x = (lon - np.nanmin(lon)) * 111.320 * np.cos(np.radians(np.nanmean(lat)))
    return (np.floor(y / block_km) * 10_000 + np.floor(x / block_km)).astype(int)


@dataclass
class BRSFeatures:
    """Fold-safe feature builder: fit on the training fold, transform any rows."""

    target: str
    threshold: float
    pc_elements: list[str]
    d_min_km: float = 0.05  # distance floor ~ sample spacing; avoids a spike of w at the centre
    scaler: StandardScaler | None = None
    pca: PCA | None = None
    pc_sign: float = 1.0
    center: tuple[float, float] | None = None
    fit_: PowerLawFit | None = None
    d_ref: np.ndarray | None = None

    def fit(self, df: pd.DataFrame) -> "BRSFeatures":
        df = df.dropna(subset=[self.target, "latitude", "longitude"])
        X = np.log10(df[self.pc_elements].clip(lower=1e-3).fillna(df[self.pc_elements].median()))
        self.scaler = StandardScaler().fit(X)
        self.pca = PCA(n_components=1).fit(self.scaler.transform(X))
        self.pc_sign = 1.0 if self.pca.components_[0].sum() >= 0 else -1.0
        i = df[self.target].idxmax()
        self.center = (float(df.at[i, "latitude"]), float(df.at[i, "longitude"]))
        d = haversine_km(df["latitude"], df["longitude"], *self.center)
        self.d_ref = d
        self.fit_ = fit_power_law(d, (df[self.target] >= self.threshold).to_numpy(), n_bins=6, d_min=self.d_min_km)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        X = np.log10(df[self.pc_elements].clip(lower=1e-3).fillna(df[self.pc_elements].median()))
        pc = self.pc_sign * self.pca.transform(self.scaler.transform(X))[:, 0]
        d = haversine_km(df["latitude"], df["longitude"], *self.center)
        density = lambda dist: self.fit_.N * np.maximum(dist, self.fit_.d_min) ** self.fit_.k  # noqa: E731
        m_max = density(self.d_ref).max()  # m_max over the training fold
        w = np.clip(density(d) / m_max, 0, 1)
        return pd.DataFrame({"w_proximity": w, "g_pc1": pc, "distance_km": d}, index=df.index)


def make_labels(df: pd.DataFrame, target: str, threshold: float) -> pd.Series:
    return (df[target] >= threshold).astype(int)


__all__ = [
    "ELEMENTS",
    "BRSFeatures",
    "haversine_km",
    "load_brs",
    "make_labels",
    "parse_censored",
    "parse_dms",
    "robust_anomaly_threshold",
    "spatial_blocks",
]
