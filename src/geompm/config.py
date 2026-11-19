"""Experiment configuration (dataclasses, serialisable to / from JSON)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

CURNAMONA_ROOT = (
    "data/external/curnamona/EarthByte-MPM_Curnamona_REE-4bb4139/"
    "Files needed for runing/Input_Data_Layers"
)


@dataclass
class CurnamonaConfig:
    """Where the Curnamona layers live and how they are turned into a model-ready grid."""

    root: str = CURNAMONA_ROOT
    reference_raster: str = "Magnetics/SA_TMI_VRTP.tif"
    raster_groups: tuple[str, ...] = ("Magnetics", "Gravity", "Radiometrics", "Remote Sensing", "DEM")
    # ASTER products encode "no data" as 0 (71-97 % of cells), so 0 is masked for these groups.
    zero_is_nodata_groups: tuple[str, ...] = ("Remote Sensing",)
    clip_frame: str = "GIS/Clip_frame.shp"
    deposits: str = "GIS/ree.shp"
    # Ore-controlling factors used to build the spatial prior w (distance-decay).
    controlling_factors: dict[str, str] = field(
        default_factory=lambda: {
            "faults": "Geology factors/original/Archean_Early_Mesoproterozoic_Faults.shp",
            "felsic_granite": "Geology factors/original/Felsic_granite.shp",
        }
    )
    geochem_csv: str = "Geochemical/ILRRPCA.csv"
    geochem_coords: tuple[str, str] = ("lon_gda94", "lat_gda94")
    # Elements whose loadings identify the mineralisation-related principal component.
    geochem_indicators: tuple[str, ...] = ("La", "Y", "Th")
    geochem_n_components: int = 8
    idw_neighbours: int = 12
    idw_power: float = 2.0
    cache: str = "data/processed/curnamona_grid.npz"


@dataclass
class PriorConfig:
    """How w is fitted from (training) deposits and how factor priors are combined."""

    n_bins: int = 6
    combine: str = "mean"  # mean | max | product
    drop_nonnegative_k: bool = True  # exclude factors with no distance decay (k >= 0) from w


@dataclass
class ModelConfig:
    name: str = "shvcnn"  # cnn | shcnn | vit | geovit | cnn_vit | shvcnn
    patch_size: int = 32
    prior_floor: float = 0.1
    dropout: float = 0.3
    hidden: int = 128
    # SHCNN
    cnn_channels: tuple[int, ...] = (32, 64)
    # SHVCNN / ViT
    stage_channels: tuple[int, ...] = (32, 64, 128)
    stage_depths: tuple[int, ...] = (1, 1, 1)
    num_heads: int = 4
    mlp_ratio: float = 2.0
    affinity: str = "product"  # none | product | learned
    vit_patch: int = 4


@dataclass
class TrainConfig:
    epochs: int = 40
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    geo_lambda: float = 0.1  # weight of the soft geological constraint (0 disables it)
    pos_radius_cells: int = 1  # positives = cells within this radius of a deposit
    neg_buffer_km: float = 5.0  # negatives are drawn farther than this from training deposits
    neg_per_pos: int = 10
    augment: bool = True
    seed: int = 42
    device: str = "auto"


@dataclass
class ExperimentConfig:
    name: str = "shvcnn"
    data: CurnamonaConfig = field(default_factory=CurnamonaConfig)
    prior: PriorConfig = field(default_factory=PriorConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    output_dir: str = "experiments/results"

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def from_dict(cls, d: dict) -> "ExperimentConfig":
        def build(dc, values):
            kwargs = {}
            for k, v in (values or {}).items():
                if k not in dc.__dataclass_fields__:
                    raise KeyError(f"Unknown config key '{k}' for {dc.__name__}")
                default = getattr(dc(), k)
                kwargs[k] = tuple(v) if isinstance(default, tuple) else v
            return dc(**kwargs)

        return cls(
            name=d.get("name", "experiment"),
            data=build(CurnamonaConfig, d.get("data")),
            prior=build(PriorConfig, d.get("prior")),
            model=build(ModelConfig, d.get("model")),
            train=build(TrainConfig, d.get("train")),
            output_dir=d.get("output_dir", "experiments/results"),
        )

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        return cls.from_dict(json.loads(Path(path).read_text()))
