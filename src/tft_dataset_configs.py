from __future__ import annotations

from pathlib import Path
from typing import Mapping

from tft_types import (
    DatasetConfig,
    DatasetFileSpec,
    DatasetName,
    FeatureSpec,
    InputPolicyConfig,
    SplitConfig,
    validate_dataset_config,
)


DEFAULT_HOURLY_SPLIT = SplitConfig(
    val_size=24 * 183,
    test_size=24 * 365,
)

DEFAULT_DAILY_SPLIT = SplitConfig(
    val_size=183,
    test_size=365,
)


def dataset_data_root(project_root: Path, dataset: DatasetName) -> Path:
    return project_root / "datasets" / dataset.upper()


def y_long_path(data_root: Path, dataset: DatasetName) -> Path:
    return data_root / f"{dataset.upper()}__y__long__standard.parquet"


def x_global_path(data_root: Path, dataset: DatasetName, variant: str) -> Path:
    suffix = "standard" if str(variant).lower() == "standard" else str(variant).lower()
    return data_root / f"{dataset.upper()}__x__global__{suffix}.parquet"


def x_hist_path(data_root: Path, dataset: DatasetName, variant: str) -> Path:
    suffix = "standard" if str(variant).lower() == "standard" else str(variant).lower()
    return data_root / f"{dataset.upper()}__x__hist__{suffix}.parquet"


def x_static_path(data_root: Path, dataset: DatasetName, variant: str) -> Path:
    suffix = "standard" if str(variant).lower() == "standard" else str(variant).lower()
    return data_root / f"{dataset.upper()}__x__static__{suffix}.parquet"


def build_dataset_files(project_root: Path, dataset: DatasetName) -> DatasetFileSpec:
    data_root = dataset_data_root(project_root, dataset)
    return DatasetFileSpec(
        y_path=y_long_path(data_root, dataset),
        futr_path=None,
        hist_path=None,
        static_path=None,
    )


def make_ecl_config(project_root: Path) -> DatasetConfig:
    cfg = DatasetConfig(
        name="ECL",
        freq="h",
        files=build_dataset_files(project_root, "ECL"),
        features=FeatureSpec(
            target_col="y",
            id_col="unique_id",
            time_col="ds",
            futr_exog_cols=(),
            hist_exog_cols=(),
            stat_exog_cols=(),
        ),
        default_horizons=(24, 48, 96, 192, 336, 720),
        default_split=DEFAULT_HOURLY_SPLIT,
        default_input_policy=InputPolicyConfig(
            name="seasonal_l_factor",
            l_factor=3,
            seasonal_min=168,
        ),
        is_single_series=False,
        available_futr_variants=("none", "standard", "min6"),
        available_hist_variants=("none",),
        available_static_variants=("none",),
        extra={
            "label": "Electricity Load Diagrams",
            "seasonal_min": 168,
            "default_step_map": {
                24: 1,
                48: 2,
                96: 4,
                192: 8,
                336: 24,
                720: 24,
            },
            "futr_variant_paths": {
                "standard": str(x_global_path(dataset_data_root(project_root, "ECL"), "ECL", "standard")),
                "min6": str(x_global_path(dataset_data_root(project_root, "ECL"), "ECL", "min6")),
            },
            "hist_variant_paths": {},
            "static_variant_paths": {},
        },
    )
    validate_dataset_config(cfg)
    return cfg


def make_pjm_config(project_root: Path) -> DatasetConfig:
    cfg = DatasetConfig(
        name="PJM",
        freq="h",
        files=build_dataset_files(project_root, "PJM"),
        features=FeatureSpec(
            target_col="y",
            id_col="unique_id",
            time_col="ds",
            futr_exog_cols=(),
            hist_exog_cols=(),
            stat_exog_cols=(),
        ),
        default_horizons=(24, 48, 96, 192, 336, 720),
        default_split=DEFAULT_HOURLY_SPLIT,
        default_input_policy=InputPolicyConfig(
            name="seasonal_l_factor",
            l_factor=3,
            seasonal_min=168,
        ),
        is_single_series=True,
        available_futr_variants=("none", "standard", "min6"),
        available_hist_variants=("none",),
        available_static_variants=("none",),
        extra={
            "label": "PJM Load",
            "seasonal_min": 168,
            "default_step_map": {
                24: 1,
                48: 2,
                96: 4,
                192: 8,
                336: 24,
                720: 24,
            },
            "futr_variant_paths": {
                "standard": str(x_global_path(dataset_data_root(project_root, "PJM"), "PJM", "standard")),
                "min6": str(x_global_path(dataset_data_root(project_root, "PJM"), "PJM", "min6")),
            },
            "hist_variant_paths": {},
            "static_variant_paths": {},
        },
    )
    validate_dataset_config(cfg)
    return cfg


def make_iran_config(project_root: Path) -> DatasetConfig:
    cfg = DatasetConfig(
        name="IRAN",
        freq="D",
        files=build_dataset_files(project_root, "IRAN"),
        features=FeatureSpec(
            target_col="y",
            id_col="unique_id",
            time_col="ds",
            futr_exog_cols=(),
            hist_exog_cols=(),
            stat_exog_cols=(),
        ),
        default_horizons=(7, 14, 30),
        default_split=DEFAULT_DAILY_SPLIT,
        default_input_policy=InputPolicyConfig(
            name="seasonal_l_factor",
            l_factor=4,
            seasonal_min=7,
        ),
        is_single_series=True,
        available_futr_variants=("none", "standard", "min6"),
        available_hist_variants=("none",),
        available_static_variants=("none",),
        extra={
            "label": "IRAN Daily Load",
            "seasonal_min": 7,
            "default_step_map": {
                7: 1,
                14: 2,
                30: 7,
            },
            "futr_variant_paths": {
                "standard": str(x_global_path(dataset_data_root(project_root, "IRAN"), "IRAN", "standard")),
                "min6": str(x_global_path(dataset_data_root(project_root, "IRAN"), "IRAN", "min6")),
            },
            "hist_variant_paths": {},
            "static_variant_paths": {},
        },
    )
    validate_dataset_config(cfg)
    return cfg


def build_dataset_configs(project_root: Path) -> dict[str, DatasetConfig]:
    return {
        "ECL": make_ecl_config(project_root),
        "PJM": make_pjm_config(project_root),
        "IRAN": make_iran_config(project_root),
    }


def get_dataset_config(project_root: Path, dataset: str) -> DatasetConfig:
    key = str(dataset).upper()
    configs = build_dataset_configs(project_root)
    if key not in configs:
        raise KeyError(f"Unsupported dataset: {dataset}. Available: {sorted(configs)}")
    return configs[key]


def replace_feature_spec(
    cfg: DatasetConfig,
    *,
    futr_exog_cols: tuple[str, ...] | None = None,
    hist_exog_cols: tuple[str, ...] | None = None,
    stat_exog_cols: tuple[str, ...] | None = None,
) -> DatasetConfig:
    features = FeatureSpec(
        target_col=cfg.features.target_col,
        id_col=cfg.features.id_col,
        time_col=cfg.features.time_col,
        futr_exog_cols=cfg.features.futr_exog_cols if futr_exog_cols is None else futr_exog_cols,
        hist_exog_cols=cfg.features.hist_exog_cols if hist_exog_cols is None else hist_exog_cols,
        stat_exog_cols=cfg.features.stat_exog_cols if stat_exog_cols is None else stat_exog_cols,
    )
    out = DatasetConfig(
        name=cfg.name,
        freq=cfg.freq,
        files=cfg.files,
        features=features,
        default_horizons=cfg.default_horizons,
        default_split=cfg.default_split,
        default_input_policy=cfg.default_input_policy,
        is_single_series=cfg.is_single_series,
        available_futr_variants=cfg.available_futr_variants,
        available_hist_variants=cfg.available_hist_variants,
        available_static_variants=cfg.available_static_variants,
        extra=cfg.extra,
    )
    validate_dataset_config(out)
    return out


def resolve_variant_path(
    cfg: DatasetConfig,
    kind: str,
    variant: str | None,
) -> Path | None:
    variant_norm = "none" if variant is None else str(variant).lower()
    if variant_norm == "none":
        return None

    key = f"{kind}_variant_paths"
    mapping = cfg.extra.get(key, {})
    if not isinstance(mapping, Mapping):
        raise ValueError(f"{cfg.name}: extra['{key}'] must be a mapping")

    raw = mapping.get(variant_norm)
    if raw is None:
        raise KeyError(f"{cfg.name}: unsupported {kind} variant '{variant_norm}'")
    return Path(raw)