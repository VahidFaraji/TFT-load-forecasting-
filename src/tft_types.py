from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Literal, Mapping, Optional, Sequence

import pandas as pd


DatasetName = Literal["ECL", "PJM", "IRAN"]
EvalMode = Literal["fixed", "rolling"]
InputPolicyName = Literal["manual", "l_factor", "seasonal_l_factor"]
RnnType = Literal["lstm", "gru"]


@dataclass(frozen=True)
class SplitConfig:
    val_size: int
    test_size: int


@dataclass(frozen=True)
class InputPolicyConfig:
    name: InputPolicyName
    l_factor: Optional[int] = None
    seasonal_min: Optional[int] = None
    manual_input_size: Optional[int] = None


@dataclass(frozen=True)
class FeatureSpec:
    target_col: str = "y"
    id_col: str = "unique_id"
    time_col: str = "ds"
    futr_exog_cols: tuple[str, ...] = ()
    hist_exog_cols: tuple[str, ...] = ()
    stat_exog_cols: tuple[str, ...] = ()


@dataclass(frozen=True)
class DatasetFileSpec:
    y_path: Path
    futr_path: Optional[Path] = None
    hist_path: Optional[Path] = None
    static_path: Optional[Path] = None


@dataclass(frozen=True)
class DatasetConfig:
    name: DatasetName
    freq: str
    files: DatasetFileSpec
    features: FeatureSpec
    default_horizons: tuple[int, ...]
    default_split: SplitConfig
    default_input_policy: InputPolicyConfig
    is_single_series: bool = False
    available_futr_variants: tuple[str, ...] = ()
    available_hist_variants: tuple[str, ...] = ()
    available_static_variants: tuple[str, ...] = ()
    extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TftModelConfig:
    hidden_size: int
    n_head: int
    attn_dropout: float
    dropout: float
    n_rnn_layers: int
    rnn_type: RnnType
    learning_rate: float
    max_steps: int
    batch_size: int
    windows_batch_size: int
    scaler_type: str = "robust"
    loss_name: str = "mae"
    precision: str | int = 32
    accelerator: str = "auto"
    devices: int = 1
    num_workers: int = 0
    pin_memory: bool = False
    early_stop_patience_steps: int = -1
    val_check_steps: Optional[int] = None


@dataclass(frozen=True)
class EvalConfig:
    mode: EvalMode
    step_size: int
    n_windows: int


@dataclass(frozen=True)
class RunIOConfig:
    run_date: str
    data_root: Path
    out_root: Path
    save_predictions: bool = True
    save_scaler: bool = True
    save_ckpt: bool = False


@dataclass(frozen=True)
class ExogSelection:
    futr_variant: Optional[str] = None
    hist_variant: Optional[str] = None
    static_variant: Optional[str] = None
    use_futr: bool = False
    use_hist: bool = False
    use_static: bool = False


@dataclass(frozen=True)
class TrainConfig:
    dataset: DatasetName
    horizon: int
    seed: int
    input_size: int
    input_policy: InputPolicyConfig
    dataset_config: DatasetConfig
    model_config: TftModelConfig
    split_config: SplitConfig
    eval_config: EvalConfig
    io_config: RunIOConfig
    exog_selection: ExogSelection
    run_name: str


@dataclass(frozen=True)
class LoadedFrames:
    y_df: pd.DataFrame
    futr_df: Optional[pd.DataFrame] = None
    hist_df: Optional[pd.DataFrame] = None
    static_df: Optional[pd.DataFrame] = None


@dataclass(frozen=True)
class PreparedData:
    full_df: pd.DataFrame
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    fit_df: pd.DataFrame
    futr_df: Optional[pd.DataFrame]
    static_df: Optional[pd.DataFrame]
    ds_train_end: pd.Timestamp
    ds_val_end: pd.Timestamp
    n_timestamps: int
    futr_exog_cols: tuple[str, ...] = ()
    hist_exog_cols: tuple[str, ...] = ()
    stat_exog_cols: tuple[str, ...] = ()
    scaler_stats: Optional[pd.DataFrame] = None


@dataclass(frozen=True)
class RunArtifacts:
    run_root: Path
    predictions_path: Optional[Path] = None
    scaler_stats_path: Optional[Path] = None
    scaler_meta_path: Optional[Path] = None
    metrics_path: Optional[Path] = None
    config_path: Optional[Path] = None
    ckpt_path: Optional[Path] = None


def as_tuple(values: Optional[Sequence[str]]) -> tuple[str, ...]:
    if not values:
        return ()
    return tuple(str(v) for v in values)


def ensure_positive_int(name: str, value: int) -> int:
    if int(value) <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")
    return int(value)


def ensure_non_negative_int(name: str, value: int) -> int:
    if int(value) < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")
    return int(value)


def validate_feature_spec(spec: FeatureSpec) -> None:
    all_cols = (
        [spec.target_col, spec.id_col, spec.time_col]
        + list(spec.futr_exog_cols)
        + list(spec.hist_exog_cols)
        + list(spec.stat_exog_cols)
    )
    seen: set[str] = set()
    dup: set[str] = set()
    for col in all_cols:
        if col in seen:
            dup.add(col)
        seen.add(col)
    if dup:
        raise ValueError(f"Duplicate column names in FeatureSpec: {sorted(dup)}")


def validate_split_config(cfg: SplitConfig) -> None:
    ensure_positive_int("val_size", cfg.val_size)
    ensure_positive_int("test_size", cfg.test_size)


def validate_input_policy_config(cfg: InputPolicyConfig) -> None:
    if cfg.name == "manual":
        if cfg.manual_input_size is None:
            raise ValueError("manual_input_size is required when input policy is 'manual'")
        ensure_positive_int("manual_input_size", cfg.manual_input_size)
        return

    if cfg.l_factor is None:
        raise ValueError(f"l_factor is required when input policy is '{cfg.name}'")
    ensure_positive_int("l_factor", cfg.l_factor)

    if cfg.name == "seasonal_l_factor":
        if cfg.seasonal_min is None:
            raise ValueError("seasonal_min is required when input policy is 'seasonal_l_factor'")
        ensure_positive_int("seasonal_min", cfg.seasonal_min)


def validate_model_config(cfg: TftModelConfig) -> None:
    ensure_positive_int("hidden_size", cfg.hidden_size)
    ensure_positive_int("n_head", cfg.n_head)
    ensure_positive_int("n_rnn_layers", cfg.n_rnn_layers)
    ensure_positive_int("batch_size", cfg.batch_size)
    ensure_positive_int("windows_batch_size", cfg.windows_batch_size)

    if cfg.hidden_size % cfg.n_head != 0:
        raise ValueError(
            f"hidden_size ({cfg.hidden_size}) must be divisible by n_head ({cfg.n_head})"
        )

    if not (0.0 <= float(cfg.dropout) < 1.0):
        raise ValueError(f"dropout must be in [0, 1), got {cfg.dropout}")

    if not (0.0 <= float(cfg.attn_dropout) < 1.0):
        raise ValueError(f"attn_dropout must be in [0, 1), got {cfg.attn_dropout}")

    if float(cfg.learning_rate) <= 0.0:
        raise ValueError(f"learning_rate must be > 0, got {cfg.learning_rate}")


def validate_eval_config(cfg: EvalConfig) -> None:
    if cfg.mode == "fixed":
        ensure_non_negative_int("step_size", cfg.step_size)
        ensure_positive_int("n_windows", cfg.n_windows)
        return

    ensure_positive_int("step_size", cfg.step_size)
    ensure_non_negative_int("n_windows", cfg.n_windows)


def validate_dataset_config(cfg: DatasetConfig) -> None:
    validate_feature_spec(cfg.features)
    validate_split_config(cfg.default_split)
    validate_input_policy_config(cfg.default_input_policy)

    if not cfg.default_horizons:
        raise ValueError(f"{cfg.name}: default_horizons cannot be empty")

    for h in cfg.default_horizons:
        ensure_positive_int("horizon", h)


def validate_train_config(cfg: TrainConfig) -> None:
    validate_dataset_config(cfg.dataset_config)
    validate_model_config(cfg.model_config)
    validate_split_config(cfg.split_config)
    validate_input_policy_config(cfg.input_policy)
    validate_eval_config(cfg.eval_config)

    ensure_positive_int("horizon", cfg.horizon)
    ensure_positive_int("input_size", cfg.input_size)

    if cfg.dataset != cfg.dataset_config.name:
        raise ValueError(
            f"TrainConfig.dataset ({cfg.dataset}) must match "
            f"dataset_config.name ({cfg.dataset_config.name})"
        )