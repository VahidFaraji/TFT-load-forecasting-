from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from tft_dataset_configs import resolve_variant_path
from tft_types import (
    DatasetConfig,
    ExogSelection,
    FeatureSpec,
    LoadedFrames,
    PreparedData,
    SplitConfig,
)


def load_y_frame(cfg: DatasetConfig) -> pd.DataFrame:
    path = cfg.files.y_path
    if not path.exists():
        raise FileNotFoundError(f"{cfg.name}: missing y parquet: {path}")

    spec = cfg.features
    df = pd.read_parquet(path)

    required = {spec.id_col, spec.time_col, spec.target_col}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{cfg.name}: y parquet missing columns: {sorted(missing)}")

    out = df[[spec.id_col, spec.time_col, spec.target_col]].copy()
    out[spec.id_col] = out[spec.id_col].astype(str)
    out[spec.time_col] = pd.to_datetime(out[spec.time_col])
    out[spec.target_col] = out[spec.target_col].astype("float32", copy=False)

    return out.sort_values([spec.id_col, spec.time_col], kind="mergesort").reset_index(drop=True)


def temporal_exog_cols(df: pd.DataFrame, spec: FeatureSpec) -> list[str]:
    return [c for c in df.columns if c not in {spec.id_col, spec.time_col, spec.target_col}]


def static_exog_cols(df: pd.DataFrame, spec: FeatureSpec) -> list[str]:
    return [c for c in df.columns if c != spec.id_col]


def load_temporal_exog_frame(
    path: Path,
    *,
    cfg: DatasetConfig,
    kind: str,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{cfg.name}: missing {kind} exog parquet: {path}")

    spec = cfg.features
    df = pd.read_parquet(path)

    if spec.time_col not in df.columns:
        raise ValueError(f"{cfg.name}: {kind} exog must include '{spec.time_col}'")

    out = df.copy()
    if spec.id_col in out.columns:
        out[spec.id_col] = out[spec.id_col].astype(str)
    out[spec.time_col] = pd.to_datetime(out[spec.time_col])

    exog_cols = temporal_exog_cols(out, spec)
    if not exog_cols:
        raise ValueError(f"{cfg.name}: {kind} exog file has no feature columns: {path}")

    for col in exog_cols:
        if pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].astype("float32", copy=False)

    sort_cols = [spec.time_col] if spec.id_col not in out.columns else [spec.id_col, spec.time_col]
    return out.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)


def load_static_exog_frame(
    path: Path,
    *,
    cfg: DatasetConfig,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{cfg.name}: missing static exog parquet: {path}")

    spec = cfg.features
    df = pd.read_parquet(path)

    if spec.id_col not in df.columns:
        raise ValueError(f"{cfg.name}: static exog must include '{spec.id_col}'")

    exog_cols = static_exog_cols(df, spec)
    if not exog_cols:
        raise ValueError(f"{cfg.name}: static exog file has no feature columns: {path}")

    out = df[[spec.id_col] + exog_cols].copy()
    out[spec.id_col] = out[spec.id_col].astype(str)

    dup = out.duplicated(subset=[spec.id_col], keep=False)
    if dup.any():
        n_dup = int(dup.sum())
        raise ValueError(f"{cfg.name}: static exog has duplicate ids in {n_dup} rows")

    for col in exog_cols:
        if pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].astype("float32", copy=False)

    return out.sort_values(spec.id_col, kind="mergesort").reset_index(drop=True)


def load_exog_frames(
    cfg: DatasetConfig,
    exog_selection: ExogSelection,
) -> LoadedFrames:
    y_df = load_y_frame(cfg)

    futr_df: Optional[pd.DataFrame] = None
    hist_df: Optional[pd.DataFrame] = None
    static_df: Optional[pd.DataFrame] = None

    if exog_selection.use_futr:
        futr_path = resolve_variant_path(cfg, "futr", exog_selection.futr_variant)
        if futr_path is None:
            raise ValueError(f"{cfg.name}: futr exog is enabled but no futr variant path was resolved")
        futr_df = load_temporal_exog_frame(futr_path, cfg=cfg, kind="futr")

    if exog_selection.use_hist:
        hist_path = resolve_variant_path(cfg, "hist", exog_selection.hist_variant)
        if hist_path is None:
            raise ValueError(f"{cfg.name}: hist exog is enabled but no hist variant path was resolved")
        hist_df = load_temporal_exog_frame(hist_path, cfg=cfg, kind="hist")

    if exog_selection.use_static:
        static_path = resolve_variant_path(cfg, "static", exog_selection.static_variant)
        if static_path is None:
            raise ValueError(f"{cfg.name}: static exog is enabled but no static variant path was resolved")
        static_df = load_static_exog_frame(static_path, cfg=cfg)

    return LoadedFrames(
        y_df=y_df,
        futr_df=futr_df,
        hist_df=hist_df,
        static_df=static_df,
    )


def validate_y_panel(df: pd.DataFrame, cfg: DatasetConfig) -> None:
    spec = cfg.features

    if df.empty:
        raise ValueError(f"{cfg.name}: y_df is empty")

    dup = df.duplicated(subset=[spec.id_col, spec.time_col], keep=False)
    if dup.any():
        n_dup = int(dup.sum())
        raise ValueError(f"{cfg.name}: duplicate ({spec.id_col}, {spec.time_col}) rows in y_df: {n_dup}")

    if df[spec.target_col].isna().any():
        n_nan = int(df[spec.target_col].isna().sum())
        raise ValueError(f"{cfg.name}: y_df has NaN target values in {n_nan} rows")

    per_id_sorted = (
        df.groupby(spec.id_col, sort=False)[spec.time_col]
        .apply(lambda s: bool(s.is_monotonic_increasing))
    )
    if not bool(per_id_sorted.all()):
        bad_ids = per_id_sorted[~per_id_sorted].index.tolist()[:10]
        raise ValueError(f"{cfg.name}: timestamps are not monotonic for ids: {bad_ids}")


def validate_temporal_exog_frame(
    exog_df: pd.DataFrame,
    *,
    cfg: DatasetConfig,
    kind: str,
) -> None:
    spec = cfg.features
    exog_cols = temporal_exog_cols(exog_df, spec)

    if exog_df.empty:
        raise ValueError(f"{cfg.name}: {kind} exog frame is empty")

    subset = [spec.time_col] if spec.id_col not in exog_df.columns else [spec.id_col, spec.time_col]
    dup = exog_df.duplicated(subset=subset, keep=False)
    if dup.any():
        n_dup = int(dup.sum())
        raise ValueError(f"{cfg.name}: duplicate keys in {kind} exog frame: {n_dup}")

    if exog_df[exog_cols].isna().any().any():
        bad = int(exog_df[exog_cols].isna().any(axis=1).sum())
        raise ValueError(f"{cfg.name}: {kind} exog frame contains NaN feature rows: {bad}")


def merge_temporal_exog(
    base_df: pd.DataFrame,
    exog_df: pd.DataFrame,
    *,
    cfg: DatasetConfig,
    kind: str,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    spec = cfg.features
    exog_cols = temporal_exog_cols(exog_df, spec)

    if spec.id_col in exog_df.columns:
        keys = [spec.id_col, spec.time_col]
    else:
        keys = [spec.time_col]

    out = base_df.merge(exog_df, on=keys, how="left", copy=False)

    if out[exog_cols].isna().any().any():
        bad = int(out[exog_cols].isna().any(axis=1).sum())
        raise RuntimeError(
            f"{cfg.name}: {kind} exog merge produced NaN in {bad} rows; "
            f"{kind} exog must cover all required keys"
        )

    for col in exog_cols:
        if pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].astype("float32", copy=False)

    return out, tuple(exog_cols)


def validate_static_alignment(
    y_df: pd.DataFrame,
    static_df: pd.DataFrame,
    *,
    cfg: DatasetConfig,
) -> tuple[str, ...]:
    spec = cfg.features
    cols = tuple(static_exog_cols(static_df, spec))

    y_ids = set(y_df[spec.id_col].astype(str).unique().tolist())
    s_ids = set(static_df[spec.id_col].astype(str).unique().tolist())

    missing = sorted(y_ids.difference(s_ids))
    if missing:
        preview = missing[:10]
        raise ValueError(f"{cfg.name}: static exog missing ids: {preview}")

    return cols


def build_full_panel(
    frames: LoadedFrames,
    *,
    cfg: DatasetConfig,
) -> tuple[pd.DataFrame, Optional[pd.DataFrame], tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    validate_y_panel(frames.y_df, cfg)

    full_df = frames.y_df.copy()
    futr_cols: tuple[str, ...] = ()
    hist_cols: tuple[str, ...] = ()
    stat_cols: tuple[str, ...] = ()

    if frames.futr_df is not None:
        validate_temporal_exog_frame(frames.futr_df, cfg=cfg, kind="futr")
        full_df, futr_cols = merge_temporal_exog(full_df, frames.futr_df, cfg=cfg, kind="futr")

    if frames.hist_df is not None:
        validate_temporal_exog_frame(frames.hist_df, cfg=cfg, kind="hist")
        full_df, hist_cols = merge_temporal_exog(full_df, frames.hist_df, cfg=cfg, kind="hist")

    static_df: Optional[pd.DataFrame] = None
    if frames.static_df is not None:
        stat_cols = validate_static_alignment(frames.y_df, frames.static_df, cfg=cfg)
        static_df = frames.static_df.copy()

    full_df = full_df.sort_values(
        [cfg.features.id_col, cfg.features.time_col],
        kind="mergesort",
    ).reset_index(drop=True)

    return full_df, static_df, futr_cols, hist_cols, stat_cols


def split_by_time(
    df: pd.DataFrame,
    *,
    cfg: DatasetConfig,
    split_cfg: SplitConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Timestamp, pd.Timestamp, int]:
    time_col = cfg.features.time_col
    ds_sorted = pd.Index(sorted(df[time_col].unique()))
    n_ts = int(len(ds_sorted))

    if split_cfg.val_size + split_cfg.test_size >= n_ts:
        raise ValueError(
            f"{cfg.name}: val_size + test_size must be < total timestamps ({n_ts}), "
            f"got {split_cfg.val_size + split_cfg.test_size}"
        )

    i_train_end = n_ts - (split_cfg.val_size + split_cfg.test_size)
    i_val_end = n_ts - split_cfg.test_size

    ds_train_end = pd.Timestamp(ds_sorted[i_train_end - 1])
    ds_val_end = pd.Timestamp(ds_sorted[i_val_end - 1])

    train_df = df[df[time_col] <= ds_train_end].copy()
    val_df = df[(df[time_col] > ds_train_end) & (df[time_col] <= ds_val_end)].copy()
    test_df = df[df[time_col] > ds_val_end].copy()

    return train_df, val_df, test_df, ds_train_end, ds_val_end, n_ts


def compute_per_series_scaler_stats(
    train_df: pd.DataFrame,
    *,
    cfg: DatasetConfig,
) -> pd.DataFrame:
    spec = cfg.features
    grouped = train_df.groupby(spec.id_col, sort=False, observed=False)[spec.target_col]
    stats = grouped.agg(["mean", "std"]).reset_index()

    stats.rename(
        columns={
            spec.id_col: "unique_id",
            "mean": "y_mean",
            "std": "y_std",
        },
        inplace=True,
    )

    stats["unique_id"] = stats["unique_id"].astype(str)
    stats["y_mean"] = stats["y_mean"].astype("float32", copy=False)
    stats["y_std"] = stats["y_std"].astype("float32", copy=False)
    stats.loc[stats["y_std"] <= 0, "y_std"] = 1.0

    return stats[["unique_id", "y_mean", "y_std"]]


def write_scaler_artifacts(
    run_root: Path,
    *,
    stats_df: pd.DataFrame,
    meta: dict[str, Any],
) -> None:
    run_root.mkdir(parents=True, exist_ok=True)
    stats_df.to_parquet(run_root / "scaler_stats.parquet", index=False)
    with open(run_root / "scaler_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def scaler_meta_payload(
    *,
    cfg: DatasetConfig,
    split_cfg: SplitConfig,
    exog_selection: ExogSelection,
    full_df: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    ds_train_end: pd.Timestamp,
    ds_val_end: pd.Timestamp,
    n_timestamps: int,
    futr_exog_cols: tuple[str, ...],
    hist_exog_cols: tuple[str, ...],
    stat_exog_cols: tuple[str, ...],
) -> dict[str, Any]:
    spec = cfg.features

    return {
        "dataset": cfg.name,
        "freq": cfg.freq,
        "scaler_type": "train_only_per_series_stats",
        "stats_scope": "train_only_per_series",
        "min_ds": pd.Timestamp(full_df[spec.time_col].min()).isoformat(),
        "max_ds": pd.Timestamp(full_df[spec.time_col].max()).isoformat(),
        "n_ts_total": int(n_timestamps),
        "n_series": int(full_df[spec.id_col].nunique()),
        "full_rows": int(len(full_df)),
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "val_size": int(split_cfg.val_size),
        "test_size": int(split_cfg.test_size),
        "ds_train_end": pd.Timestamp(ds_train_end).isoformat(),
        "ds_val_end": pd.Timestamp(ds_val_end).isoformat(),
        "use_futr_exog": bool(exog_selection.use_futr),
        "use_hist_exog": bool(exog_selection.use_hist),
        "use_static_exog": bool(exog_selection.use_static),
        "futr_variant": exog_selection.futr_variant,
        "hist_variant": exog_selection.hist_variant,
        "static_variant": exog_selection.static_variant,
        "futr_exog_cols": list(futr_exog_cols),
        "hist_exog_cols": list(hist_exog_cols),
        "stat_exog_cols": list(stat_exog_cols),
    }


def prepare_data(
    *,
    cfg: DatasetConfig,
    split_cfg: SplitConfig,
    exog_selection: ExogSelection,
) -> PreparedData:
    frames = load_exog_frames(cfg, exog_selection)

    full_df, static_df, futr_cols, hist_cols, stat_cols = build_full_panel(frames, cfg=cfg)

    train_df, val_df, test_df, ds_train_end, ds_val_end, n_ts = split_by_time(
        full_df,
        cfg=cfg,
        split_cfg=split_cfg,
    )

    fit_df = pd.concat([train_df, val_df], axis=0, ignore_index=True)
    # Always compute scaler_stats for normalized metrics, regardless of whether
    # scaler artifacts are later persisted to disk.
    scaler_stats = compute_per_series_scaler_stats(train_df, cfg=cfg)

    return PreparedData(
        full_df=full_df,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        fit_df=fit_df,
        futr_df=None,
        static_df=static_df,
        ds_train_end=ds_train_end,
        ds_val_end=ds_val_end,
        n_timestamps=n_ts,
        futr_exog_cols=futr_cols,
        hist_exog_cols=hist_cols,
        stat_exog_cols=stat_cols,
        scaler_stats=scaler_stats,
    )