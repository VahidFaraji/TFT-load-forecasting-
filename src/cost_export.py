from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd


# =========================================================
# CONSTANTS
# =========================================================

TIMING_RUN_COLS: List[str] = [
    "summary_type",
    "dataset",
    "model",
    "model_family",
    "use_exog",
    "exog_variant",
    "H",
    "L_factor",
    "input_size",
    "batch_size",
    "n_channels",
    "M_batch",
    "memory_proxy_formula",
    "seed",
    "train_seconds",
    "inf_seconds",
    "run_date",
    "run_name",
    "mode",
    "eval_mode",
    "step_size",
    "step_size_spec",
    "n_windows",
    "train_size",
    "val_size",
    "test_size",
    "split_mode",
    "split_source",
]

TIMING_SUMMARY_GROUP_COLS: List[str] = [
    "dataset",
    "model",
    "model_family",
    "use_exog",
    "exog_variant",
    "H",
    "L_factor",
    "input_size",
    "batch_size",
    "n_channels",
    "memory_proxy_formula",
    "mode",
    "eval_mode",
    "step_size",
    "step_size_spec",
    "n_windows",
    "train_size",
    "val_size",
    "test_size",
    "split_mode",
    "split_source",
]

TIMING_VALUE_COLS: List[str] = [
    "train_seconds",
    "inf_seconds",
    "M_batch",
]

COST_BENEFIT_COLS: List[str] = [
    "dataset",
    "model",
    "model_family",
    "H",
    "L_factor",
    "input_size",
    "mode",
    "eval_mode",
    "step_size",
    "step_size_spec",
    "n_windows",
    "train_size",
    "val_size",
    "test_size",
    "split_mode",
    "split_source",
    "exog_variant",
    "MAE_norm_noX",
    "MAE_norm_X",
    "train_seconds_noX",
    "train_seconds_X",
    "inf_seconds_noX",
    "inf_seconds_X",
    "M_batch_noX",
    "M_batch_X",
    "delta_MAE_norm",
    "delta_T_train",
    "delta_T_inf",
    "delta_M_batch",
]


# =========================================================
# BASIC HELPERS
# =========================================================

def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _safe_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if x is None or (isinstance(x, str) and x.strip() == ""):
            return default
        v = float(x)
        return v if np.isfinite(v) else default
    except Exception:
        return default


def _safe_int(x: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        if x is None or (isinstance(x, str) and x.strip() == ""):
            return default
        return int(x)
    except Exception:
        return default


def _safe_div(num: Any, den: Any) -> Optional[float]:
    num_f = _safe_float(num, None)
    den_f = _safe_float(den, None)
    if num_f is None or den_f is None or abs(den_f) < 1e-12:
        return None
    return float(num_f / den_f)


def _read_csv_safe(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _reindex_with_tail(df: pd.DataFrame, ordered_cols: Sequence[str]) -> pd.DataFrame:
    head = [c for c in ordered_cols if c in df.columns]
    tail = [c for c in df.columns if c not in head]
    return df.reindex(columns=head + tail)


def _blank_row_like(columns: Sequence[str]) -> Dict[str, Any]:
    return {c: "" for c in columns}


def _summary_std(series: pd.Series) -> Optional[float]:
    x = pd.to_numeric(series, errors="coerce")
    n = x.notna().sum()
    if n == 0:
        return None
    if n == 1:
        return 0.0
    return float(x.std(ddof=1))


def _summary_mean(series: pd.Series) -> Optional[float]:
    x = pd.to_numeric(series, errors="coerce")
    if x.notna().sum() == 0:
        return None
    return float(x.mean())


def _assert_constant_within_group(g: pd.DataFrame, cols: Sequence[str]) -> None:
    for c in cols:
        if c in g.columns and g[c].astype(str).nunique(dropna=False) > 1:
            raise ValueError(f"Inconsistent column within group: {c}")


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# =========================================================
# BUILD RUN-LEVEL TIMING ROW
# =========================================================

def build_timing_row(
    *,
    meta: Dict[str, Any],
    train_seconds: float,
    inf_seconds: float,
    batch_size: int,
    input_size: int,
    n_channels: int,
    step_size_spec: Optional[int | str] = None,
) -> Dict[str, Any]:
    """
    Shared row-builder for MLP / Transformer pipelines.

    Required:
      - meta must contain dataset/model/use_exog/H/seed/run_name at minimum.
      - batch_size, input_size, n_channels must refer to the actual run config.

    M_batch is a hardware-independent input-memory proxy:
        M_batch = batch_size * input_size * n_channels
    """
    dataset = str(meta.get("dataset", "")).upper()
    model = str(meta.get("model", "")).upper()
    model_family = str(meta.get("model_family", meta.get("family", ""))).upper()
    use_exog = _safe_int(meta.get("use_exog"), 0)
    H = _safe_int(meta.get("H"))
    seed = _safe_int(meta.get("seed"))
    L_factor = _safe_int(meta.get("L_factor"))

    batch_size_i = int(batch_size)
    input_size_i = int(input_size)
    n_channels_i = int(n_channels)

    row: Dict[str, Any] = {
        "summary_type": "RUN",
        "dataset": dataset,
        "model": model,
        "model_family": model_family,
        "use_exog": use_exog,
        "exog_variant": meta.get("exog_variant"),
        "H": H,
        "L_factor": L_factor,
        "input_size": input_size_i,
        "batch_size": batch_size_i,
        "n_channels": n_channels_i,
        "M_batch": float(batch_size_i * input_size_i * n_channels_i),
        "memory_proxy_formula": "batch_size*input_size*n_channels",
        "seed": seed,
        "train_seconds": float(train_seconds),
        "inf_seconds": float(inf_seconds),
        "run_date": meta.get("run_date"),
        "run_name": meta.get("run_name"),
        "mode": meta.get("mode"),
        "eval_mode": meta.get("eval_mode"),
        "step_size": _safe_int(meta.get("step_size")),
        "step_size_spec": meta.get("step_size_spec", step_size_spec),
        "n_windows": _safe_int(meta.get("n_windows")),
        "train_size": _safe_int(meta.get("train_size")),
        "val_size": _safe_int(meta.get("val_size")),
        "test_size": _safe_int(meta.get("test_size")),
        "split_mode": meta.get("split_mode"),
        "split_source": meta.get("split_source"),
    }
    return row


# =========================================================
# RUN-LEVEL CSV UPSERT
# =========================================================

def upsert_timings_runs_csv(
    csv_path: Path,
    row: Dict[str, Any],
    key_cols: Sequence[str],
) -> None:
    """
    Upsert RUN rows only.
    Summary rows are rebuilt later from RUN rows.
    """
    _ensure_parent(csv_path)

    row = dict(row)
    row["summary_type"] = "RUN"

    if not csv_path.exists():
        df = pd.DataFrame([row])
        df = _reindex_with_tail(df, TIMING_RUN_COLS)
        df.to_csv(csv_path, index=False, encoding="utf-8")
        return

    df = pd.read_csv(csv_path)

    if "summary_type" not in df.columns:
        df["summary_type"] = "RUN"

    df = df[df["summary_type"].fillna("RUN") == "RUN"].copy()

    for c in row.keys():
        if c not in df.columns:
            df[c] = np.nan

    row_df = pd.DataFrame([row])
    row_df = _reindex_with_tail(row_df, df.columns)

    def _key(frame: pd.DataFrame) -> pd.Series:
        return frame[list(key_cols)].astype(str).agg("§".join, axis=1)

    if len(df) == 0:
        df = row_df.copy()
    else:
        k_df = _key(df)
        k_row = _key(row_df).iloc[0]
        hit = k_df == k_row

        if hit.any():
            idx = int(np.flatnonzero(hit.to_numpy())[0])
            for c in row_df.columns:
                val = row_df.at[0, c]
                if c in df.columns and df[c].dtype != object and isinstance(val, str):
                    df[c] = df[c].astype(object)
                df.at[idx, c] = val
        else:
            for c in row_df.columns:
                val = row_df.at[0, c]
                if c in df.columns and df[c].dtype != object and isinstance(val, str):
                    df[c] = df[c].astype(object)
            row_df = row_df.reindex(columns=df.columns, fill_value=np.nan)
            df = pd.concat([df, row_df], axis=0, ignore_index=True)

    df = _reindex_with_tail(df, TIMING_RUN_COLS)
    df.to_csv(csv_path, index=False, encoding="utf-8")


# =========================================================
# SUMMARY CSV (RUN / AVG / STD / blank divider)
# =========================================================


def rebuild_timings_runs_with_summary(
    csv_path: Path,
    *,
    group_cols: Optional[Sequence[str]] = None,
    value_cols: Optional[Sequence[str]] = None,
) -> None:
    """
    Rebuilds the file in block form:

      RUN
      RUN
      ...
      AVG
      STD
      <blank row>

    Summary is always computed only from RUN rows.
    """
    df = _read_csv_safe(csv_path)
    if df.empty:
        return

    if "summary_type" not in df.columns:
        df["summary_type"] = "RUN"

    df_run = df[df["summary_type"].fillna("RUN") == "RUN"].copy()
    if df_run.empty:
        return

    group_cols_use = [c for c in (group_cols or TIMING_SUMMARY_GROUP_COLS) if c in df_run.columns]
    value_cols_use = [c for c in (value_cols or TIMING_VALUE_COLS) if c in df_run.columns]

    sort_head = [c for c in ["dataset", "model", "use_exog", "H", "input_size", "seed"] if c in df_run.columns]
    if sort_head:
        df_run = df_run.sort_values(sort_head, kind="mergesort").reset_index(drop=True)

    blocks: List[pd.DataFrame] = []

    for _, g in df_run.groupby(group_cols_use, sort=False, dropna=False):
        g = g.sort_values([c for c in ["seed"] if c in g.columns], kind="mergesort").copy()
        _assert_constant_within_group(g, group_cols_use)
        blocks.append(g)

        avg_row = _blank_row_like(df_run.columns)
        avg_row["summary_type"] = "AVG"
        for c in value_cols_use:
            avg_row[c] = _summary_mean(g[c])

        std_row = _blank_row_like(df_run.columns)
        std_row["summary_type"] = "STD"
        for c in value_cols_use:
            std_row[c] = _summary_std(g[c])

        blocks.append(pd.DataFrame([avg_row]))
        blocks.append(pd.DataFrame([std_row]))
        blocks.append(pd.DataFrame([_blank_row_like(df_run.columns)]))

    df_out = pd.concat(blocks, axis=0, ignore_index=True)
    df_out = _reindex_with_tail(df_out, TIMING_RUN_COLS)
    df_out.to_csv(csv_path, index=False, encoding="utf-8")


def build_timings_summary_csv(
    runs_csv_path: Path,
    out_csv_path: Path,
    *,
    group_cols: Optional[Sequence[str]] = None,
    value_cols: Optional[Sequence[str]] = None,
) -> None:
    """
    Produces one clean AVG row per config, plus *_std columns.
    This file is better suited for article tables than the block-style RUN/AVG/STD file.
    """
    df = _read_csv_safe(runs_csv_path)
    if df.empty:
        return

    if "summary_type" not in df.columns:
        df["summary_type"] = "RUN"

    df_run = df[df["summary_type"].fillna("RUN") == "RUN"].copy()
    if df_run.empty:
        return

    group_cols_use = [c for c in (group_cols or TIMING_SUMMARY_GROUP_COLS) if c in df_run.columns]
    value_cols_use = [c for c in (value_cols or TIMING_VALUE_COLS) if c in df_run.columns]

    rows: List[Dict[str, Any]] = []
    for _, g in df_run.groupby(group_cols_use, sort=False, dropna=False):
        _assert_constant_within_group(g, group_cols_use)
        row: Dict[str, Any] = {c: g.iloc[0][c] for c in group_cols_use}
        for c in value_cols_use:
            row[c] = _summary_mean(g[c])
            row[f"{c}_std"] = _summary_std(g[c])
        row["n_seeds"] = int(len(g))
        rows.append(row)

    out = pd.DataFrame(rows)
    preferred = [
        "dataset", "model", "model_family", "use_exog", "exog_variant", "H", "L_factor",
        "input_size", "batch_size", "n_channels", "memory_proxy_formula", "mode", "eval_mode", "step_size",
        "step_size_spec", "n_windows", "train_size", "val_size", "test_size",
        "split_mode", "split_source",
        "train_seconds", "train_seconds_std",
        "inf_seconds", "inf_seconds_std",
        "M_batch", "M_batch_std",
        "n_seeds",
    ]
    out = _reindex_with_tail(out, preferred)
    _ensure_parent(out_csv_path)
    out.to_csv(out_csv_path, index=False, encoding="utf-8")


# =========================================================
# COST-BENEFIT TABLE FOR ARTICLE
# =========================================================


def build_cost_benefit_csv(
    *,
    metrics_runs_csv: Path,
    timings_runs_csv: Path,
    out_csv_path: Path,
    metric_col: str = "MAE_pooled_norm",
) -> None:
    """
    Builds article-ready cost-benefit rows by pairing:
      use_exog = 0  vs  use_exog = 1

    delta_MAE_norm is defined so that positive values mean improvement:
        delta_MAE_norm = (MAE_noX - MAE_X) / MAE_noX

    Inputs:
      - metrics_runs_csv: metrics file with row_type in {"seed","mean","std",...}
      - timings_runs_csv: RUN/AVG/STD style timings file

    Uses aggregated mean rows when available; otherwise falls back to run rows.
    Always writes a CSV with headers, even when no pairs are available yet.
    """
    df_m = _read_csv_safe(metrics_runs_csv)
    df_t = _read_csv_safe(timings_runs_csv)
    _ensure_parent(out_csv_path)

    if df_m.empty or df_t.empty:
        pd.DataFrame(columns=COST_BENEFIT_COLS).to_csv(out_csv_path, index=False, encoding="utf-8")
        return

    if "row_type" in df_m.columns and (df_m["row_type"] == "mean").any():
        df_m_use = df_m[df_m["row_type"] == "mean"].copy()
    else:
        if "row_type" not in df_m.columns:
            df_m["row_type"] = "seed"
        df_m_use = df_m[df_m["row_type"].fillna("seed") == "seed"].copy()

    if "summary_type" not in df_t.columns:
        df_t["summary_type"] = "RUN"

    if (df_t["summary_type"] == "AVG").any():
        df_t_use = df_t[df_t["summary_type"] == "AVG"].copy()
    else:
        df_t_use = df_t[df_t["summary_type"].fillna("RUN") == "RUN"].copy()

    join_keys = [
        "dataset", "model", "model_family", "use_exog", "exog_variant",
        "H", "L_factor", "input_size", "mode", "eval_mode",
        "step_size", "step_size_spec", "n_windows",
        "train_size", "val_size", "test_size", "split_mode", "split_source",
    ]
    join_keys = [c for c in join_keys if c in df_m_use.columns and c in df_t_use.columns]

    keep_metrics = join_keys + [metric_col]
    keep_timings = join_keys + ["train_seconds", "inf_seconds", "M_batch"]

    missing_metric = metric_col not in df_m_use.columns
    if missing_metric:
        pd.DataFrame(columns=COST_BENEFIT_COLS).to_csv(out_csv_path, index=False, encoding="utf-8")
        return

    df_m_use = df_m_use[keep_metrics].copy()
    df_t_use = df_t_use[keep_timings].copy()

    df_mt = pd.merge(df_m_use, df_t_use, on=join_keys, how="inner")

    pair_keys = [
        "dataset", "model", "model_family", "H", "L_factor", "input_size",
        "mode", "eval_mode", "step_size", "step_size_spec", "n_windows",
        "train_size", "val_size", "test_size", "split_mode", "split_source",
    ]
    pair_keys = [c for c in pair_keys if c in df_mt.columns]

    left_noX = df_mt[df_mt["use_exog"] == 0].copy()
    left_X = df_mt[df_mt["use_exog"] == 1].copy()

    if "exog_variant" not in left_noX.columns:
        left_noX["exog_variant"] = None
    if "exog_variant" not in left_X.columns:
        left_X["exog_variant"] = None

    merged = pd.merge(
        left_noX,
        left_X,
        on=pair_keys,
        how="inner",
        suffixes=("_noX", "_X"),
        validate="one_to_many",
    )

    rows: List[Dict[str, Any]] = []
    for _, r in merged.iterrows():
        mae_noX = _safe_float(r.get(f"{metric_col}_noX"))
        mae_X = _safe_float(r.get(f"{metric_col}_X"))
        tr_noX = _safe_float(r.get("train_seconds_noX"))
        tr_X = _safe_float(r.get("train_seconds_X"))
        inf_noX = _safe_float(r.get("inf_seconds_noX"))
        inf_X = _safe_float(r.get("inf_seconds_X"))
        mem_noX = _safe_float(r.get("M_batch_noX"))
        mem_X = _safe_float(r.get("M_batch_X"))

        row = {
            "dataset": r.get("dataset"),
            "model": r.get("model"),
            "model_family": r.get("model_family"),
            "H": r.get("H"),
            "L_factor": r.get("L_factor"),
            "input_size": r.get("input_size"),
            "mode": r.get("mode"),
            "eval_mode": r.get("eval_mode"),
            "step_size": r.get("step_size"),
            "step_size_spec": r.get("step_size_spec"),
            "n_windows": r.get("n_windows"),
            "train_size": r.get("train_size"),
            "val_size": r.get("val_size"),
            "test_size": r.get("test_size"),
            "split_mode": r.get("split_mode"),
            "split_source": r.get("split_source"),
            "exog_variant": r.get("exog_variant_X"),
            "MAE_norm_noX": mae_noX,
            "MAE_norm_X": mae_X,
            "train_seconds_noX": tr_noX,
            "train_seconds_X": tr_X,
            "inf_seconds_noX": inf_noX,
            "inf_seconds_X": inf_X,
            "M_batch_noX": mem_noX,
            "M_batch_X": mem_X,
            "delta_MAE_norm": _safe_div((mae_noX - mae_X) if mae_X is not None and mae_noX is not None else None, mae_noX),
            "delta_T_train": _safe_div((tr_X - tr_noX) if tr_X is not None and tr_noX is not None else None, tr_noX),
            "delta_T_inf": _safe_div((inf_X - inf_noX) if inf_X is not None and inf_noX is not None else None, inf_noX),
            "delta_M_batch": _safe_div((mem_X - mem_noX) if mem_X is not None and mem_noX is not None else None, mem_noX),
        }
        rows.append(row)

    out = pd.DataFrame(rows, columns=COST_BENEFIT_COLS)
    out = _reindex_with_tail(out, COST_BENEFIT_COLS)
    out.to_csv(out_csv_path, index=False, encoding="utf-8")


# =========================================================
# OPTIONAL: BUILD FROM metrics.json
# =========================================================

def build_timing_row_from_metrics_json(
    metrics_json_path: Path,
    *,
    train_seconds: float,
    inf_seconds: float,
    batch_size: int,
    input_size: int,
    n_channels: int,
    step_size_spec: Optional[int | str] = None,
) -> Dict[str, Any]:
    payload = load_json(metrics_json_path)
    meta = payload.get("meta", payload)
    return build_timing_row(
        meta=meta,
        train_seconds=train_seconds,
        inf_seconds=inf_seconds,
        batch_size=batch_size,
        input_size=input_size,
        n_channels=n_channels,
        step_size_spec=step_size_spec,
    )
