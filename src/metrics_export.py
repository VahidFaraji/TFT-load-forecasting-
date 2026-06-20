from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


BASE_COLS = ("unique_id", "ds", "y", "cutoff")


# =========================================================
# IO
# =========================================================

def load_predictions_cv(path: Path, *, yhat_col: str = "yhat") -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing predictions file: {path}")

    df = pd.read_parquet(path)

    req = {"unique_id", "ds", "y"}
    miss = req.difference(df.columns)
    if miss:
        raise ValueError(f"predictions missing columns: {sorted(miss)}")

    if yhat_col not in df.columns:
        cand = [c for c in df.columns if c not in BASE_COLS]
        if not cand:
            raise ValueError(f"predictions missing '{yhat_col}' and no candidate yhat column found")
        df = df.rename(columns={cand[0]: yhat_col})

    keep = ["unique_id", "ds", "y", yhat_col] + (["cutoff"] if "cutoff" in df.columns else [])
    df = df[keep].copy()
    df["unique_id"] = df["unique_id"].astype(str)
    df["ds"] = pd.to_datetime(df["ds"])
    if "cutoff" in df.columns:
        df["cutoff"] = pd.to_datetime(df["cutoff"])

    return df.sort_values(["unique_id", "ds"], kind="mergesort").reset_index(drop=True)


def load_scaler_stats(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing scaler stats: {path}")

    df = pd.read_parquet(path)
    req = {"unique_id", "y_mean", "y_std"}
    miss = req.difference(df.columns)
    if miss:
        raise ValueError(f"scaler_stats missing columns: {sorted(miss)}")

    out = df[["unique_id", "y_mean", "y_std"]].copy()
    out["unique_id"] = out["unique_id"].astype(str)
    out["y_mean"] = out["y_mean"].astype(np.float64, copy=False)
    out["y_std"] = out["y_std"].astype(np.float64, copy=False)
    return out


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


# =========================================================
# VIEWS
# =========================================================

def make_view_stitched(df_cv: pd.DataFrame) -> pd.DataFrame:
    if "cutoff" not in df_cv.columns:
        return df_cv[["unique_id", "ds", "y", "yhat"]].copy()

    df = df_cv.sort_values(["unique_id", "ds", "cutoff"], kind="mergesort")
    df = df.drop_duplicates(subset=["unique_id", "ds"], keep="last")
    return df[["unique_id", "ds", "y", "yhat"]].sort_values(["unique_id", "ds"], kind="mergesort").reset_index(drop=True)


def make_view_cutoffmean(df_cv: pd.DataFrame) -> pd.DataFrame:
    if "cutoff" not in df_cv.columns:
        return df_cv[["unique_id", "ds", "y", "yhat"]].copy()

    out = (
        df_cv.groupby(["unique_id", "ds"], as_index=False, sort=False)
        .agg(y=("y", "first"), yhat=("yhat", "mean"))
        .sort_values(["unique_id", "ds"], kind="mergesort")
        .reset_index(drop=True)
    )
    return out


# =========================================================
# CORE METRICS
# =========================================================

def _smape_from_arrays(y: np.ndarray, yhat: np.ndarray, eps: float = 1e-12) -> float:
    y64 = y.astype(np.float64, copy=False)
    yhat64 = yhat.astype(np.float64, copy=False)
    denom = np.abs(y64) + np.abs(yhat64) + eps
    return float(np.mean(2.0 * np.abs(yhat64 - y64) / denom))


def stitched_pooled_metrics(df_cv: pd.DataFrame) -> Dict[str, float]:
    df = make_view_stitched(df_cv)
    y = df["y"].to_numpy(dtype=np.float32, copy=False)
    yhat = df["yhat"].to_numpy(dtype=np.float32, copy=False)

    err = (yhat - y).astype(np.float64, copy=False)
    mae = float(np.mean(np.abs(err)))
    mse = float(np.mean(err * err))
    rmse = float(np.sqrt(mse))
    smape = _smape_from_arrays(y, yhat)

    return {
        "MAE_pooled": mae,
        "RMSE_pooled": rmse,
        "MSE_pooled": mse,
        "sMAPE_pooled": smape,
    }


def _factorize_cutoff(df_cv: pd.DataFrame) -> Tuple[np.ndarray, int]:
    cutoff = df_cv["cutoff"]
    codes, uniques = pd.factorize(cutoff, sort=False)
    return codes.astype(np.int64, copy=False), int(uniques.size)


def rolling_mae_series_stats_streaming(df_cv: pd.DataFrame) -> Dict[str, Optional[float]]:
    if "cutoff" not in df_cv.columns:
        return {
            "MAE_cutoff_series_mean": None,
            "MAE_cutoff_series_std": None,
            "n_cutoffs": None,
        }

    codes, n = _factorize_cutoff(df_cv)
    if n == 0:
        return {
            "MAE_cutoff_series_mean": None,
            "MAE_cutoff_series_std": None,
            "n_cutoffs": 0,
        }

    y = df_cv["y"].to_numpy(dtype=np.float32, copy=False)
    yhat = df_cv["yhat"].to_numpy(dtype=np.float32, copy=False)

    abs_err = np.abs(yhat - y).astype(np.float64, copy=False)
    cnt = np.bincount(codes, minlength=n).astype(np.float64, copy=False)
    sabs = np.bincount(codes, weights=abs_err, minlength=n).astype(np.float64, copy=False)

    mae_by = sabs / np.maximum(1.0, cnt)
    return {
        "MAE_cutoff_series_mean": float(np.mean(mae_by)),
        "MAE_cutoff_series_std": float(np.std(mae_by, ddof=0)),
        "n_cutoffs": int(n),
    }


def cutoffmean_metrics_streaming(df_cv: pd.DataFrame) -> Dict[str, Optional[float]]:
    if "cutoff" not in df_cv.columns:
        m = stitched_pooled_metrics(df_cv)
        return {
            "MAE_cutoff_mean": m["MAE_pooled"],
            "RMSE_cutoff_mean": m["RMSE_pooled"],
            "sMAPE_cutoff_mean": m["sMAPE_pooled"],
        }

    codes, n = _factorize_cutoff(df_cv)
    if n == 0:
        return {
            "MAE_cutoff_mean": None,
            "RMSE_cutoff_mean": None,
            "sMAPE_cutoff_mean": None,
        }

    y = df_cv["y"].to_numpy(dtype=np.float32, copy=False)
    yhat = df_cv["yhat"].to_numpy(dtype=np.float32, copy=False)

    err = (yhat - y).astype(np.float64, copy=False)
    abs_err = np.abs(err)
    sq_err = err * err

    denom = np.abs(y.astype(np.float64, copy=False)) + np.abs(yhat.astype(np.float64, copy=False)) + 1e-12
    sm_obs = 2.0 * abs_err / denom

    cnt = np.bincount(codes, minlength=n).astype(np.float64, copy=False)
    s_abs = np.bincount(codes, weights=abs_err, minlength=n).astype(np.float64, copy=False)
    s_sq = np.bincount(codes, weights=sq_err, minlength=n).astype(np.float64, copy=False)
    s_sm = np.bincount(codes, weights=sm_obs, minlength=n).astype(np.float64, copy=False)

    mae_by = s_abs / np.maximum(1.0, cnt)
    rmse_by = np.sqrt(s_sq / np.maximum(1.0, cnt))
    smape_by = s_sm / np.maximum(1.0, cnt)

    return {
        "MAE_cutoff_mean": float(np.mean(mae_by)),
        "RMSE_cutoff_mean": float(np.mean(rmse_by)),
        "sMAPE_cutoff_mean": float(np.mean(smape_by)),
    }


# =========================================================
# COMPATIBILITY EXPORTS
# =========================================================

def global_metrics_from_arrays(y: np.ndarray, yhat: np.ndarray) -> Dict[str, float]:
    y = np.asarray(y, dtype=np.float64)
    yhat = np.asarray(yhat, dtype=np.float64)

    err = yhat - y
    mse = float(np.mean(err ** 2))
    mae = float(np.mean(np.abs(err)))
    denom = np.abs(y) + np.abs(yhat) + 1e-12
    smape = float(np.mean(2.0 * np.abs(err) / denom))

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "sMAPE": smape,
        "MAE_pooled": mae,
    }


# =========================================================
# DIAGNOSTICS
# =========================================================

def per_series_metrics_fast(df: pd.DataFrame, eps: float = 1e-12) -> pd.DataFrame:
    uid = df["unique_id"].to_numpy()
    y = df["y"].to_numpy(dtype=np.float64, copy=False)
    yhat = df["yhat"].to_numpy(dtype=np.float64, copy=False)

    codes, uniques = pd.factorize(uid, sort=False)
    n = int(uniques.size)

    abs_err = np.abs(yhat - y)
    abs_y = np.abs(y)

    cnt = np.bincount(codes, minlength=n).astype(np.float64, copy=False)
    sum_abs_err = np.bincount(codes, weights=abs_err, minlength=n).astype(np.float64, copy=False)
    sum_abs_y = np.bincount(codes, weights=abs_y, minlength=n).astype(np.float64, copy=False)

    max_abs_err = np.zeros(n, dtype=np.float64)
    np.maximum.at(max_abs_err, codes, abs_err)

    denom = np.maximum(1.0, cnt)
    mae = sum_abs_err / denom
    mean_abs_y = sum_abs_y / denom
    nmae = mae / (mean_abs_y + eps)

    out = pd.DataFrame(
        {
            "unique_id": uniques.astype(str),
            "sum_abs_err": sum_abs_err,
            "mae": mae,
            "max_abs_err": max_abs_err,
            "mean_abs_y": mean_abs_y,
            "nmae": nmae,
        }
    )
    return out.sort_values("mae", ascending=False, kind="mergesort").reset_index(drop=True)


def dist_stats(x: np.ndarray) -> Dict[str, float]:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {
            "median": float("nan"),
            "mean": float("nan"),
            "p90": float("nan"),
            "p95": float("nan"),
        }
    return {
        "median": float(np.median(x)),
        "mean": float(np.mean(x)),
        "p90": float(np.quantile(x, 0.90)),
        "p95": float(np.quantile(x, 0.95)),
    }


def pareto_shares(weights: np.ndarray, shares: Sequence[float] = (0.01, 0.05, 0.10)) -> Dict[str, float]:
    w = weights[np.isfinite(weights)]
    if w.size == 0:
        return {f"top_{int(s * 100)}pct_share": float("nan") for s in shares}

    w = np.sort(w)[::-1]
    total = float(np.sum(w))
    if total <= 0:
        return {f"top_{int(s * 100)}pct_share": 0.0 for s in shares}

    out: Dict[str, float] = {}
    n = w.size
    for s in shares:
        k = max(1, int(np.ceil(float(s) * n)))
        out[f"top_{int(s * 100)}pct_share"] = float(np.sum(w[:k]) / total)
    return out


def diagnostics_from_per_series(per_uid: pd.DataFrame) -> Dict[str, float]:
    nmae = per_uid["nmae"].to_numpy(dtype=np.float64, copy=False)
    max_abs_err = per_uid["max_abs_err"].to_numpy(dtype=np.float64, copy=False)
    sum_abs_err = per_uid["sum_abs_err"].to_numpy(dtype=np.float64, copy=False)

    out: Dict[str, float] = {}
    ds = dist_stats(nmae)
    out["nmae_median"] = ds["median"]
    out["nmae_mean"] = ds["mean"]
    out["nmae_p90"] = ds["p90"]
    out["nmae_p95"] = ds["p95"]

    out["max_abs_err_max"] = float(np.nanmax(max_abs_err)) if max_abs_err.size else float("nan")
    out["max_abs_err_p95"] = float(np.nanquantile(max_abs_err, 0.95)) if max_abs_err.size else float("nan")

    ps = pareto_shares(sum_abs_err, shares=(0.01, 0.05, 0.10))
    out["pareto_top_1pct_share"] = ps["top_1pct_share"]
    out["pareto_top_5pct_share"] = ps["top_5pct_share"]
    out["pareto_top_10pct_share"] = ps["top_10pct_share"]

    return out


# =========================================================
# NORMALIZATION
# =========================================================

def apply_train_zscore(df: pd.DataFrame, scaler_stats: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    stats = scaler_stats.set_index("unique_id")[["y_mean", "y_std"]]

    uid = df["unique_id"].astype(str)
    mu = uid.map(stats["y_mean"]).to_numpy(dtype=np.float64)
    sd = uid.map(stats["y_std"]).to_numpy(dtype=np.float64)

    mu = np.where(np.isfinite(mu), mu, 0.0)
    sd = np.where(np.isfinite(sd) & (sd > 0), sd, 1.0)

    y = df["y"].to_numpy(dtype=np.float64)
    yhat = df["yhat"].to_numpy(dtype=np.float64)

    return (y - mu) / sd, (yhat - mu) / sd


def _scaler_index(scaler_stats: pd.DataFrame) -> pd.DataFrame:
    st = scaler_stats[["unique_id", "y_mean", "y_std"]].copy()
    st["unique_id"] = st["unique_id"].astype(str)
    st["y_mean"] = st["y_mean"].astype(np.float64, copy=False)
    st["y_std"] = st["y_std"].astype(np.float64, copy=False)
    return st.set_index("unique_id")[["y_mean", "y_std"]]


def _scaler_has_full_coverage(df: pd.DataFrame, scaler_stats: pd.DataFrame) -> bool:
    stats_ids = set(scaler_stats["unique_id"].astype(str).unique().tolist())
    df_ids = set(df["unique_id"].astype(str).unique().tolist())
    return df_ids.issubset(stats_ids)


def _map_mu_sd(df: pd.DataFrame, stats_idx: pd.DataFrame) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    uid = df["unique_id"].astype(str)
    mu = uid.map(stats_idx["y_mean"]).to_numpy(dtype=np.float64, copy=False)
    sd = uid.map(stats_idx["y_std"]).to_numpy(dtype=np.float64, copy=False)

    if not (np.isfinite(mu).all() and np.isfinite(sd).all()):
        return None, None
    if not np.all(sd > 0):
        return None, None
    return mu, sd


def _zscore_arrays(y: np.ndarray, yhat: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    y64 = y.astype(np.float64, copy=False)
    yhat64 = yhat.astype(np.float64, copy=False)
    return (y64 - mu) / sd, (yhat64 - mu) / sd


def stitched_pooled_metrics_norm(df_cv: pd.DataFrame, scaler_stats: pd.DataFrame) -> Optional[Dict[str, float]]:
    df = make_view_stitched(df_cv)
    stats_idx = _scaler_index(scaler_stats)
    mu, sd = _map_mu_sd(df, stats_idx)
    if mu is None:
        return None

    y = df["y"].to_numpy(dtype=np.float32, copy=False)
    yhat = df["yhat"].to_numpy(dtype=np.float32, copy=False)
    y_z, yhat_z = _zscore_arrays(y, yhat, mu, sd)

    err = (yhat_z - y_z).astype(np.float64, copy=False)
    mae = float(np.mean(np.abs(err)))
    mse = float(np.mean(err * err))
    rmse = float(np.sqrt(mse))
    smape = _smape_from_arrays(y_z, yhat_z)

    return {
        "MAE_pooled": mae,
        "RMSE_pooled": rmse,
        "MSE_pooled": mse,
        "sMAPE_pooled": smape,
    }


def cutoffmean_metrics_norm_streaming(df_cv: pd.DataFrame, scaler_stats: pd.DataFrame) -> Optional[Dict[str, Optional[float]]]:
    stats_idx = _scaler_index(scaler_stats)
    mu, sd = _map_mu_sd(df_cv, stats_idx)
    if mu is None:
        return None

    y = df_cv["y"].to_numpy(dtype=np.float32, copy=False)
    yhat = df_cv["yhat"].to_numpy(dtype=np.float32, copy=False)
    y_z, yhat_z = _zscore_arrays(y, yhat, mu, sd)

    if "cutoff" not in df_cv.columns:
        err = (yhat_z - y_z).astype(np.float64, copy=False)
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err * err)))
        smape = _smape_from_arrays(y_z, yhat_z)
        return {
            "MAE_cutoff_mean": mae,
            "RMSE_cutoff_mean": rmse,
            "sMAPE_cutoff_mean": smape,
        }

    codes, n = _factorize_cutoff(df_cv)
    if n == 0:
        return {
            "MAE_cutoff_mean": None,
            "RMSE_cutoff_mean": None,
            "sMAPE_cutoff_mean": None,
        }

    err = (yhat_z - y_z).astype(np.float64, copy=False)
    abs_err = np.abs(err)
    sq_err = err * err

    denom = np.abs(y_z) + np.abs(yhat_z) + 1e-12
    sm_obs = 2.0 * abs_err / denom

    cnt = np.bincount(codes, minlength=n).astype(np.float64, copy=False)
    s_abs = np.bincount(codes, weights=abs_err, minlength=n).astype(np.float64, copy=False)
    s_sq = np.bincount(codes, weights=sq_err, minlength=n).astype(np.float64, copy=False)
    s_sm = np.bincount(codes, weights=sm_obs, minlength=n).astype(np.float64, copy=False)

    mae_by = s_abs / np.maximum(1.0, cnt)
    rmse_by = np.sqrt(s_sq / np.maximum(1.0, cnt))
    smape_by = s_sm / np.maximum(1.0, cnt)

    return {
        "MAE_cutoff_mean": float(np.mean(mae_by)),
        "RMSE_cutoff_mean": float(np.mean(rmse_by)),
        "sMAPE_cutoff_mean": float(np.mean(smape_by)),
    }


def rolling_mae_series_stats_norm_streaming(df_cv: pd.DataFrame, scaler_stats: pd.DataFrame) -> Optional[Dict[str, Optional[float]]]:
    if "cutoff" not in df_cv.columns:
        return {
            "MAE_cutoff_series_mean": None,
            "MAE_cutoff_series_std": None,
            "n_cutoffs": None,
        }

    stats_idx = _scaler_index(scaler_stats)
    mu, sd = _map_mu_sd(df_cv, stats_idx)
    if mu is None:
        return None

    codes, n = _factorize_cutoff(df_cv)
    if n == 0:
        return {
            "MAE_cutoff_series_mean": None,
            "MAE_cutoff_series_std": None,
            "n_cutoffs": 0,
        }

    y = df_cv["y"].to_numpy(dtype=np.float32, copy=False)
    yhat = df_cv["yhat"].to_numpy(dtype=np.float32, copy=False)
    y_z, yhat_z = _zscore_arrays(y, yhat, mu, sd)

    abs_err = np.abs(yhat_z - y_z).astype(np.float64, copy=False)
    cnt = np.bincount(codes, minlength=n).astype(np.float64, copy=False)
    sabs = np.bincount(codes, weights=abs_err, minlength=n).astype(np.float64, copy=False)

    mae_by = sabs / np.maximum(1.0, cnt)
    return {
        "MAE_cutoff_series_mean": float(np.mean(mae_by)),
        "MAE_cutoff_series_std": float(np.std(mae_by, ddof=0)),
        "n_cutoffs": int(n),
    }


# =========================================================
# PAYLOAD
# =========================================================

def build_metrics_payload(
    *,
    meta: Dict[str, Any],
    df_cv: pd.DataFrame,
    scaler_stats: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    raw_stitched = stitched_pooled_metrics(df_cv)
    per_uid_raw = per_series_metrics_fast(make_view_stitched(df_cv))
    raw_diag = diagnostics_from_per_series(per_uid_raw)
    raw_stitched.update(raw_diag)

    raw_cutoffmean = cutoffmean_metrics_streaming(df_cv)
    raw_rolling = rolling_mae_series_stats_streaming(df_cv)

    out: Dict[str, Any] = {
        "meta": dict(meta),
        "metrics": {
            "raw": {
                "stitched": raw_stitched,
                "cutoffmean": raw_cutoffmean,
                "rolling": {
                    "MAE_cutoff_series_mean": raw_rolling["MAE_cutoff_series_mean"],
                    "MAE_cutoff_series_std": raw_rolling["MAE_cutoff_series_std"],
                    "n_cutoffs": raw_rolling["n_cutoffs"],
                },
            },
            "norm": {
                "available": False,
                "stitched": None,
                "cutoffmean": None,
                "rolling": None,
            },
        },
    }

    if scaler_stats is None:
        return out

    if not _scaler_has_full_coverage(df_cv, scaler_stats):
        return out

    stitched_n = stitched_pooled_metrics_norm(df_cv, scaler_stats)
    if stitched_n is None:
        return out

    df_norm = make_view_stitched(df_cv).copy()
    y_z, yhat_z = apply_train_zscore(df_norm[["unique_id", "y", "yhat"]], scaler_stats)
    df_norm["y"] = y_z
    df_norm["yhat"] = yhat_z

    per_uid_norm = per_series_metrics_fast(df_norm)
    norm_diag = diagnostics_from_per_series(per_uid_norm)
    stitched_n.update(norm_diag)

    cutoffmean_n = cutoffmean_metrics_norm_streaming(df_cv, scaler_stats)
    rolling_n = rolling_mae_series_stats_norm_streaming(df_cv, scaler_stats)
    if cutoffmean_n is None or rolling_n is None:
        return out

    out["metrics"]["norm"] = {
        "available": True,
        "stitched": stitched_n,
        "cutoffmean": cutoffmean_n,
        "rolling": rolling_n,
    }
    return out


# =========================================================
# METRICS ROW SCHEMA / GROUPING / CSV EXPORT
# =========================================================

def _payload_meta(payload: Dict[str, Any]) -> Dict[str, Any]:
    meta = payload.get("meta")
    return meta if isinstance(meta, dict) else payload


def _raw_blocks(payload: Dict[str, Any]) -> Dict[str, Any]:
    metrics = payload.get("metrics")
    if isinstance(metrics, dict) and isinstance(metrics.get("raw"), dict):
        return metrics["raw"]
    return {"stitched": {}, "cutoffmean": {}, "rolling": {}}


def _norm_blocks(payload: Dict[str, Any]) -> Dict[str, Any]:
    metrics = payload.get("metrics")
    if isinstance(metrics, dict) and isinstance(metrics.get("norm"), dict):
        return metrics["norm"]
    return {"available": False, "stitched": {}, "cutoffmean": {}, "rolling": {}}


def metrics_runs_column_order() -> List[str]:
    return [
        "row_type",
        "dataset",
        "model",
        "mode",
        "eval_mode",
        "H",
        "L_factor",
        "step_size",
        "use_exog",
        "use_futr_exog",
        "use_hist_exog",
        "use_static_exog",
        "seed",
        "MAE_pooled_raw",
        "RMSE_pooled_raw",
        "MSE_pooled_raw",
        "MAE_pooled_norm",
        "RMSE_pooled_norm",
        "MSE_pooled_norm",
        "row_group",
        "row_label",
        "run_date",
        "run_name",
        "exog_variant",
        "futr_variant",
        "hist_variant",
        "static_variant",
        "input_policy",
        "input_size",
        "n_windows",
        "freq",
        "hidden_size",
        "n_head",
        "n_rnn_layers",
        "rnn_type",
        "dropout",
        "attn_dropout",
        "scaler_type",
        "loss_name",
        "val_size",
        "test_size",
        "n_rows",
        "n_uids",
        "is_single_series",
        "min_ds",
        "max_ds",
        "ds_train_end",
        "ds_val_end",
        "norm_available",
        "sMAPE_pooled_raw",
        "MAE_cutoff_mean_raw",
        "RMSE_cutoff_mean_raw",
        "sMAPE_cutoff_mean_raw",
        "MAE_cutoff_series_mean_raw",
        "MAE_cutoff_series_std_raw",
        "n_cutoffs_raw",
        "sMAPE_pooled_norm",
        "MAE_cutoff_mean_norm",
        "RMSE_cutoff_mean_norm",
        "sMAPE_cutoff_mean_norm",
        "MAE_cutoff_series_mean_norm",
        "MAE_cutoff_series_std_norm",
        "n_cutoffs_norm",
        "nmae_median_raw",
        "nmae_mean_raw",
        "nmae_p90_raw",
        "nmae_p95_raw",
        "max_abs_err_max_raw",
        "max_abs_err_p95_raw",
        "pareto_top_1pct_share_raw",
        "pareto_top_5pct_share_raw",
        "pareto_top_10pct_share_raw",
        "nmae_median_norm",
        "nmae_mean_norm",
        "nmae_p90_norm",
        "nmae_p95_norm",
        "max_abs_err_max_norm",
        "max_abs_err_p95_norm",
        "pareto_top_1pct_share_norm",
        "pareto_top_5pct_share_norm",
        "pareto_top_10pct_share_norm",
    ]


def metrics_group_keys() -> List[str]:
    return [
        "dataset",
        "model",
        "mode",
        "eval_mode",
        "use_exog",
        "use_futr_exog",
        "use_hist_exog",
        "use_static_exog",
        "exog_variant",
        "futr_variant",
        "hist_variant",
        "static_variant",
        "H",
        "L_factor",
        "input_policy",
        "input_size",
        "step_size",
        "n_windows",
        "freq",
        "hidden_size",
        "n_head",
        "n_rnn_layers",
        "rnn_type",
        "dropout",
        "attn_dropout",
        "scaler_type",
        "loss_name",
        "val_size",
        "test_size",
    ]


def metrics_numeric_columns() -> List[str]:
    return [
        "MAE_pooled_raw",
        "RMSE_pooled_raw",
        "MSE_pooled_raw",
        "MAE_pooled_norm",
        "RMSE_pooled_norm",
        "MSE_pooled_norm",
        "norm_available",
        "sMAPE_pooled_raw",
        "MAE_cutoff_mean_raw",
        "RMSE_cutoff_mean_raw",
        "sMAPE_cutoff_mean_raw",
        "MAE_cutoff_series_mean_raw",
        "MAE_cutoff_series_std_raw",
        "n_cutoffs_raw",
        "sMAPE_pooled_norm",
        "MAE_cutoff_mean_norm",
        "RMSE_cutoff_mean_norm",
        "sMAPE_cutoff_mean_norm",
        "MAE_cutoff_series_mean_norm",
        "MAE_cutoff_series_std_norm",
        "n_cutoffs_norm",
        "nmae_median_raw",
        "nmae_mean_raw",
        "nmae_p90_raw",
        "nmae_p95_raw",
        "max_abs_err_max_raw",
        "max_abs_err_p95_raw",
        "pareto_top_1pct_share_raw",
        "pareto_top_5pct_share_raw",
        "pareto_top_10pct_share_raw",
        "nmae_median_norm",
        "nmae_mean_norm",
        "nmae_p90_norm",
        "nmae_p95_norm",
        "max_abs_err_max_norm",
        "max_abs_err_p95_norm",
        "pareto_top_1pct_share_norm",
        "pareto_top_5pct_share_norm",
        "pareto_top_10pct_share_norm",
    ]


def _blank_row() -> Dict[str, Any]:
    return {c: None for c in metrics_runs_column_order()}


def _group_signature(row: Dict[str, Any]) -> str:
    parts = []
    for key in metrics_group_keys():
        value = row.get(key)
        parts.append("" if value is None else str(value))
    return " | ".join(parts)


def reorder_metrics_row(row: Dict[str, Any]) -> Dict[str, Any]:
    ordered = {c: row.get(c) for c in metrics_runs_column_order()}
    extra = [c for c in row.keys() if c not in ordered]
    for c in extra:
        ordered[c] = row[c]
    return ordered


def flatten_metrics_row(
    payload: Dict[str, Any],
    *,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    meta = _payload_meta(payload)
    raw = _raw_blocks(payload)
    norm = _norm_blocks(payload)

    raw_st = raw.get("stitched") or {}
    raw_cm = raw.get("cutoffmean") or {}
    raw_roll = raw.get("rolling") or {}

    norm_available = bool(norm.get("available", False))
    norm_st = (norm.get("stitched") or {}) if norm_available else {}
    norm_cm = (norm.get("cutoffmean") or {}) if norm_available else {}
    norm_roll = (norm.get("rolling") or {}) if norm_available else {}

    row = _blank_row()
    row["row_type"] = "seed"
    row["row_label"] = "seed"

    meta_fields = [
        "dataset",
        "model",
        "mode",
        "eval_mode",
        "run_date",
        "run_name",
        "seed",
        "use_exog",
        "use_futr_exog",
        "use_hist_exog",
        "use_static_exog",
        "exog_variant",
        "futr_variant",
        "hist_variant",
        "static_variant",
        "H",
        "L_factor",
        "input_policy",
        "input_size",
        "step_size",
        "n_windows",
        "freq",
        "hidden_size",
        "n_head",
        "n_rnn_layers",
        "rnn_type",
        "dropout",
        "attn_dropout",
        "scaler_type",
        "loss_name",
        "val_size",
        "test_size",
        "n_rows",
        "n_uids",
        "is_single_series",
        "min_ds",
        "max_ds",
        "ds_train_end",
        "ds_val_end",
    ]
    for key in meta_fields:
        row[key] = meta.get(key)

    row["norm_available"] = int(norm_available)

    row["MAE_pooled_raw"] = raw_st.get("MAE_pooled")
    row["RMSE_pooled_raw"] = raw_st.get("RMSE_pooled")
    row["MSE_pooled_raw"] = raw_st.get("MSE_pooled")
    row["sMAPE_pooled_raw"] = raw_st.get("sMAPE_pooled")

    row["MAE_cutoff_mean_raw"] = raw_cm.get("MAE_cutoff_mean")
    row["RMSE_cutoff_mean_raw"] = raw_cm.get("RMSE_cutoff_mean")
    row["sMAPE_cutoff_mean_raw"] = raw_cm.get("sMAPE_cutoff_mean")

    row["MAE_cutoff_series_mean_raw"] = raw_roll.get("MAE_cutoff_series_mean")
    row["MAE_cutoff_series_std_raw"] = raw_roll.get("MAE_cutoff_series_std")
    row["n_cutoffs_raw"] = raw_roll.get("n_cutoffs")

    row["MAE_pooled_norm"] = norm_st.get("MAE_pooled") if norm_available else None
    row["RMSE_pooled_norm"] = norm_st.get("RMSE_pooled") if norm_available else None
    row["MSE_pooled_norm"] = norm_st.get("MSE_pooled") if norm_available else None
    row["sMAPE_pooled_norm"] = norm_st.get("sMAPE_pooled") if norm_available else None

    row["MAE_cutoff_mean_norm"] = norm_cm.get("MAE_cutoff_mean") if norm_available else None
    row["RMSE_cutoff_mean_norm"] = norm_cm.get("RMSE_cutoff_mean") if norm_available else None
    row["sMAPE_cutoff_mean_norm"] = norm_cm.get("sMAPE_cutoff_mean") if norm_available else None

    row["MAE_cutoff_series_mean_norm"] = norm_roll.get("MAE_cutoff_series_mean") if norm_available else None
    row["MAE_cutoff_series_std_norm"] = norm_roll.get("MAE_cutoff_series_std") if norm_available else None
    row["n_cutoffs_norm"] = norm_roll.get("n_cutoffs") if norm_available else None

    row["nmae_median_raw"] = raw_st.get("nmae_median")
    row["nmae_mean_raw"] = raw_st.get("nmae_mean")
    row["nmae_p90_raw"] = raw_st.get("nmae_p90")
    row["nmae_p95_raw"] = raw_st.get("nmae_p95")
    row["max_abs_err_max_raw"] = raw_st.get("max_abs_err_max")
    row["max_abs_err_p95_raw"] = raw_st.get("max_abs_err_p95")
    row["pareto_top_1pct_share_raw"] = raw_st.get("pareto_top_1pct_share")
    row["pareto_top_5pct_share_raw"] = raw_st.get("pareto_top_5pct_share")
    row["pareto_top_10pct_share_raw"] = raw_st.get("pareto_top_10pct_share")

    row["nmae_median_norm"] = norm_st.get("nmae_median") if norm_available else None
    row["nmae_mean_norm"] = norm_st.get("nmae_mean") if norm_available else None
    row["nmae_p90_norm"] = norm_st.get("nmae_p90") if norm_available else None
    row["nmae_p95_norm"] = norm_st.get("nmae_p95") if norm_available else None
    row["max_abs_err_max_norm"] = norm_st.get("max_abs_err_max") if norm_available else None
    row["max_abs_err_p95_norm"] = norm_st.get("max_abs_err_p95") if norm_available else None
    row["pareto_top_1pct_share_norm"] = norm_st.get("pareto_top_1pct_share") if norm_available else None
    row["pareto_top_5pct_share_norm"] = norm_st.get("pareto_top_5pct_share") if norm_available else None
    row["pareto_top_10pct_share_norm"] = norm_st.get("pareto_top_10pct_share") if norm_available else None

    if overrides:
        row.update(overrides)

    row["row_group"] = _group_signature(row)
    return reorder_metrics_row(row)


def select_primary_score(row: Dict[str, Any]) -> float:
    for key in (
        "MAE_pooled_norm",
        "MAE_pooled_raw",
        "MAE_cutoff_mean_norm",
        "MAE_cutoff_mean_raw",
    ):
        value = row.get(key)
        if value is not None and pd.notna(value):
            return float(value)
    return float("inf")


def _aggregate_group_rows(group_df: pd.DataFrame, row_type: str) -> Dict[str, Any]:
    if row_type not in {"mean", "std"}:
        raise ValueError(f"Unsupported row_type: {row_type}")

    row = _blank_row()
    row["row_type"] = row_type
    row["row_label"] = row_type

    numeric_cols = [c for c in metrics_numeric_columns() if c in group_df.columns]
    numeric_df = group_df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    agg = (
        numeric_df.mean(axis=0, skipna=True)
        if row_type == "mean"
        else numeric_df.std(axis=0, ddof=0, skipna=True)
    )

    for col in numeric_cols:
        value = agg.get(col)
        row[col] = None if pd.isna(value) else float(value)

    return reorder_metrics_row(row)


def make_separator_row() -> Dict[str, Any]:
    row = _blank_row()
    row["row_type"] = ""
    row["row_group"] = ""
    row["row_label"] = ""
    return reorder_metrics_row(row)


def build_metrics_runs_table(seed_rows: Sequence[Dict[str, Any]]) -> pd.DataFrame:
    if not seed_rows:
        return pd.DataFrame(columns=metrics_runs_column_order())

    seed_df = pd.DataFrame([reorder_metrics_row(r) for r in seed_rows])
    for col in metrics_runs_column_order():
        if col not in seed_df.columns:
            seed_df[col] = None

    out_rows: List[Dict[str, Any]] = []
    grouped = seed_df.groupby(metrics_group_keys(), dropna=False, sort=False)

    for _, g in grouped:
        g = g.copy().sort_values(["seed", "run_name"], kind="mergesort", na_position="last")
        out_rows.extend(g.to_dict(orient="records"))
        out_rows.append(_aggregate_group_rows(g, "mean"))
        out_rows.append(_aggregate_group_rows(g, "std"))
        out_rows.append(make_separator_row())

    out_df = pd.DataFrame(out_rows)
    for col in metrics_runs_column_order():
        if col not in out_df.columns:
            out_df[col] = None

    return out_df[metrics_runs_column_order()]


def _seed_key_columns() -> List[str]:
    return ["run_date", "run_name", "seed"]


def _merge_seed_rows_with_history(
    old_seed_rows: Sequence[Dict[str, Any]],
    new_seed_rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    key_cols = _seed_key_columns()
    merged: Dict[Tuple[Any, ...], Dict[str, Any]] = {}

    for row in list(old_seed_rows) + list(new_seed_rows):
        key = tuple(row.get(c) for c in key_cols)
        merged[key] = reorder_metrics_row(dict(row))

    return list(merged.values())


def write_metrics_runs_csv(
    csv_path: Path,
    seed_rows: Sequence[Dict[str, Any]],
    *,
    append_history: bool = True,
) -> pd.DataFrame:
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    final_seed_rows: List[Dict[str, Any]] = [reorder_metrics_row(dict(r)) for r in seed_rows]

    if append_history and csv_path.exists():
        old_df = pd.read_csv(csv_path)
        if "row_type" in old_df.columns:
            old_df = old_df[old_df["row_type"] == "seed"].copy()
        old_seed_rows = old_df.to_dict(orient="records")
        final_seed_rows = _merge_seed_rows_with_history(old_seed_rows, final_seed_rows)

    out_df = build_metrics_runs_table(final_seed_rows)
    out_df.to_csv(csv_path, index=False, encoding="utf-8")
    return out_df

def upsert_metrics_runs_csv(
    csv_path: Path,
    payload: Dict[str, Any],
    *,
    overrides: Optional[Dict[str, Any]] = None,
    append_history: bool = True,
) -> pd.DataFrame:
    seed_row = flatten_metrics_row(payload, overrides=overrides)
    return write_metrics_runs_csv(
        csv_path=csv_path,
        seed_rows=[seed_row],
        append_history=append_history,
    )
