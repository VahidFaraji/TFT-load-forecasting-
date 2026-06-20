from __future__ import annotations

from typing import Any, Optional

import pandas as pd
from neuralforecast import NeuralForecast

from tft_types import EvalConfig, PreparedData, validate_eval_config


BASE_COLS = {"unique_id", "ds", "y", "cutoff"}


def select_prediction_col(df: pd.DataFrame) -> str:
    pred_cols = [c for c in df.columns if c not in BASE_COLS]
    if len(pred_cols) == 0:
        raise RuntimeError("No prediction column found.")
    if len(pred_cols) > 1:
        raise RuntimeError(f"Multiple prediction columns found: {pred_cols}")
    return pred_cols[0]


def make_fixed_futr_df(prepared: PreparedData, horizon: int) -> pd.DataFrame | None:
    if not prepared.futr_exog_cols:
        return None

    cols = ["unique_id", "ds", *prepared.futr_exog_cols]
    futr_df = (
        prepared.test_df
        .groupby("unique_id", as_index=False, sort=False)
        .head(int(horizon))[cols]
        .copy()
    )

    if futr_df.empty:
        raise RuntimeError("Fixed futr_df is empty.")

    return futr_df


def _predict_kwargs(
    *,
    fit_df: pd.DataFrame,
    futr_df: Optional[pd.DataFrame],
    static_df: Optional[pd.DataFrame],
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"df": fit_df}
    if futr_df is not None:
        kwargs["futr_df"] = futr_df
    if static_df is not None:
        kwargs["static_df"] = static_df
    return kwargs


def _cross_validation_kwargs(
    *,
    full_df: pd.DataFrame,
    static_df: Optional[pd.DataFrame],
    eval_cfg: EvalConfig,
    val_size: int,
    test_size: int,
    verbose: bool,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "df": full_df,
        "step_size": int(eval_cfg.step_size),
        "refit": False,
        "verbose": bool(verbose),
    }

    if static_df is not None:
        kwargs["static_df"] = static_df

    if int(eval_cfg.n_windows) > 0:
        kwargs["n_windows"] = int(eval_cfg.n_windows)
    else:
        kwargs["n_windows"] = None
        kwargs["val_size"] = int(val_size)
        kwargs["test_size"] = int(test_size)

    return kwargs


def _adjust_rolling_test_size(
    *,
    horizon: int,
    step_size: int,
    test_size: int,
) -> tuple[int, Optional[str]]:
    h = int(horizon)
    s = int(step_size)
    t = int(test_size)

    if t <= h:
        raise ValueError(f"Invalid rolling setup: test_size({t}) must be > horizon({h})")

    max_shift = t - h
    n_windows_valid = (max_shift // s) + 1
    test_eff = h + s * (n_windows_valid - 1)

    if test_eff == t:
        return t, None

    dropped = t - test_eff
    msg = (
        f"[ADJUST] rolling: keep H={h}, step={s}; "
        f"test_size {t}->{test_eff} (drop_last={dropped}, n_windows={n_windows_valid})"
    )
    return test_eff, msg


def _normalize_cv_output(df: pd.DataFrame, pred_col: str) -> pd.DataFrame:
    out = df.copy()

    if pred_col != "yhat":
        out = out.rename(columns={pred_col: "yhat"})

    keep_cols = ["unique_id", "ds", "y", "yhat"]
    if "cutoff" in out.columns:
        keep_cols.append("cutoff")

    out = out[keep_cols].copy()
    out["unique_id"] = out["unique_id"].astype(str)
    out["ds"] = pd.to_datetime(out["ds"])
    out["y"] = out["y"].astype("float32", copy=False)
    out["yhat"] = out["yhat"].astype("float32", copy=False)

    if "cutoff" in out.columns:
        out["cutoff"] = pd.to_datetime(out["cutoff"])

    if out["yhat"].isna().any():
        n_missing = int(out["yhat"].isna().sum())
        raise RuntimeError(
            f"Forecast output contains {n_missing} NaN predictions; "
            f"check ds/freq alignment and future exogenous coverage."
        )

    sort_cols = ["unique_id", "ds"]
    if "cutoff" in out.columns:
        sort_cols.append("cutoff")

    return out.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)


def evaluate_fixed(
    nf: NeuralForecast,
    *,
    prepared: PreparedData,
    horizon: int,
    fixed_cutoff: pd.Timestamp,
) -> pd.DataFrame:
    futr_df = make_fixed_futr_df(prepared, horizon)

    pred_df = nf.predict(
        **_predict_kwargs(
            fit_df=prepared.fit_df,
            futr_df=futr_df,
            static_df=prepared.static_df,
        )
    )

    gt_df = (
        prepared.test_df
        .groupby("unique_id", as_index=False, sort=False)
        .head(int(horizon))[["unique_id", "ds", "y"]]
        .copy()
    )

    cv_df = gt_df.merge(pred_df, on=["unique_id", "ds"], how="left", copy=False)
    pred_col = select_prediction_col(cv_df)
    cv_df["cutoff"] = pd.Timestamp(fixed_cutoff)
    return _normalize_cv_output(cv_df, pred_col)


def evaluate_rolling(
    nf: NeuralForecast,
    *,
    prepared: PreparedData,
    eval_cfg: EvalConfig,
    val_size: int,
    test_size: int,
    horizon: int,
    verbose: bool = False,
) -> pd.DataFrame:
    test_size_eff = int(test_size)

    if int(eval_cfg.n_windows) <= 0:
        test_size_eff, warn = _adjust_rolling_test_size(
            horizon=int(horizon),
            step_size=int(eval_cfg.step_size),
            test_size=int(test_size),
        )
        if warn:
            print(warn)

    kwargs = _cross_validation_kwargs(
        full_df=prepared.full_df,
        static_df=prepared.static_df,
        eval_cfg=eval_cfg,
        val_size=int(val_size),
        test_size=int(test_size_eff),
        verbose=verbose,
    )
    cv_df = nf.cross_validation(**kwargs)
    pred_col = select_prediction_col(cv_df)
    return _normalize_cv_output(cv_df, pred_col)


def evaluate_tft(
    nf: NeuralForecast,
    *,
    prepared: PreparedData,
    eval_cfg: EvalConfig,
    val_size: int,
    test_size: int,
    horizon: int,
    fixed_cutoff: Optional[pd.Timestamp] = None,
    verbose: bool = False,
) -> pd.DataFrame:
    validate_eval_config(eval_cfg)

    if eval_cfg.mode == "fixed":
        if fixed_cutoff is None:
            raise ValueError("fixed evaluation requires fixed_cutoff")
        return evaluate_fixed(
            nf,
            prepared=prepared,
            horizon=int(horizon),
            fixed_cutoff=pd.Timestamp(fixed_cutoff),
        )

    if eval_cfg.mode == "rolling":
        return evaluate_rolling(
            nf,
            prepared=prepared,
            eval_cfg=eval_cfg,
            val_size=int(val_size),
            test_size=int(test_size),
            horizon=int(horizon),
            verbose=verbose,
        )

    raise ValueError(f"Unsupported eval mode: {eval_cfg.mode}")