from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import torch
import time
from neuralforecast import NeuralForecast

import metrics_export as mx
from tft_data_module import prepare_data, scaler_meta_payload, write_scaler_artifacts
from tft_dataset_configs import get_dataset_config
from tft_eval import evaluate_tft
from tft_input_policy import resolve_input_size, validate_resolved_input_size
from tft_model_factory import build_tft_model
from tft_paths import (
    build_run_artifacts,
    build_run_name,
    build_run_root,
    default_data_root,
    default_experiments_root,
    ensure_run_root,
    project_root_from_file,
    today_run_date,
)
from tft_types import (
    EvalConfig,
    ExogSelection,
    InputPolicyConfig,
    RunIOConfig,
    SplitConfig,
    TftModelConfig,
    validate_eval_config,
    validate_model_config,
    validate_split_config,
)

from cost_export import (
    build_timing_row,
    upsert_timings_runs_csv,
    rebuild_timings_runs_with_summary,
    build_timings_summary_csv,
    build_cost_benefit_csv,
)

warnings.filterwarnings(
    "ignore",
    message="TypedStorage is deprecated.*",
    category=UserWarning,
)


USE_GPU = torch.cuda.is_available()

MODE_CONFIGS: Dict[str, Dict[str, Any]] = {
    "debug": {
        "hidden_size": 32,
        "n_head": 4,
        "attn_dropout": 0.0,
        "dropout": 0.1,
        "n_rnn_layers": 1,
        "rnn_type": "lstm",
        "learning_rate": 1e-3,
        "max_steps": 200,
        "batch_size": 16 if USE_GPU else 8,
        "windows_batch_size": 8 if USE_GPU else 4,
        "num_workers": 4 if USE_GPU else 0,
        "pin_memory": True if USE_GPU else False,
    },
    "lite": {
        "hidden_size": 64,
        "n_head": 4,
        "attn_dropout": 0.0,
        "dropout": 0.1,
        "n_rnn_layers": 1,
        "rnn_type": "lstm",
        "learning_rate": 1e-3,
        "max_steps": 800,
        "batch_size": 16 if USE_GPU else 8,
        "windows_batch_size": 8 if USE_GPU else 4,
        "num_workers": 4 if USE_GPU else 0,
        "pin_memory": True if USE_GPU else False,
    },
    "full": {
        "hidden_size": 128,
        "n_head": 4,
        "attn_dropout": 0.0,
        "dropout": 0.1,
        "n_rnn_layers": 1,
        "rnn_type": "lstm",
        "learning_rate": 7e-4 if USE_GPU else 1e-3,
        "max_steps": 1500 if USE_GPU else 800,
        "batch_size": 8 if USE_GPU else 4,
        "windows_batch_size": 8 if USE_GPU else 4,
        "num_workers": 4 if USE_GPU else 0,
        "pin_memory": True if USE_GPU else False,
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--dataset", type=str, choices=["ECL", "PJM", "IRAN"], required=True)
    p.add_argument("--mode", type=str, choices=list(MODE_CONFIGS.keys()), default="debug")

    p.add_argument("--run_date", type=str, default=None)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--H", type=int, required=True)

    p.add_argument("--input_policy", type=str, choices=["manual", "l_factor", "seasonal_l_factor"], default=None)
    p.add_argument("--L_factor", type=int, default=None)
    p.add_argument("--seasonal_min", type=int, default=None)
    p.add_argument("--input_size", type=int, default=None)

    p.add_argument("--eval_mode", type=str, choices=["fixed", "rolling"], default="rolling")
    p.add_argument("--step_size", type=int, default=None)
    p.add_argument("--n_windows", type=int, default=-1)

    p.add_argument("--val_size", type=int, default=None)
    p.add_argument("--test_size", type=int, default=None)

    p.add_argument("--use_futr_exog", type=int, choices=[0, 1], default=0)
    p.add_argument("--use_hist_exog", type=int, choices=[0, 1], default=0)
    p.add_argument("--use_static_exog", type=int, choices=[0, 1], default=0)

    p.add_argument("--futr_variant", type=str, default="none")
    p.add_argument("--hist_variant", type=str, default="none")
    p.add_argument("--static_variant", type=str, default="none")

    p.add_argument("--hidden_size", type=int, default=None)
    p.add_argument("--n_head", type=int, default=None)
    p.add_argument("--attn_dropout", type=float, default=None)
    p.add_argument("--dropout", type=float, default=None)
    p.add_argument("--n_rnn_layers", type=int, default=None)
    p.add_argument("--rnn_type", type=str, choices=["lstm", "gru"], default=None)
    p.add_argument("--learning_rate", type=float, default=None)
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--windows_batch_size", type=int, default=None)
    p.add_argument("--scaler_type", type=str, default="robust")
    p.add_argument("--loss_name", type=str, default="mae")
    p.add_argument("--early_stop_patience_steps", type=int, default=-1)
    p.add_argument("--val_check_steps", type=int, default=None)

    p.add_argument("--accelerator", type=str, default="auto")
    p.add_argument("--devices", type=int, default=1)
    p.add_argument("--precision", type=str, default=None)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--pin_memory", type=int, choices=[0, 1], default=None)

    p.add_argument("--save_predictions", type=int, choices=[0, 1], default=1)
    p.add_argument("--save_scaler", type=int, choices=[0, 1], default=1)
    p.add_argument("--save_ckpt", type=int, choices=[0, 1], default=0)
    p.add_argument("--export_timings", type=int, choices=[0, 1], default=1)
    return p.parse_args()


def infer_precision(user_value: Optional[str]) -> str | int:
    if user_value is not None:
        return user_value
    return "16-mixed" if torch.cuda.is_available() else 32


def build_input_policy_config(args: argparse.Namespace, dataset_cfg) -> InputPolicyConfig:
    policy_name = str(args.input_policy).lower() if args.input_policy else dataset_cfg.default_input_policy.name

    if policy_name == "manual":
        if args.input_size is None:
            raise ValueError("--input_size is required when --input_policy=manual")
        return InputPolicyConfig(
            name="manual",
            manual_input_size=int(args.input_size),
        )

    if policy_name == "l_factor":
        l_factor = args.L_factor if args.L_factor is not None else dataset_cfg.default_input_policy.l_factor
        if l_factor is None:
            raise ValueError("L_factor is required for input_policy=l_factor")
        return InputPolicyConfig(
            name="l_factor",
            l_factor=int(l_factor),
        )

    if policy_name == "seasonal_l_factor":
        l_factor = args.L_factor if args.L_factor is not None else dataset_cfg.default_input_policy.l_factor
        seasonal_min = args.seasonal_min if args.seasonal_min is not None else dataset_cfg.default_input_policy.seasonal_min
        if l_factor is None:
            raise ValueError("L_factor is required for input_policy=seasonal_l_factor")
        return InputPolicyConfig(
            name="seasonal_l_factor",
            l_factor=int(l_factor),
            seasonal_min=None if seasonal_min is None else int(seasonal_min),
        )

    raise ValueError(f"Unsupported input policy: {policy_name}")


def build_split_config(args: argparse.Namespace, dataset_cfg) -> SplitConfig:
    cfg = SplitConfig(
        val_size=int(args.val_size if args.val_size is not None else dataset_cfg.default_split.val_size),
        test_size=int(args.test_size if args.test_size is not None else dataset_cfg.default_split.test_size),
    )
    validate_split_config(cfg)
    return cfg


def build_eval_config(args: argparse.Namespace) -> EvalConfig:
    mode = str(args.eval_mode).lower()

    if mode == "fixed":
        cfg = EvalConfig(
            mode="fixed",
            step_size=0 if args.step_size is None else int(args.step_size),
            n_windows=1,
        )
        validate_eval_config(cfg)
        return cfg

    step_size = 1 if args.step_size is None else int(args.step_size)
    n_windows = 0 if int(args.n_windows) < 0 else int(args.n_windows)

    cfg = EvalConfig(
        mode="rolling",
        step_size=step_size,
        n_windows=n_windows,
    )
    validate_eval_config(cfg)
    return cfg


def build_exog_selection(args: argparse.Namespace) -> ExogSelection:
    futr_variant = str(args.futr_variant).lower()
    hist_variant = str(args.hist_variant).lower()
    static_variant = str(args.static_variant).lower()

    use_futr = bool(int(args.use_futr_exog))
    use_hist = bool(int(args.use_hist_exog))
    use_static = bool(int(args.use_static_exog))

    if use_futr and futr_variant == "none":
        raise ValueError("Future exog is enabled but futr_variant='none'")
    if use_hist and hist_variant == "none":
        raise ValueError("Historic exog is enabled but hist_variant='none'")
    if use_static and static_variant == "none":
        raise ValueError("Static exog is enabled but static_variant='none'")

    return ExogSelection(
        futr_variant=None if futr_variant == "none" else futr_variant,
        hist_variant=None if hist_variant == "none" else hist_variant,
        static_variant=None if static_variant == "none" else static_variant,
        use_futr=use_futr,
        use_hist=use_hist,
        use_static=use_static,
    )


def build_model_config(args: argparse.Namespace) -> TftModelConfig:
    base = dict(MODE_CONFIGS[str(args.mode).lower()])

    cfg = TftModelConfig(
        hidden_size=int(args.hidden_size if args.hidden_size is not None else base["hidden_size"]),
        n_head=int(args.n_head if args.n_head is not None else base["n_head"]),
        attn_dropout=float(args.attn_dropout if args.attn_dropout is not None else base["attn_dropout"]),
        dropout=float(args.dropout if args.dropout is not None else base["dropout"]),
        n_rnn_layers=int(args.n_rnn_layers if args.n_rnn_layers is not None else base["n_rnn_layers"]),
        rnn_type=str(args.rnn_type if args.rnn_type is not None else base["rnn_type"]),
        learning_rate=float(args.learning_rate if args.learning_rate is not None else base["learning_rate"]),
        max_steps=int(args.max_steps if args.max_steps is not None else base["max_steps"]),
        batch_size=int(args.batch_size if args.batch_size is not None else base["batch_size"]),
        windows_batch_size=int(
            args.windows_batch_size if args.windows_batch_size is not None else base["windows_batch_size"]
        ),
        scaler_type=str(args.scaler_type),
        loss_name=str(args.loss_name),
        precision=infer_precision(args.precision),
        accelerator=str(args.accelerator),
        devices=int(args.devices),
        num_workers=int(args.num_workers if args.num_workers is not None else base["num_workers"]),
        pin_memory=bool(int(args.pin_memory)) if args.pin_memory is not None else bool(base["pin_memory"]),
        early_stop_patience_steps=int(args.early_stop_patience_steps),
        val_check_steps=None if args.val_check_steps is None else int(args.val_check_steps),
    )
    validate_model_config(cfg)
    return cfg


def build_io_config(project_root: Path, args: argparse.Namespace, run_date: str) -> RunIOConfig:
    return RunIOConfig(
        run_date=run_date,
        data_root=default_data_root(project_root),
        out_root=default_experiments_root(project_root),
        save_predictions=bool(args.save_predictions),
        save_scaler=bool(args.save_scaler),
        save_ckpt=bool(args.save_ckpt),
    )


def count_enabled_exog(selection: ExogSelection) -> int:
    return int(selection.use_futr or selection.use_hist or selection.use_static)


def primary_exog_variant(selection: ExogSelection) -> str:
    if selection.use_futr:
        return str(selection.futr_variant)
    if selection.use_hist:
        return str(selection.hist_variant)
    if selection.use_static:
        return str(selection.static_variant)
    return "none"


def config_payload(
    *,
    args: argparse.Namespace,
    dataset_cfg,
    input_policy_cfg: InputPolicyConfig,
    split_cfg: SplitConfig,
    eval_cfg: EvalConfig,
    exog_selection: ExogSelection,
    model_cfg: TftModelConfig,
    input_size: int,
    run_name: str,
    run_root: Path,
) -> Dict[str, Any]:
    return {
        "dataset": dataset_cfg.name,
        "mode": str(args.mode).lower(),
        "run_name": run_name,
        "run_root": str(run_root),
        "horizon": int(args.H),
        "seed": int(args.seed),
        "input_size": int(input_size),
        "input_policy": asdict(input_policy_cfg),
        "split_config": asdict(split_cfg),
        "eval_config": asdict(eval_cfg),
        "exog_selection": asdict(exog_selection),
        "dataset_config": {
            "name": dataset_cfg.name,
            "freq": dataset_cfg.freq,
            "is_single_series": bool(dataset_cfg.is_single_series),
            "default_horizons": list(dataset_cfg.default_horizons),
            "files": {
                "y_path": str(dataset_cfg.files.y_path),
                "futr_path": None if dataset_cfg.files.futr_path is None else str(dataset_cfg.files.futr_path),
                "hist_path": None if dataset_cfg.files.hist_path is None else str(dataset_cfg.files.hist_path),
                "static_path": None if dataset_cfg.files.static_path is None else str(dataset_cfg.files.static_path),
            },
        },
        "model_config": asdict(model_cfg),
        "argv": vars(args),
    }


def build_fit_kwargs(prepared, split_cfg: SplitConfig) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "df": prepared.fit_df,
        "val_size": int(split_cfg.val_size),
    }
    if prepared.static_df is not None:
        kwargs["static_df"] = prepared.static_df
    return kwargs


def main() -> None:
    args = parse_args()

    project_root = project_root_from_file(__file__, parents=2)
    run_date = str(args.run_date) if args.run_date else today_run_date()

    dataset_cfg = get_dataset_config(project_root, str(args.dataset).upper())
    input_policy_cfg = build_input_policy_config(args, dataset_cfg)
    split_cfg = build_split_config(args, dataset_cfg)
    eval_cfg = build_eval_config(args)
    exog_selection = build_exog_selection(args)
    model_cfg = build_model_config(args)
    io_cfg = build_io_config(project_root, args, run_date)

    input_size = resolve_input_size(
        policy=input_policy_cfg,
        horizon=int(args.H),
        dataset_config=dataset_cfg,
    )
    validate_resolved_input_size(input_size=input_size, horizon=int(args.H))

    run_name = build_run_name(
        dataset=dataset_cfg.name,
        model_name="TFT",
        mode=str(args.mode).lower(),
        horizon=int(args.H),
        seed=int(args.seed),
        input_policy_cfg=input_policy_cfg,
        resolved_input_size=int(input_size),
        exog_selection=exog_selection,
        eval_mode=eval_cfg.mode,
        step_size=int(eval_cfg.step_size),
        n_windows=int(eval_cfg.n_windows),
    )
    run_root = build_run_root(
        experiments_root=io_cfg.out_root,
        dataset=dataset_cfg.name,
        run_date=run_date,
        horizon=int(args.H),
        run_name=run_name,
    )
    ensure_run_root(run_root)
    artifacts = build_run_artifacts(run_root, save_ckpt=io_cfg.save_ckpt)

    prepared = prepare_data(
        cfg=dataset_cfg,
        split_cfg=split_cfg,
        exog_selection=exog_selection,
    )

    if io_cfg.save_scaler and prepared.scaler_stats is not None:
        scaler_meta = scaler_meta_payload(
            cfg=dataset_cfg,
            split_cfg=split_cfg,
            exog_selection=exog_selection,
            full_df=prepared.full_df,
            train_df=prepared.train_df,
            val_df=prepared.val_df,
            test_df=prepared.test_df,
            ds_train_end=prepared.ds_train_end,
            ds_val_end=prepared.ds_val_end,
            n_timestamps=prepared.n_timestamps,
            futr_exog_cols=prepared.futr_exog_cols,
            hist_exog_cols=prepared.hist_exog_cols,
            stat_exog_cols=prepared.stat_exog_cols,
        )
        write_scaler_artifacts(
            run_root,
            stats_df=prepared.scaler_stats,
            meta=scaler_meta,
        )

    model = build_tft_model(
        horizon=int(args.H),
        input_size=int(input_size),
        random_seed=int(args.seed),
        cfg=model_cfg,
        futr_exog_list=prepared.futr_exog_cols,
        hist_exog_list=prepared.hist_exog_cols,
        stat_exog_list=prepared.stat_exog_cols,
        alias=run_name,
    )

    nf = NeuralForecast(models=[model], freq=dataset_cfg.freq)

    if int(args.export_timings) == 1 and torch.cuda.is_available():
        torch.cuda.synchronize()
    t0_train = time.perf_counter()

    nf.fit(**build_fit_kwargs(prepared, split_cfg))

    if int(args.export_timings) == 1 and torch.cuda.is_available():
        torch.cuda.synchronize()
    train_seconds = time.perf_counter() - t0_train

    if io_cfg.save_ckpt:
        model_dir = run_root / "model"
        nf.save(path=str(model_dir), overwrite=True)

    if int(args.export_timings) == 1 and torch.cuda.is_available():
        torch.cuda.synchronize()
    t0_inf = time.perf_counter()

    with torch.no_grad():
        cv_df = evaluate_tft(
            nf,
            prepared=prepared,
            eval_cfg=eval_cfg,
            val_size=int(split_cfg.val_size),
            test_size=int(split_cfg.test_size),
            horizon=int(args.H),
            fixed_cutoff=prepared.ds_val_end,
            verbose=False,
        )

    if int(args.export_timings) == 1 and torch.cuda.is_available():
        torch.cuda.synchronize()
    inf_seconds = time.perf_counter() - t0_inf

    meta: Dict[str, Any] = {
        "dataset": dataset_cfg.name,
        "run_date": run_date,
        "run_name": run_name,
        "model": "TFT",
        "mode": str(args.mode).lower(),
        "eval_mode": eval_cfg.mode,
        "use_exog": int(
            exog_selection.use_futr or exog_selection.use_hist or exog_selection.use_static
        ),
        "use_futr_exog": int(exog_selection.use_futr),
        "use_hist_exog": int(exog_selection.use_hist),
        "use_static_exog": int(exog_selection.use_static),
        "exog_variant": primary_exog_variant(exog_selection),
        "futr_variant": exog_selection.futr_variant or "none",
        "hist_variant": exog_selection.hist_variant or "none",
        "static_variant": exog_selection.static_variant or "none",
        "seed": int(args.seed),
        "H": int(args.H),
        "L_factor": input_policy_cfg.l_factor,
        "input_policy": input_policy_cfg.name,
        "input_size": int(input_size),
        "step_size": int(eval_cfg.step_size),
        "n_windows": int(eval_cfg.n_windows),
        "freq": dataset_cfg.freq,
        "min_ds": pd.Timestamp(prepared.full_df["ds"].min()).isoformat(),
        "max_ds": pd.Timestamp(prepared.full_df["ds"].max()).isoformat(),
        "n_rows": int(len(prepared.full_df)),
        "n_uids": int(prepared.full_df["unique_id"].nunique()),
        "is_single_series": int(bool(dataset_cfg.is_single_series)),
        "ds_train_end": prepared.ds_train_end.isoformat(),
        "ds_val_end": prepared.ds_val_end.isoformat(),
        "val_size": int(split_cfg.val_size),
        "test_size": int(split_cfg.test_size),
        "train_size": int(len(prepared.train_df)),
        "split_mode": "explicit_sizes",
        "split_source": "dataset_config_or_cli",
        "step_size_spec": (
            "fixed_step_0_nw_1"
            if eval_cfg.mode == "fixed"
            else f"rolling_step_{int(eval_cfg.step_size)}"
        ),
        "model_family": "TRANSFORMER_FAMILY",
        "futr_exog_cols": list(prepared.futr_exog_cols),
        "hist_exog_cols": list(prepared.hist_exog_cols),
        "stat_exog_cols": list(prepared.stat_exog_cols),
        "hidden_size": int(model_cfg.hidden_size),
        "n_head": int(model_cfg.n_head),
        "n_rnn_layers": int(model_cfg.n_rnn_layers),
        "rnn_type": str(model_cfg.rnn_type),
        "dropout": float(model_cfg.dropout),
        "attn_dropout": float(model_cfg.attn_dropout),
        "scaler_type": str(model_cfg.scaler_type),
        "loss_name": str(model_cfg.loss_name),
    }

    # Metrics normalization must remain independent of scaler artifact persistence.
    # Even when --save_scaler=0, prepared.scaler_stats stays available in memory.
    scaler_for_metrics = prepared.scaler_stats

    payload = mx.build_metrics_payload(
        meta=meta,
        df_cv=cv_df,
        scaler_stats=scaler_for_metrics,
    )
    mx.write_json(artifacts.metrics_path, payload)

    metrics_runs_csv = io_cfg.out_root / dataset_cfg.name / "metrics_runs.csv"
    mx.upsert_metrics_runs_csv(
        csv_path=metrics_runs_csv,
        payload=payload,
    )

    config_json = config_payload(
        args=args,
        dataset_cfg=dataset_cfg,
        input_policy_cfg=input_policy_cfg,
        split_cfg=split_cfg,
        eval_cfg=eval_cfg,
        exog_selection=exog_selection,
        model_cfg=model_cfg,
        input_size=int(input_size),
        run_name=run_name,
        run_root=run_root,
    )
    with open(artifacts.config_path, "w", encoding="utf-8") as f:
        json.dump(config_json, f, indent=2)

    if io_cfg.save_predictions:
        cv_df.to_parquet(artifacts.predictions_path, index=False)

    if int(args.export_timings) == 1:
        n_channels = 1 + len(prepared.futr_exog_cols) + len(prepared.hist_exog_cols) + len(prepared.stat_exog_cols)

        timing_row = build_timing_row(
            meta=meta,
            train_seconds=float(train_seconds),
            inf_seconds=float(inf_seconds),
            batch_size=int(model_cfg.batch_size),
            input_size=int(input_size),
            n_channels=int(n_channels),
            step_size_spec=meta["step_size_spec"],
        )

        timings_runs_csv = io_cfg.out_root / dataset_cfg.name / "timings_runs.csv"
        timings_summary_csv = io_cfg.out_root / dataset_cfg.name / "timings_summary.csv"
        cost_benefit_csv = io_cfg.out_root / dataset_cfg.name / "cost_benefit.csv"

        upsert_timings_runs_csv(
            csv_path=timings_runs_csv,
            row=timing_row,
            key_cols=[
                "dataset",
                "model",
                "use_exog",
                "exog_variant",
                "H",
                "L_factor",
                "input_size",
                "seed",
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
                "run_name",
            ],
        )
        rebuild_timings_runs_with_summary(csv_path=timings_runs_csv)
        
        build_timings_summary_csv(
            runs_csv_path=timings_runs_csv,
            out_csv_path=timings_summary_csv,
        )

        build_cost_benefit_csv(
            metrics_runs_csv=metrics_runs_csv,
            timings_runs_csv=timings_runs_csv,
            out_csv_path=cost_benefit_csv,
            metric_col="MAE_pooled_norm",
        )

    print(json.dumps(
        {
            "status": "ok",
            "dataset": dataset_cfg.name,
            "run_name": run_name,
            "run_root": str(run_root),
            "H": int(args.H),
            "input_size": int(input_size),
            "eval_mode": eval_cfg.mode,
            "seed": int(args.seed),
        },
        ensure_ascii=False,
    ))


if __name__ == "__main__":
    main()