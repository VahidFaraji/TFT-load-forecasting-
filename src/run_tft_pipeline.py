# python src/run_tft_pipeline.py --dataset ECL --mode full --horizons 24 48 96 192 336 720 --eval_mode rolling --seed_base 1 --n_seeds 8 --input_policy seasonal_l_factor --L_factor 3 --seasonal_min 168 --use_futr_exog 1 --futr_variant min6 --use_hist_exog 0 --use_static_exog 0 --save_predictions 0 --save_scaler 1 --save_ckpt 1 --append_history 1 --keep_all_seeds 1 --val_check_steps 100 --early_stop_patience_steps 200 --export_timings 1

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any
import time

import metrics_export as mx

from tft_dataset_configs import get_dataset_config
from tft_input_policy import resolve_input_size
from tft_paths import (
    build_run_name,
    build_run_root,
    default_experiments_root,
    project_root_from_file,
    today_run_date,
)
from tft_types import ExogSelection, InputPolicyConfig


MODE_DEFAULTS: dict[str, dict[str, Any]] = {
    "debug": {
        "seeds": 1,
        "save_predictions": True,
        "save_scaler": True,
        "save_ckpt": False,
    },
    "lite": {
        "seeds": 1,
        "save_predictions": True,
        "save_scaler": True,
        "save_ckpt": False,
    },
    "full": {
        "seeds": 3,
        "save_predictions": True,
        "save_scaler": True,
        "save_ckpt": False,
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--dataset", type=str, choices=["ECL", "PJM", "IRAN"], required=True)
    p.add_argument("--mode", type=str, choices=["debug", "lite", "full"], default="debug")

    p.add_argument("--run_date", type=str, default=None)
    p.add_argument("--seed_base", type=int, default=1)
    p.add_argument("--n_seeds", type=int, default=None)

    p.add_argument("--horizons", type=str, default=None)
    p.add_argument("--eval_mode", type=str, choices=["fixed", "rolling"], default="rolling")
    p.add_argument("--step_size", type=int, default=None)
    p.add_argument("--n_windows", type=int, default=-1)

    p.add_argument("--val_size", type=int, default=None)
    p.add_argument("--test_size", type=int, default=None)

    p.add_argument("--input_policy", type=str, choices=["manual", "l_factor", "seasonal_l_factor"], default=None)
    p.add_argument("--L_factor", type=int, default=None)
    p.add_argument("--seasonal_min", type=int, default=None)
    p.add_argument("--input_size", type=int, default=None)

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

    p.add_argument("--save_predictions", type=int, choices=[0, 1], default=None)
    p.add_argument("--save_scaler", type=int, choices=[0, 1], default=None)
    p.add_argument("--save_ckpt", type=int, choices=[0, 1], default=None)

    p.add_argument("--keep_all_seeds", type=int, choices=[0, 1], default=1)
    p.add_argument("--append_history", type=int, choices=[0, 1], default=1)
    p.add_argument("--metrics_csv_name", type=str, default="metrics_runs.csv")

    return p.parse_args()


def parse_horizons(raw: str | None, dataset_cfg) -> list[int]:
    if raw is None or not str(raw).strip():
        return list(dataset_cfg.default_horizons)

    values: list[int] = []
    for part in str(raw).split(","):
        item = part.strip()
        if item:
            values.append(int(item))

    if not values:
        raise ValueError("No valid horizons were parsed")
    return values


def resolve_n_seeds(args: argparse.Namespace) -> int:
    if args.n_seeds is not None:
        return int(args.n_seeds)
    return int(MODE_DEFAULTS[str(args.mode)]["seeds"])


def resolve_bool_flag(user_value: int | None, default_value: bool) -> bool:
    if user_value is None:
        return bool(default_value)
    return bool(int(user_value))


def build_input_policy(args: argparse.Namespace, dataset_cfg) -> InputPolicyConfig:
    base = dataset_cfg.default_input_policy
    name = str(args.input_policy).lower() if args.input_policy else base.name

    if name == "manual":
        if args.input_size is None:
            raise ValueError("--input_size is required when --input_policy=manual")
        return InputPolicyConfig(
            name="manual",
            manual_input_size=int(args.input_size),
        )

    if name == "l_factor":
        l_factor = args.L_factor if args.L_factor is not None else base.l_factor
        if l_factor is None:
            raise ValueError("L_factor is required for input_policy=l_factor")
        return InputPolicyConfig(
            name="l_factor",
            l_factor=int(l_factor),
        )

    if name == "seasonal_l_factor":
        l_factor = args.L_factor if args.L_factor is not None else base.l_factor
        seasonal_min = args.seasonal_min if args.seasonal_min is not None else base.seasonal_min
        if l_factor is None:
            raise ValueError("L_factor is required for input_policy=seasonal_l_factor")
        return InputPolicyConfig(
            name="seasonal_l_factor",
            l_factor=int(l_factor),
            seasonal_min=None if seasonal_min is None else int(seasonal_min),
        )

    raise ValueError(f"Unsupported input_policy: {name}")


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


def default_step_size(dataset_cfg, horizon: int, eval_mode: str, user_step_size: int | None) -> int:
    if eval_mode == "fixed":
        return 0
    if user_step_size is not None:
        return int(user_step_size)
    step_map = dataset_cfg.extra.get("default_step_map", {})
    return int(step_map.get(int(horizon), 1))


def build_train_cmd(
    *,
    python_exe: str,
    train_script: Path,
    args: argparse.Namespace,
    dataset_cfg,
    run_date: str,
    horizon: int,
    seed: int,
    input_policy_cfg: InputPolicyConfig,
    input_size: int,
    exog_selection: ExogSelection,
    save_predictions: bool,
    save_scaler: bool,
    save_ckpt: bool,
) -> list[str]:
    step_size = default_step_size(dataset_cfg, horizon, str(args.eval_mode), args.step_size)

    cmd = [
        python_exe,
        str(train_script),
        "--dataset", str(dataset_cfg.name),
        "--mode", str(args.mode),
        "--run_date", str(run_date),
        "--seed", str(seed),
        "--H", str(int(horizon)),
        "--input_policy", str(input_policy_cfg.name),
        "--eval_mode", str(args.eval_mode),
        "--step_size", str(step_size),
        "--n_windows", str(int(args.n_windows)),
        "--val_size", str(int(args.val_size if args.val_size is not None else dataset_cfg.default_split.val_size)),
        "--test_size", str(int(args.test_size if args.test_size is not None else dataset_cfg.default_split.test_size)),
        "--use_futr_exog", str(int(exog_selection.use_futr)),
        "--use_hist_exog", str(int(exog_selection.use_hist)),
        "--use_static_exog", str(int(exog_selection.use_static)),
        "--futr_variant", str(exog_selection.futr_variant or "none"),
        "--hist_variant", str(exog_selection.hist_variant or "none"),
        "--static_variant", str(exog_selection.static_variant or "none"),
        "--scaler_type", str(args.scaler_type),
        "--loss_name", str(args.loss_name),
        "--early_stop_patience_steps", str(int(args.early_stop_patience_steps)),
        "--accelerator", str(args.accelerator),
        "--devices", str(int(args.devices)),
    ]

    if input_policy_cfg.name == "manual":
        cmd += ["--input_size", str(int(input_size))]
    else:
        cmd += ["--L_factor", str(int(input_policy_cfg.l_factor))]
        if input_policy_cfg.name == "seasonal_l_factor" and input_policy_cfg.seasonal_min is not None:
            cmd += ["--seasonal_min", str(int(input_policy_cfg.seasonal_min))]

    optional_pairs: list[tuple[str, Any]] = [
        ("--hidden_size", args.hidden_size),
        ("--n_head", args.n_head),
        ("--attn_dropout", args.attn_dropout),
        ("--dropout", args.dropout),
        ("--n_rnn_layers", args.n_rnn_layers),
        ("--rnn_type", args.rnn_type),
        ("--learning_rate", args.learning_rate),
        ("--max_steps", args.max_steps),
        ("--batch_size", args.batch_size),
        ("--windows_batch_size", args.windows_batch_size),
        ("--val_check_steps", args.val_check_steps),
        ("--precision", args.precision),
        ("--num_workers", args.num_workers),
        ("--pin_memory", args.pin_memory),
    ]
    for key, value in optional_pairs:
        if value is not None:
            cmd += [key, str(value)]

    cmd += ["--save_predictions", str(int(save_predictions))]
    cmd += ["--save_scaler", str(int(save_scaler))]
    cmd += ["--save_ckpt", str(int(save_ckpt))]

    return cmd


def expected_run_name(
    *,
    dataset_cfg,
    mode: str,
    horizon: int,
    seed: int,
    input_policy_cfg: InputPolicyConfig,
    input_size: int,
    exog_selection: ExogSelection,
    eval_mode: str,
    step_size: int,
    n_windows: int,
) -> str:
    return build_run_name(
        dataset=dataset_cfg.name,
        model_name="TFT",
        mode=mode,
        horizon=int(horizon),
        seed=int(seed),
        input_policy_cfg=input_policy_cfg,
        resolved_input_size=int(input_size),
        exog_selection=exog_selection,
        eval_mode=eval_mode,
        step_size=int(step_size),
        n_windows=int(1 if eval_mode == "fixed" else n_windows),
    )


def load_metrics_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing metrics.json: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def cleanup_non_best_run(run_root: Path) -> None:
    preds = run_root / "predictions_cv.parquet"
    if preds.exists():
        preds.unlink()

    model_dir = run_root / "model"
    if model_dir.exists() and model_dir.is_dir():
        shutil.rmtree(model_dir)

    ckpt_file = run_root / "model.ckpt"
    if ckpt_file.exists():
        ckpt_file.unlink()


def main() -> None:
    args = parse_args()

    project_root = project_root_from_file(__file__, parents=2)
    experiments_root = default_experiments_root(project_root)
    train_script = project_root / "src" / "train_tft.py"

    if not train_script.exists():
        raise FileNotFoundError(f"Missing train script: {train_script}")

    dataset_cfg = get_dataset_config(project_root, str(args.dataset).upper())
    run_date = str(args.run_date) if args.run_date else today_run_date()
    horizons = parse_horizons(args.horizons, dataset_cfg)
    n_seeds = resolve_n_seeds(args)

    save_predictions = resolve_bool_flag(args.save_predictions, MODE_DEFAULTS[str(args.mode)]["save_predictions"])
    save_scaler = resolve_bool_flag(args.save_scaler, MODE_DEFAULTS[str(args.mode)]["save_scaler"])
    save_ckpt = resolve_bool_flag(args.save_ckpt, MODE_DEFAULTS[str(args.mode)]["save_ckpt"])
    append_history = bool(int(args.append_history))

    input_policy_cfg_base = build_input_policy(args, dataset_cfg)
    exog_selection = build_exog_selection(args)

    seed_rows_all: list[dict[str, Any]] = []
    best_runs: dict[int, dict[str, Any]] = {}
    total_runs = len(horizons) * int(n_seeds)
    done_runs = 0
    durations_sec: list[float] = []

    def fmt_hms(seconds: float) -> str:
        total = max(0, int(seconds))
        h = total // 3600
        m = (total % 3600) // 60
        s = total % 60
        return f"{h:02d}:{m:02d}:{s:02d}"

    for horizon in horizons:
        run_rows_this_h: list[dict[str, Any]] = []
        best_score_this_h: float | None = None
        best_run_root_this_h: Path | None = None

        effective_input_policy = input_policy_cfg_base
        if effective_input_policy.name == "manual":
            effective_input_policy = replace(
                effective_input_policy,
                manual_input_size=int(args.input_size),
            )

        resolved_input_size = resolve_input_size(
            policy=effective_input_policy,
            horizon=int(horizon),
            dataset_config=dataset_cfg,
        )

        step_size = default_step_size(dataset_cfg, int(horizon), str(args.eval_mode), args.step_size)
        n_windows_for_name = 1 if str(args.eval_mode) == "fixed" else (0 if int(args.n_windows) < 0 else int(args.n_windows))

        for seed in range(int(args.seed_base), int(args.seed_base) + int(n_seeds)):
            run_name = expected_run_name(
                dataset_cfg=dataset_cfg,
                mode=str(args.mode),
                horizon=int(horizon),
                seed=int(seed),
                input_policy_cfg=effective_input_policy,
                input_size=int(resolved_input_size),
                exog_selection=exog_selection,
                eval_mode=str(args.eval_mode),
                step_size=int(step_size),
                n_windows=int(n_windows_for_name),
            )
            run_root = build_run_root(
                experiments_root=experiments_root,
                dataset=dataset_cfg.name,
                run_date=run_date,
                horizon=int(horizon),
                run_name=run_name,
            )

            cmd = build_train_cmd(
                python_exe=sys.executable,
                train_script=train_script,
                args=args,
                dataset_cfg=dataset_cfg,
                run_date=run_date,
                horizon=int(horizon),
                seed=int(seed),
                input_policy_cfg=effective_input_policy,
                input_size=int(resolved_input_size),
                exog_selection=exog_selection,
                save_predictions=save_predictions,
                save_scaler=save_scaler,
                save_ckpt=save_ckpt,
            )

            done_runs += 1
            print(
                f"[{done_runs}/{total_runs}] START | "
                f"seed={int(seed)} | H={int(horizon)} | input={int(resolved_input_size)} | "
                f"eval={str(args.eval_mode)} | step={int(step_size)} | nw={int(n_windows_for_name)} | "
                f"run_name={run_name}",
                flush=True,
            )
            print("CMD:", " ".join(cmd), flush=True)

            t0 = time.perf_counter()
            subprocess.run(cmd, check=True)
            dt = time.perf_counter() - t0

            durations_sec.append(dt)
            avg_sec = sum(durations_sec) / len(durations_sec)
            remaining_runs = total_runs - done_runs
            remaining_sec = avg_sec * remaining_runs

            print(
                f"[{done_runs}/{total_runs}] DONE  | "
                f"{fmt_hms(dt)} | avg/run={fmt_hms(avg_sec)} | remaining≈{fmt_hms(remaining_sec)}",
                flush=True,
            )

            payload_raw = load_metrics_json(run_root / "metrics.json")
            row = mx.flatten_metrics_row(payload_raw)
            seed_rows_all.append(row)
            run_rows_this_h.append(row)

            score = mx.select_primary_score(row)
            if best_score_this_h is None or score < best_score_this_h:
                best_score_this_h = score
                best_run_root_this_h = run_root

        if best_run_root_this_h is None:
            raise RuntimeError(f"No successful runs completed for H={horizon}")

        best_runs[int(horizon)] = {
            "best_score": float(best_score_this_h),
            "best_run_root": str(best_run_root_this_h),
        }

        if not bool(int(args.keep_all_seeds)):
            for row in run_rows_this_h:
                candidate_root = build_run_root(
                    experiments_root=experiments_root,
                    dataset=dataset_cfg.name,
                    run_date=run_date,
                    horizon=int(row["H"]),
                    run_name=str(row["run_name"]),
                )
                if candidate_root != best_run_root_this_h:
                    cleanup_non_best_run(candidate_root)

    dataset_metrics_dir = experiments_root / dataset_cfg.name
    dataset_metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv_path = dataset_metrics_dir / str(args.metrics_csv_name)

    metrics_df = mx.write_metrics_runs_csv(
        metrics_csv_path,
        seed_rows_all,
        append_history=append_history,
    )

    summary = {
        "status": "ok",
        "dataset": dataset_cfg.name,
        "run_date": run_date,
        "mode": str(args.mode),
        "eval_mode": str(args.eval_mode),
        "append_history": append_history,
        "n_seed_rows_this_run": int(len(seed_rows_all)),
        "rows_in_metrics_csv": int(len(metrics_df)),
        "metrics_csv": str(metrics_csv_path),
        "best_runs": best_runs,
    }
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()