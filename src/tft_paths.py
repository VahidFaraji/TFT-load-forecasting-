from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from tft_types import DatasetName, EvalMode, ExogSelection, InputPolicyConfig, RunArtifacts


def project_root_from_file(file_path: str | Path, parents: int = 1) -> Path:
    path = Path(file_path).resolve()
    for _ in range(parents):
        path = path.parent
    return path


def default_data_root(project_root: Path) -> Path:
    return project_root / "datasets"


def default_experiments_root(project_root: Path) -> Path:
    return project_root / "experiments"


def today_run_date() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def exog_tag(selection: ExogSelection) -> str:
    parts: list[str] = []

    if selection.use_futr:
        parts.append(f"FUTR-{selection.futr_variant or 'ON'}")
    else:
        parts.append("FUTR-OFF")

    if selection.use_hist:
        parts.append(f"HIST-{selection.hist_variant or 'ON'}")
    else:
        parts.append("HIST-OFF")

    if selection.use_static:
        parts.append(f"STAT-{selection.static_variant or 'ON'}")
    else:
        parts.append("STAT-OFF")

    return "_".join(parts)


def input_policy_tag(cfg: InputPolicyConfig, resolved_input_size: int) -> str:
    if cfg.name == "manual":
        return f"INP-MANUAL-{resolved_input_size}"
    if cfg.name == "l_factor":
        return f"INP-LF{cfg.l_factor}-{resolved_input_size}"
    if cfg.name == "seasonal_l_factor":
        return f"INP-SLF{cfg.l_factor}-SM{cfg.seasonal_min}-{resolved_input_size}"
    raise ValueError(f"Unsupported input policy: {cfg.name}")


def eval_tag(mode: EvalMode, step_size: int, n_windows: int) -> str:
    return f"{mode.upper()}-STEP{step_size}-NW{n_windows}"


def build_run_name(
    dataset: DatasetName,
    model_name: str,
    mode: str,
    horizon: int,
    seed: int,
    input_policy_cfg: InputPolicyConfig,
    resolved_input_size: int,
    exog_selection: ExogSelection,
    eval_mode: EvalMode,
    step_size: int,
    n_windows: int,
    prefix: Optional[str] = None,
) -> str:
    parts = [
        model_name.upper(),
        dataset.upper(),
        mode.upper(),
        f"H{int(horizon)}",
        input_policy_tag(input_policy_cfg, resolved_input_size),
        exog_tag(exog_selection),
        eval_tag(eval_mode, step_size, n_windows),
        f"SEED{int(seed)}",
    ]
    if prefix:
        parts.insert(0, prefix.upper())
    return "__".join(parts)


def build_run_root(
    experiments_root: Path,
    dataset: DatasetName,
    run_date: str,
    horizon: int,
    run_name: str,
) -> Path:
    return experiments_root / dataset.upper() / run_date / f"H{int(horizon)}" / run_name


def build_run_artifacts(run_root: Path, save_ckpt: bool = False) -> RunArtifacts:
    ckpt_path = run_root / "model.ckpt" if save_ckpt else None
    return RunArtifacts(
        run_root=run_root,
        predictions_path=run_root / "predictions_cv.parquet",
        scaler_stats_path=run_root / "scaler_stats.parquet",
        scaler_meta_path=run_root / "scaler_meta.json",
        metrics_path=run_root / "metrics.json",
        config_path=run_root / "config.json",
        ckpt_path=ckpt_path,
    )


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_run_root(run_root: Path) -> Path:
    return ensure_dir(run_root)


def dataset_root(data_root: Path, dataset: DatasetName) -> Path:
    return data_root / dataset.upper()


def serialize_input_policy(cfg: InputPolicyConfig) -> dict:
    return asdict(cfg)