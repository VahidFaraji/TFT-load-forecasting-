# tft_model_factory.py

from typing import Any

from neuralforecast.losses.pytorch import MAE, MSE, QuantileLoss, SMAPE
from neuralforecast.models import TFT

from tft_types import TftModelConfig, validate_model_config


def resolve_loss(loss_name: str):
    key = str(loss_name).strip().lower()
    if key == "mae":
        return MAE()
    if key == "mse":
        return MSE()
    if key == "smape":
        return SMAPE()
    if key.startswith("quantile:"):
        q = float(key.split(":", 1)[1])
        return QuantileLoss(q=q)
    raise ValueError(f"Unsupported loss_name: {loss_name}")


def build_dataloader_kwargs(cfg: TftModelConfig) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if cfg.num_workers >= 0:
        kwargs["num_workers"] = int(cfg.num_workers)
    if cfg.pin_memory:
        kwargs["pin_memory"] = bool(cfg.pin_memory)
    return kwargs


def build_trainer_kwargs(cfg: TftModelConfig) -> dict[str, Any]:
    return {
        "accelerator": cfg.accelerator,
        "devices": cfg.devices,
        "precision": cfg.precision,
        "num_sanity_val_steps": 0,
        "limit_val_batches": 0,
    }


def build_tft_model(
    *,
    horizon: int,
    input_size: int,
    random_seed: int,
    cfg: TftModelConfig,
    futr_exog_list: tuple[str, ...] = (),
    hist_exog_list: tuple[str, ...] = (),
    stat_exog_list: tuple[str, ...] = (),
    alias: str | None = None,
) -> TFT:
    validate_model_config(cfg)

    train_loss = resolve_loss(cfg.loss_name)
    valid_loss = resolve_loss(cfg.loss_name)

    dataloader_kwargs = build_dataloader_kwargs(cfg)
    trainer_kwargs = build_trainer_kwargs(cfg)

    val_check_steps = (
        int(cfg.val_check_steps)
        if cfg.val_check_steps is not None
        else max(1, min(100, int(cfg.max_steps)))
    )

    model = TFT(
        h=int(horizon),
        input_size=int(input_size),
        stat_exog_list=list(stat_exog_list) or None,
        hist_exog_list=list(hist_exog_list) or None,
        futr_exog_list=list(futr_exog_list) or None,
        hidden_size=int(cfg.hidden_size),
        n_head=int(cfg.n_head),
        attn_dropout=float(cfg.attn_dropout),
        n_rnn_layers=int(cfg.n_rnn_layers),
        rnn_type=str(cfg.rnn_type),
        dropout=float(cfg.dropout),
        loss=train_loss,
        valid_loss=valid_loss,
        max_steps=int(cfg.max_steps),
        learning_rate=float(cfg.learning_rate),
        early_stop_patience_steps=int(cfg.early_stop_patience_steps),
        val_check_steps=int(val_check_steps),
        batch_size=int(cfg.batch_size),
        valid_batch_size=int(cfg.batch_size),
        windows_batch_size=int(cfg.windows_batch_size),
        inference_windows_batch_size=int(cfg.windows_batch_size),
        step_size=1,
        scaler_type=str(cfg.scaler_type),
        random_seed=int(random_seed),
        alias=alias,
        dataloader_kwargs=dataloader_kwargs or None,
        **trainer_kwargs,
    )
    return model