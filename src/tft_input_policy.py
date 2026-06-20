from __future__ import annotations

from tft_types import DatasetConfig, InputPolicyConfig, ensure_positive_int, validate_input_policy_config


def seasonal_min_for_dataset(cfg: DatasetConfig) -> int:
    value = cfg.extra.get("seasonal_min")
    if value is None:
        freq = str(cfg.freq).lower()
        if freq == "h":
            return 168
        if freq == "d":
            return 7
        raise ValueError(f"{cfg.name}: seasonal_min is not defined and no default rule exists for freq={cfg.freq}")
    return ensure_positive_int("seasonal_min", int(value))


def resolve_input_size(
    policy: InputPolicyConfig,
    horizon: int,
    dataset_config: DatasetConfig,
) -> int:
    validate_input_policy_config(policy)
    h = ensure_positive_int("horizon", int(horizon))

    if policy.name == "manual":
        return ensure_positive_int("manual_input_size", int(policy.manual_input_size))

    if policy.name == "l_factor":
        return ensure_positive_int("input_size", int(policy.l_factor) * h)

    if policy.name == "seasonal_l_factor":
        seasonal_min = (
            ensure_positive_int("seasonal_min", int(policy.seasonal_min))
            if policy.seasonal_min is not None
            else seasonal_min_for_dataset(dataset_config)
        )
        candidate = int(policy.l_factor) * h
        return ensure_positive_int("input_size", max(seasonal_min, candidate))

    raise ValueError(f"Unsupported input policy: {policy.name}")


def input_policy_summary(
    policy: InputPolicyConfig,
    horizon: int,
    dataset_config: DatasetConfig,
) -> dict:
    resolved = resolve_input_size(
        policy=policy,
        horizon=horizon,
        dataset_config=dataset_config,
    )
    summary = {
        "policy_name": policy.name,
        "horizon": int(horizon),
        "resolved_input_size": int(resolved),
    }

    if policy.name == "manual":
        summary["manual_input_size"] = int(policy.manual_input_size)
        return summary

    summary["l_factor"] = int(policy.l_factor)

    if policy.name == "seasonal_l_factor":
        summary["seasonal_min"] = (
            int(policy.seasonal_min)
            if policy.seasonal_min is not None
            else int(seasonal_min_for_dataset(dataset_config))
        )

    return summary


def validate_resolved_input_size(
    input_size: int,
    horizon: int,
) -> int:
    size = ensure_positive_int("input_size", int(input_size))
    ensure_positive_int("horizon", int(horizon))
    if size < 2:
        raise ValueError(f"input_size must be >= 2, got {size}")
    return size