"""Normalize inverse-analysis search settings before Optuna suggestions."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

from . import utils as _utils


def _normalize_bound(value: Any, *, param_name: str, bound_name: str) -> float:
    """Convert one numeric search bound to a finite float."""

    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Inverse-analysis {bound_name} for {param_name!r} must be numeric."
        ) from exc
    if not math.isfinite(normalized):
        raise ValueError(
            f"Inverse-analysis {bound_name} for {param_name!r} must be finite. "
            "Clean the training column or specify an explicit bound."
        )
    return normalized


def normalize_optional_step(
    step: Any,
    *,
    dtype: str,
    param_name: str,
) -> float | int | None:
    """Normalize pandas missing values and validate an Optuna step value."""

    if step is None:
        return 1 if dtype == "int" else None
    try:
        if bool(pd.isna(step)):
            return 1 if dtype == "int" else None
    except (TypeError, ValueError):
        pass

    try:
        normalized = float(step)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Inverse-analysis step for {param_name!r} must be numeric or None."
        ) from exc
    if not math.isfinite(normalized) or normalized <= 0:
        raise ValueError(
            f"Inverse-analysis step for {param_name!r} must be finite and positive."
        )
    if dtype == "int":
        if not normalized.is_integer():
            raise ValueError(
                f"Integer inverse-analysis step for {param_name!r} must be integral."
            )
        return int(normalized)
    return normalized


def safe_suggest_parameter(
    trial: Any,
    param_name: str,
    dtype: str,
    param_min: Any,
    param_max: Any,
    step: Any = None,
    categories: list[Any] | None = None,
) -> Any:
    """Suggest one parameter after normalizing pandas-derived settings."""

    if dtype in {"float", "int"}:
        lower = _normalize_bound(
            param_min,
            param_name=param_name,
            bound_name="lower bound",
        )
        upper = _normalize_bound(
            param_max,
            param_name=param_name,
            bound_name="upper bound",
        )
        if upper < lower:
            raise ValueError(
                f"Inverse-analysis upper bound for {param_name!r} must be greater "
                "than or equal to its lower bound."
            )
        normalized_step = normalize_optional_step(
            step,
            dtype=dtype,
            param_name=param_name,
        )
        if dtype == "float":
            return trial.suggest_float(
                param_name,
                lower,
                upper,
                step=normalized_step,
            )

        if not lower.is_integer() or not upper.is_integer():
            raise ValueError(
                f"Integer inverse-analysis bounds for {param_name!r} must be integral."
            )
        return trial.suggest_int(
            param_name,
            int(lower),
            int(upper),
            step=normalized_step,
        )

    if dtype == "object" and categories:
        return trial.suggest_categorical(param_name, categories)
    raise ValueError(f"Unsupported data type: {dtype}")


def install_step_normalization() -> None:
    """Install the safe suggestion function into the legacy utility module."""

    if _utils.suggest_parameter is safe_suggest_parameter:
        return
    _utils.suggest_parameter = safe_suggest_parameter


__all__ = [
    "install_step_normalization",
    "normalize_optional_step",
    "safe_suggest_parameter",
]
