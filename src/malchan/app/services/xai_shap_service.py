"""Service adapter for exporting all cached SHAP values through FastAPI."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import pandas as pd

from malchan.app.schemas import XaiShapValuesResponse

from .xai_service import XaiNotReadyError, _cached_target, _shared_columns


def _ordered_shap_features(
    child: Any,
    shap_cache: Mapping[str, Any],
) -> list[str]:
    """Return cached SHAP features in the model's original column order."""

    cached_features = [str(feature) for feature in shap_cache]
    model_features = [
        feature
        for feature in _shared_columns(child, "all_cols")
        if feature in shap_cache
    ]
    return model_features + [
        feature for feature in cached_features if feature not in model_features
    ]


def _shap_value_columns(frame: pd.DataFrame) -> list[str]:
    """Return regression or class-specific SHAP columns from one cache frame."""

    return [
        str(column)
        for column in frame.columns
        if str(column) == "shap" or str(column).startswith("shap_")
    ]


def get_xai_shap_values(
    service: Any,
    model_id: str,
    target: str,
) -> XaiShapValuesResponse:
    """Return all raw feature values and aligned SHAP matrices for one target.

    The existing XAI cache stores one SHAP scatter frame per raw feature. This
    function reconstructs a row-by-feature matrix without recalculating SHAP.
    Regression normally returns one ``shap`` matrix. Classification may return
    one matrix per class, such as ``shap_OK`` and ``shap_NG``.
    """

    child, cache = _cached_target(service, model_id, target)
    shap_cache = cache.get("shap_pd")
    if not isinstance(shap_cache, Mapping) or not shap_cache:
        raise XaiNotReadyError(
            f"Cached SHAP values are unavailable for target {target!r}."
        )

    features = _ordered_shap_features(child, shap_cache)
    if not features:
        raise XaiNotReadyError(
            f"Cached SHAP values contain no features for target {target!r}."
        )

    frames: dict[str, pd.DataFrame] = {}
    row_count: int | None = None
    output_names: list[str] | None = None

    for feature in features:
        cached = shap_cache[feature]
        frame = cached if isinstance(cached, pd.DataFrame) else pd.DataFrame(cached)
        frame = frame.reset_index(drop=True)
        if feature not in frame.columns:
            raise ValueError(
                f"Cached SHAP records for feature {feature!r} do not contain "
                "the raw feature column."
            )

        current_outputs = _shap_value_columns(frame)
        if not current_outputs:
            raise XaiNotReadyError(
                f"Cached SHAP records for feature {feature!r} contain no SHAP values."
            )

        if row_count is None:
            row_count = len(frame)
            output_names = current_outputs
        elif len(frame) != row_count:
            raise ValueError(
                "Cached SHAP feature frames must contain the same number of rows."
            )

        if current_outputs != output_names:
            raise ValueError(
                "Cached SHAP feature frames must expose the same SHAP output columns."
            )
        frames[feature] = frame

    assert output_names is not None

    raw_frame = pd.DataFrame(
        {feature: frames[feature][feature] for feature in features}
    )
    records = json.loads(
        raw_frame.to_json(
            orient="records",
            date_format="iso",
            date_unit="ms",
        )
    )

    shap_values: dict[str, list[list[float | None]]] = {}
    for output_name in output_names:
        values_frame = pd.DataFrame(
            {
                feature: pd.to_numeric(
                    frames[feature][output_name],
                    errors="coerce",
                )
                for feature in features
            }
        )
        shap_values[output_name] = json.loads(
            values_frame.to_json(orient="values")
        )

    cat_cols = [
        feature
        for feature in _shared_columns(child, "cat_cols")
        if feature in features
    ]
    return XaiShapValuesResponse(
        model_id=model_id,
        target=target,
        features=features,
        cat_cols=cat_cols,
        output_names=output_names,
        records=records,
        shap_values=shap_values,
    )


def install_xai_shap_service(service_cls: type[Any]) -> None:
    """Attach the full-SHAP export operation to a model service class."""

    service_cls.get_xai_shap_values = get_xai_shap_values


__all__ = ["install_xai_shap_service"]
