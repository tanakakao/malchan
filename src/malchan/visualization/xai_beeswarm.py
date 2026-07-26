"""Plotly beeswarm adapter for FastAPI full-SHAP responses."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .machine_learning_plots import show_shap_beeswarm


def _as_mapping(payload: Any) -> Mapping[str, Any]:
    """Convert an HTTP response, Pydantic model, or mapping into a mapping."""

    if isinstance(payload, Mapping):
        return payload

    model_dump = getattr(payload, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump(mode="json")
        if isinstance(dumped, Mapping):
            return dumped

    json_method = getattr(payload, "json", None)
    if callable(json_method):
        dumped = json_method()
        if isinstance(dumped, Mapping):
            return dumped

    raise TypeError(
        "response must be a mapping, an HTTP response with json(), "
        "or a Pydantic model with model_dump()."
    )


def _resolve_output_name(
    output_names: list[str],
    target_item: Any | None,
) -> str:
    """Resolve one regression or class-specific SHAP output matrix."""

    if not output_names:
        raise ValueError("XAI SHAP values response contains no output names.")
    if target_item is None:
        return output_names[0] if len(output_names) == 1 else output_names[-1]

    requested = str(target_item)
    for candidate in (requested, f"shap_{requested}"):
        if candidate in output_names:
            return candidate
    raise ValueError(
        f"SHAP output for target_item={target_item!r} is unavailable. "
        f"Available: {output_names}"
    )


def show_xai_shap_beeswarm(
    response: Any,
    n_shap_top: int = 10,
    target_item: Any | None = None,
) -> go.Figure:
    """Visualize all-feature FastAPI SHAP values as a Plotly beeswarm.

    Args:
        response: ``/xai/{target}/shap-values`` response, its JSON mapping, or
            the corresponding Pydantic response model.
        n_shap_top: Number of features ranked by total absolute SHAP value.
        target_item: Classification class label or explicit SHAP output name.
            For a multi-class response, omission selects the last output matrix.

    Returns:
        Plotly beeswarm figure.

    Raises:
        ValueError: If required response fields, the selected output, or matrix
            dimensions are invalid.
    """

    if n_shap_top < 1:
        raise ValueError("n_shap_top must be at least 1.")

    payload = _as_mapping(response)
    features = payload.get("features")
    records = payload.get("records")
    output_names = payload.get("output_names")
    shap_values_by_output = payload.get("shap_values")
    cat_cols = payload.get("cat_cols") or []

    if not isinstance(features, list) or not features:
        raise ValueError("XAI SHAP values response must contain a non-empty 'features' list.")
    features = [str(feature) for feature in features]
    if not isinstance(records, list):
        raise ValueError("XAI SHAP values response must contain a 'records' list.")
    if not isinstance(output_names, list):
        raise ValueError("XAI SHAP values response must contain an 'output_names' list.")
    output_names = [str(name) for name in output_names]
    if not isinstance(shap_values_by_output, Mapping):
        raise ValueError("XAI SHAP values response must contain a 'shap_values' mapping.")
    if not isinstance(cat_cols, list):
        raise ValueError("XAI SHAP values response 'cat_cols' must be a list.")

    output_name = _resolve_output_name(output_names, target_item)
    matrix = shap_values_by_output.get(output_name)
    if not isinstance(matrix, list):
        raise ValueError(f"SHAP matrix {output_name!r} is unavailable.")

    feature_frame = pd.DataFrame(records)
    missing_features = [feature for feature in features if feature not in feature_frame]
    if missing_features:
        raise ValueError(f"SHAP records are missing features: {missing_features}")
    feature_frame = feature_frame[features]

    shap_array = np.asarray(matrix, dtype=float)
    expected_shape = (len(feature_frame), len(features))
    if shap_array.shape != expected_shape:
        raise ValueError(
            f"SHAP matrix {output_name!r} has shape {shap_array.shape}; "
            f"expected {expected_shape}."
        )

    fig = show_shap_beeswarm(
        X=feature_frame,
        shap_values=shap_array,
        n_shap_top=min(n_shap_top, len(features)),
        cat_cols=[str(column) for column in cat_cols if str(column) in features],
    )
    target = payload.get("target", "target")
    fig.update_layout(title=f"SHAP beeswarm: {target} / {output_name}")
    return fig


__all__ = ["show_xai_shap_beeswarm"]
