"""Plotly figures used by the FastAPI web application.

The web layer should serialize figures produced here instead of rebuilding chart
geometry in JavaScript.  Existing notebook-oriented visualization functions are
reused where their model contract already fits, while web-specific adapters
normalize single-output and multi-output pipelines.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .machine_learning_plots import yy_plot_ml


class _SingleOutputVisualizationAdapter:
    """Expose a single-output pipeline through the multi-output plot contract."""

    def __init__(self, model: Any, target: str) -> None:
        self._model = model
        self.models = {target: model}

    def predict(self, *args: Any, **kwargs: Any) -> Any:
        """Delegate prediction to the wrapped single-output model."""

        return self._model.predict(*args, **kwargs)


def _resolve_child_model(model: Any, target: str) -> Any:
    """Return the target-specific child model from either pipeline style."""

    model_map = getattr(model, "models", None)
    if isinstance(model_map, Mapping):
        if target not in model_map:
            raise ValueError(f"Unknown visualization target {target!r}.")
        return model_map[target]

    fitted_target = getattr(model, "target_col", None)
    if fitted_target != target:
        raise ValueError(
            f"Unknown visualization target {target!r}; fitted target is {fitted_target!r}."
        )
    return model


def _model_for_yy_plot(model: Any, target: str) -> Any:
    """Return a model satisfying :func:`yy_plot_ml`'s target-model contract."""

    if isinstance(getattr(model, "models", None), Mapping):
        return model
    return _SingleOutputVisualizationAdapter(model, target)


def _training_features(child_model: Any) -> pd.DataFrame:
    """Return raw training features stored by a fitted child model."""

    values = (
        child_model._get_X()
        if hasattr(child_model, "_get_X")
        else getattr(child_model, "X", None)
    )
    if values is None:
        raise ValueError("Training features are unavailable for visualization.")
    return values if isinstance(values, pd.DataFrame) else pd.DataFrame(values)


def _training_target(child_model: Any) -> np.ndarray | None:
    """Return numeric target values when they can color the observed-data layer."""

    values = (
        child_model._get_y()
        if hasattr(child_model, "_get_y")
        else getattr(child_model, "y", None)
    )
    if values is None:
        return None
    array = np.asarray(values).ravel()
    try:
        numeric = array.astype(float)
    except (TypeError, ValueError):
        return None
    return numeric if np.isfinite(numeric).any() else None


def visualization_outputs(model: Any, target: str) -> list[str]:
    """Return selectable prediction outputs for a target visualization."""

    child_model = _resolve_child_model(model, target)
    if getattr(child_model, "task", None) != "classification":
        return [target]

    target_items = getattr(child_model, "target_items", None)
    if target_items is None:
        return []
    return [str(value) for value in list(target_items)]


def show_model_diagnostics(
    model: Any,
    target: str,
    *,
    cv: bool = False,
    residual: bool = False,
) -> go.Figure:
    """Create the standard malchan Y-Y or classification diagnostic figure."""

    return yy_plot_ml(
        model=_model_for_yy_plot(model, target),
        target=target,
        cv=cv,
        residual=residual,
    )


def _ordered_unique(values: np.ndarray) -> np.ndarray:
    """Return unique grid values while preserving their original order."""

    return np.asarray(pd.unique(pd.Series(values.ravel())))


def show_model_pd_2d(
    model: Any,
    target: str,
    feature_names: Sequence[str],
    *,
    output_index: int = -1,
) -> go.Figure:
    """Create a two-feature partial-dependence contour using the fitted model.

    Args:
        model: Fitted single-output or multi-output malchan pipeline.
        target: Output column to visualize.
        feature_names: Exactly two raw numeric feature names.
        output_index: Classification output index. Ignored for regression.

    Returns:
        Plotly contour figure generated from ``child_model.get_pd_2d``.
    """

    features = list(feature_names)
    if len(features) != 2 or features[0] == features[1]:
        raise ValueError("feature_names must contain two different features.")

    child_model = _resolve_child_model(model, target)
    x_pd, x_grid, y_grid = child_model.get_pd_2d(target_cols=features)
    array = np.asarray(x_pd, dtype=float)
    if array.ndim == 3:
        if not -array.shape[2] <= output_index < array.shape[2]:
            raise ValueError(
                f"output_index {output_index} is outside the available range "
                f"for shape {array.shape}."
            )
        array = array[:, :, output_index]
    if array.ndim != 2:
        raise ValueError(f"Unsupported 2D partial-dependence shape: {array.shape}")

    x_flat = np.asarray(x_grid).ravel()
    y_flat = np.asarray(y_grid).ravel()
    mean_values = np.nanmean(array, axis=1)
    x_values = _ordered_unique(x_flat)
    y_values = _ordered_unique(y_flat)
    expected = len(x_values) * len(y_values)
    if len(mean_values) != expected:
        raise ValueError(
            "The two-dimensional partial-dependence grid is inconsistent: "
            f"received {len(mean_values)} values for {len(x_values)} x-values "
            f"and {len(y_values)} y-values."
        )
    z_values = mean_values.reshape(len(y_values), len(x_values))

    fig = go.Figure()
    fig.add_trace(
        go.Contour(
            z=z_values,
            x=x_values,
            y=y_values,
            ncontours=25,
            colorscale="RdBu_r",
            contours_coloring="heatmap",
            colorbar={"title": "Prediction"},
            name="Partial dependence",
        )
    )

    training_x = _training_features(child_model)
    missing = [feature for feature in features if feature not in training_x.columns]
    if missing:
        raise ValueError(f"Training data is missing PD features: {missing}")
    marker: dict[str, Any] = {
        "size": 8,
        "line": {"color": "black", "width": 1},
    }
    observed_target = _training_target(child_model)
    if observed_target is not None and len(observed_target) == len(training_x):
        marker.update({"color": observed_target, "colorscale": "RdBu_r"})
    else:
        marker["color"] = "rgba(40,40,40,0.65)"

    fig.add_trace(
        go.Scatter(
            x=training_x[features[0]],
            y=training_x[features[1]],
            mode="markers",
            marker=marker,
            name="Observed data",
        )
    )
    fig.update_layout(
        title=f"2D partial dependence: {target}",
        xaxis_title=features[0],
        yaxis_title=features[1],
        width=700,
        height=620,
    )
    return fig


__all__ = [
    "show_model_diagnostics",
    "show_model_pd_2d",
    "visualization_outputs",
]
