"""Plotly figures used by the FastAPI web application.

The web layer serializes figures produced by the existing
``malchan.visualization.machine_learning_plots`` functions. These adapters only
normalize the model contract for single-output and multi-output pipelines; they
do not rebuild chart geometry or styling.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .machine_learning_plots import (
    show_pd_2d,
    show_pd_and_ice,
    show_shap_scatter,
    yy_plot_ml,
)


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


def _model_for_visualization(model: Any, target: str) -> Any:
    """Return a model satisfying the visualization functions' model contract."""

    if isinstance(getattr(model, "models", None), Mapping):
        return model
    _resolve_child_model(model, target)
    return _SingleOutputVisualizationAdapter(model, target)


def _training_features(child_model: Any) -> pd.DataFrame:
    """Return the raw feature frame used to fit a target-specific model."""

    values = (
        child_model._get_X()
        if hasattr(child_model, "_get_X")
        else getattr(child_model, "X", None)
    )
    if values is None:
        raise ValueError("Training features are unavailable for visualization.")
    return values if isinstance(values, pd.DataFrame) else pd.DataFrame(values)


def _training_target(child_model: Any) -> Any:
    """Return the target values stored by a target-specific model."""

    return (
        child_model._get_y()
        if hasattr(child_model, "_get_y")
        else getattr(child_model, "y", None)
    )


def _categorical_values(child_model: Any, feature_name: str) -> list[Any] | None:
    """Return configured category values when ``feature_name`` is categorical."""

    unique_values = (
        child_model._shared_attr("unique_cols")
        if hasattr(child_model, "_shared_attr")
        else getattr(child_model, "unique_cols", None)
    )
    if not isinstance(unique_values, Mapping) or feature_name not in unique_values:
        return None
    values = unique_values[feature_name]
    if values is None:
        return []
    return list(values)


def _categorical_pd_and_ice(
    child_model: Any,
    feature_name: str,
    categories: Sequence[Any],
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, Any]:
    """Compute categorical PD/ICE without modifying similarly named columns.

    The legacy implementation identifies categorical columns by substring.
    A feature such as ``material`` therefore also matches ``material_grade`` and
    overwrites both columns while creating the PD grid. This helper changes only
    the exact raw categorical column, which keeps the prediction frame valid.
    """

    X = _training_features(child_model)
    if feature_name not in X.columns:
        raise ValueError(
            f"Categorical feature {feature_name!r} is unavailable in training data."
        )
    category_values = np.asarray(list(categories), dtype=object)
    if category_values.size == 0:
        raise ValueError(f"Categorical feature {feature_name!r} has no categories.")

    X_sample = X.sample(300, random_state=0) if len(X) > 300 else X
    pd_values: list[np.ndarray] = []
    for row_index in range(len(X_sample)):
        frame = pd.concat(
            [X_sample.iloc[[row_index]].copy()] * len(category_values),
            ignore_index=True,
        )
        frame.loc[:, feature_name] = category_values
        prediction = child_model.predict(
            frame,
            proba=getattr(child_model, "task", "") == "classification",
        )
        array = np.asarray(getattr(prediction, "values", prediction))
        if array.ndim == 1:
            array = array.reshape(-1, 1)
        elif array.ndim == 2 and array.shape[1] > 1:
            array = array.reshape(-1, 1, array.shape[1])
        elif array.ndim != 2:
            raise ValueError(
                "Categorical partial-dependence predictions must be one- or "
                f"two-dimensional, received shape {array.shape}."
            )
        pd_values.append(array)

    return (
        np.concatenate(pd_values, axis=1),
        category_values,
        X,
        _training_target(child_model),
    )


def visualization_diagnostic_options(model: Any, target: str) -> dict[str, Any]:
    """Return task and cross-validation availability for one diagnostic target."""

    child_model = _resolve_child_model(model, target)
    task = str(getattr(child_model, "task", ""))
    cv_predictions = getattr(child_model, "cv_preds", None)
    if isinstance(cv_predictions, Mapping):
        cv_splits = [
            split
            for split in ("train", "test")
            if cv_predictions.get(split) is not None
        ]
    else:
        cv_splits = []

    cv_available = (
        all(split in cv_splits for split in ("train", "test"))
        if task == "regression"
        else bool(cv_splits)
    )
    return {
        "task": task,
        "cv_available": cv_available,
        "cv_splits": cv_splits,
    }


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
    train_test: str = "test",
) -> go.Figure:
    """Create the standard malchan Y-Y, residual, or classification figure."""

    options = visualization_diagnostic_options(model, target)
    if train_test not in {"train", "test"}:
        raise ValueError("train_test must be either 'train' or 'test'.")
    if residual and options["task"] == "classification":
        raise ValueError("Residual plots are available for regression targets only.")
    if cv and not options["cv_available"]:
        raise ValueError(
            "Cross-validation predictions are unavailable. Run cv_score() before "
            "requesting a CV diagnostic plot."
        )
    if cv and options["task"] == "classification" and train_test not in options["cv_splits"]:
        raise ValueError(
            f"Cross-validation predictions for split {train_test!r} are unavailable."
        )

    return yy_plot_ml(
        model=_model_for_visualization(model, target),
        target=target,
        cv=cv,
        residual=residual,
        train_test=train_test,
    )


def show_model_shap_scatter(
    model: Any,
    target: str,
    feature_name: str,
    *,
    interactive_col: str | None = None,
    target_item: Any = None,
) -> go.Figure:
    """Return the existing ``show_shap_scatter`` visualization without restyling.

    Args:
        model (Any): Registered single-output or multi-output model pipeline.
        target (str): Purpose-variable name whose cached SHAP values are used.
        feature_name (str): Feature placed on the horizontal axis.
        interactive_col (str | None): Optional feature used to colour or group points.
        target_item (Any): Optional classification output or class label.

    Returns:
        go.Figure: Plotly SHAP dependence-style scatter plot.

    Raises:
        ValueError: If the feature or interaction column is empty or invalid.
    """

    if not feature_name:
        raise ValueError("feature_name must not be empty.")
    if interactive_col == "":
        interactive_col = None
    return show_shap_scatter(
        model=_model_for_visualization(model, target),
        target=target,
        target_col=feature_name,
        interactive_col=interactive_col,
        target_item=target_item,
    )


def show_model_pd_and_ice(
    model: Any,
    target: str,
    feature_name: str,
    *,
    ice: bool = True,
    output_index: int = -1,
) -> go.Figure:
    """Return a one-dimensional PD/ICE figure for numeric or categorical input."""

    if not feature_name:
        raise ValueError("feature_name must not be empty.")

    child_model = _resolve_child_model(model, target)
    categories = _categorical_values(child_model, feature_name)
    if categories is None:
        return show_pd_and_ice(
            model=_model_for_visualization(model, target),
            target=target,
            target_col=feature_name,
            ice=ice,
            col_idx=output_index,
        )

    X_PD, ticks, X, y = _categorical_pd_and_ice(
        child_model,
        feature_name,
        categories,
    )
    figure = show_pd_and_ice(
        X_PD=X_PD,
        target_col=feature_name,
        xticks=ticks,
        X=X,
        y=y,
        ice=ice,
        col_idx=output_index,
    )
    figure.update_xaxes(
        type="category",
        categoryorder="array",
        categoryarray=list(ticks),
    )
    for trace in figure.data:
        if (
            getattr(trace, "type", "") == "scatter"
            and str(getattr(trace, "name", "")).startswith("Partial Dependence")
        ):
            trace.mode = "lines+markers"
    return figure


def show_model_pd_2d(
    model: Any,
    target: str,
    feature_names: Sequence[str],
    *,
    output_index: int = -1,
) -> go.Figure:
    """Return the existing ``show_pd_2d`` visualization without restyling."""

    features = list(feature_names)
    if len(features) != 2 or features[0] == features[1]:
        raise ValueError("feature_names must contain two different features.")
    return show_pd_2d(
        model=_model_for_visualization(model, target),
        target=target,
        target_cols=features,
        col_idx=output_index,
    )


__all__ = [
    "show_model_diagnostics",
    "show_model_pd_2d",
    "show_model_pd_and_ice",
    "show_model_shap_scatter",
    "visualization_diagnostic_options",
    "visualization_outputs",
]
