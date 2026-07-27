"""Plotly figures used by the FastAPI web application.

The web layer serializes figures produced by the existing
``malchan.visualization.machine_learning_plots`` functions. These adapters only
normalize the model contract for single-output and multi-output pipelines; they
do not rebuild chart geometry or styling.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import plotly.graph_objects as go

from .machine_learning_plots import show_pd_2d, show_pd_and_ice, yy_plot_ml


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


def show_model_pd_and_ice(
    model: Any,
    target: str,
    feature_name: str,
    *,
    ice: bool = True,
    output_index: int = -1,
) -> go.Figure:
    """Return the existing ``show_pd_and_ice`` visualization without restyling."""

    if not feature_name:
        raise ValueError("feature_name must not be empty.")
    return show_pd_and_ice(
        model=_model_for_visualization(model, target),
        target=target,
        target_col=feature_name,
        ice=ice,
        col_idx=output_index,
    )


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
    "visualization_diagnostic_options",
    "visualization_outputs",
]
