"""Scikit-learn diagram and cached evaluation services for registered models."""

from __future__ import annotations

import re
from html import escape
from typing import Any, Callable

from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.utils import estimator_html_repr

from malchan.app.schemas import (
    ModelEvaluationRequest,
    ModelEvaluationResponse,
    ModelVisualizationResponse,
    TargetModelDiagram,
)


def _evaluation_cache(service: Any) -> dict[str, ModelEvaluationResponse]:
    """Return the process-local validation-result cache."""

    cache = getattr(service, "_model_evaluation_cache", None)
    if cache is None:
        cache = {}
        service._model_evaluation_cache = cache
    return cache


def _target_model(registered_model: Any, target: str) -> Any:
    """Resolve one target-specific malchan pipeline."""

    children = getattr(registered_model, "models", None)
    if isinstance(children, dict) and target in children:
        return children[target]
    return registered_model


def _estimator_from_target_model(target_model: Any) -> Any:
    """Resolve the fitted sklearn-compatible estimator from a malchan pipeline."""

    estimator = getattr(target_model, "model", None)
    if estimator is None:
        estimator = target_model
    return estimator


def _display_leaf(estimator: Any) -> BaseEstimator:
    """Create a harmless sklearn estimator preserving a leaf's name and details."""

    class_name = re.sub(r"\W|^(?=\d)", "_", type(estimator).__name__) or "Estimator"
    details = repr(estimator)

    def fit(self: BaseEstimator, *_args: Any, **_kwargs: Any) -> BaseEstimator:
        return self

    def estimator_repr(self: BaseEstimator) -> str:
        return details

    display_type = type(
        class_name,
        (BaseEstimator,),
        {
            "fit": fit,
            "__repr__": estimator_repr,
            "__str__": estimator_repr,
            "__module__": __name__,
        },
    )
    return display_type()


def _display_estimator(estimator: Any, seen: set[int] | None = None) -> Any:
    """Copy an estimator's connection structure into sklearn display primitives."""

    if estimator is None or isinstance(estimator, str):
        return estimator

    visited = set() if seen is None else seen
    estimator_id = id(estimator)
    if estimator_id in visited:
        return _display_leaf(estimator)
    visited.add(estimator_id)

    steps = getattr(estimator, "steps", None)
    if isinstance(steps, (list, tuple)) and steps:
        normalized_steps = [
            (str(name), _display_estimator(child, visited.copy()))
            for name, child in steps
        ]
        return Pipeline(steps=normalized_steps)

    transformers = getattr(estimator, "transformers_", None)
    if not isinstance(transformers, (list, tuple)):
        transformers = getattr(estimator, "transformers", None)
    if isinstance(transformers, (list, tuple)) and transformers:
        normalized_transformers = []
        for name, transformer, columns in transformers:
            normalized_transformers.append(
                (
                    str(name),
                    transformer
                    if transformer in {"drop", "passthrough"}
                    else _display_estimator(transformer, visited.copy()),
                    columns,
                )
            )
        return ColumnTransformer(transformers=normalized_transformers)

    members = getattr(estimator, "estimators", None)
    if isinstance(members, (list, tuple)) and members and all(
        isinstance(item, tuple) and len(item) == 2 for item in members
    ):
        normalized_members = [
            (str(name), _display_estimator(member, visited.copy()))
            for name, member in members
            if member not in {None, "drop"}
        ]
        if normalized_members:
            return FeatureUnion(transformer_list=normalized_members)

    return _display_leaf(estimator)


def _diagram_html(estimator: Any) -> tuple[str, str]:
    """Return sklearn HTML and preserve a block diagram when direct rendering fails."""

    try:
        return estimator_html_repr(estimator), "sklearn"
    except Exception:
        try:
            display_estimator = _display_estimator(estimator)
            return estimator_html_repr(display_estimator), "sklearn"
        except Exception:
            text = escape(repr(estimator))
            return f'<pre class="malchan-estimator-fallback">{text}</pre>', "text"


def get_model_visualization(self: Any, model_id: str) -> ModelVisualizationResponse:
    """Return target-specific estimator diagrams and the latest evaluation result."""

    registered = self._get_registered(model_id)
    diagrams: list[TargetModelDiagram] = []
    task_by_target = dict(zip(registered.info.target_cols, registered.info.tasks))

    for target in registered.info.target_cols:
        target_model = _target_model(registered.model, target)
        estimator = _estimator_from_target_model(target_model)
        diagram, renderer = _diagram_html(estimator)
        diagrams.append(
            TargetModelDiagram(
                target=target,
                task=task_by_target[target],
                model_names=list(registered.info.model_names_by_target.get(target, [])),
                html=diagram,
                renderer=renderer,
            )
        )

    return ModelVisualizationResponse(
        model_id=model_id,
        targets=diagrams,
        evaluation=_evaluation_cache(self).get(model_id),
    )


def install_model_visualization_service(service_cls: type[Any]) -> None:
    """Attach model diagrams and evaluation caching to the model service."""

    if getattr(service_cls, "_model_visualization_service_installed", False):
        return

    original_evaluate: Callable[..., ModelEvaluationResponse] = service_cls.evaluate_model
    original_delete: Callable[..., None] = service_cls.delete

    def evaluate_and_cache(
        self: Any,
        model_id: str,
        request: ModelEvaluationRequest,
    ) -> ModelEvaluationResponse:
        result = original_evaluate(self, model_id, request)
        _evaluation_cache(self)[model_id] = result
        return result

    def delete_and_clear(self: Any, model_id: str) -> None:
        original_delete(self, model_id)
        _evaluation_cache(self).pop(model_id, None)

    service_cls.evaluate_model = evaluate_and_cache
    service_cls.get_model_visualization = get_model_visualization
    service_cls.delete = delete_and_clear
    service_cls._model_visualization_service_installed = True


__all__ = ["install_model_visualization_service"]
