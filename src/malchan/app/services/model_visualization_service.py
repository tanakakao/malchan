"""Native model structures and cached evaluation services for registered models."""

from __future__ import annotations

from typing import Any, Callable

from malchan.app.schemas import (
    ModelEvaluationRequest,
    ModelEvaluationResponse,
    ModelVisualizationResponse,
    TargetModelDiagram,
)

from .estimator_structure import build_estimator_structure


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
    """Resolve the fitted estimator from a malchan pipeline wrapper."""

    estimator = getattr(target_model, "model", None)
    return target_model if estimator is None else estimator


def get_model_visualization(self: Any, model_id: str) -> ModelVisualizationResponse:
    """Return target-specific native structures and the latest evaluation result."""

    registered = self._get_registered(model_id)
    task_by_target = dict(zip(registered.info.target_cols, registered.info.tasks))
    diagrams: list[TargetModelDiagram] = []

    for target in registered.info.target_cols:
        target_model = _target_model(registered.model, target)
        estimator = _estimator_from_target_model(target_model)
        diagrams.append(
            TargetModelDiagram(
                target=target,
                task=task_by_target[target],
                model_names=list(registered.info.model_names_by_target.get(target, [])),
                structure=build_estimator_structure(estimator),
                renderer="native",
            )
        )

    return ModelVisualizationResponse(
        model_id=model_id,
        targets=diagrams,
        evaluation=_evaluation_cache(self).get(model_id),
    )


def install_model_visualization_service(service_cls: type[Any]) -> None:
    """Attach model structures and evaluation caching to the model service."""

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
