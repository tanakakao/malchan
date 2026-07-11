"""Service extensions for model comparison and best-model tuning."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import pandas as pd

from malchan.app.schemas import (
    CompareModelsRequest,
    ModelComparisonResponse,
    TargetComparisonResponse,
    TuneBestModelRequest,
)


class ComparisonNotFoundError(LookupError):
    """Raised when no comparison has been run for a registered model."""


def _json_default(value: Any) -> Any:
    """Convert common scientific Python values into JSON-compatible objects."""

    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, (set, tuple)):
        return list(value)
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _json_safe(value: Any) -> Any:
    """Round-trip an arbitrary comparison value through JSON."""

    return json.loads(json.dumps(value, default=_json_default))


def _frame_records(value: Any) -> list[dict[str, Any]]:
    """Convert a dataframe-like value into JSON-safe records."""

    if value is None:
        return []
    frame = value if isinstance(value, pd.DataFrame) else pd.DataFrame(value)
    return json.loads(frame.to_json(orient="records", date_format="iso"))


def _serialize_target_result(target: str, result: Any) -> TargetComparisonResponse:
    """Serialize one ``ModelComparisonResult``-compatible object."""

    cv_scores = getattr(result, "best_cv_scores", None)
    serialized_cv = None
    if cv_scores is not None:
        serialized_cv = {
            split: _frame_records(frame)
            for split, frame in cv_scores.items()
        }

    return TargetComparisonResponse(
        target=target,
        metric=str(getattr(result, "metric")),
        higher_is_better=bool(getattr(result, "higher_is_better")),
        ranking=_frame_records(getattr(result, "ranking")),
        failures=dict(getattr(result, "failures", {})),
        best_model_name=getattr(result, "best_model_name", None),
        best_params=_json_safe(getattr(result, "best_params", None)),
        best_is_tuned=bool(getattr(result, "best_is_tuned", False)),
        best_cv_scores=serialized_cv,
    )


def _serialize_comparison(
    model_id: str,
    target_cols: list[str],
    result: Any,
) -> ModelComparisonResponse:
    """Serialize single-output or multi-output comparison state."""

    child_results = getattr(result, "results", None)
    if child_results is None:
        child_results = {target_cols[0]: result}
    return ModelComparisonResponse(
        model_id=model_id,
        targets={
            target: _serialize_target_result(target, child_result)
            for target, child_result in child_results.items()
        },
    )


def _validate_target_mapping(
    value: Mapping[str, Any] | None,
    target_cols: list[str],
    field_name: str,
) -> None:
    """Reject per-target mappings containing unknown targets."""

    if value is None:
        return
    unknown = sorted(set(value).difference(target_cols))
    if unknown:
        raise ValueError(f"{field_name} contains unknown targets: {unknown}")


def run_comparison(
    self: Any,
    model_id: str,
    request: CompareModelsRequest,
) -> ModelComparisonResponse:
    """Compare candidates for a registered single- or multi-output model."""

    registered = self._get_registered(model_id)
    target_cols = list(registered.info.target_cols)
    is_multi_output = len(target_cols) > 1

    if is_multi_output:
        if isinstance(request.model_names, Mapping):
            _validate_target_mapping(request.model_names, target_cols, "model_names")
        if isinstance(request.metric, Mapping):
            _validate_target_mapping(request.metric, target_cols, "metric")
        if isinstance(request.tuning_trials, Mapping):
            _validate_target_mapping(
                request.tuning_trials,
                target_cols,
                "tuning_trials",
            )
        result = registered.model.compare(
            model_names=request.model_names,
            model_params=request.model_params,
            method=request.method,
            n_splits=request.n_splits,
            metric=request.metric,
            tuning=request.tuning,
            tune_best=request.tune_best,
            tuning_trials=request.tuning_trials,
            tuning_verbose=request.tuning_verbose,
            continue_on_error=request.continue_on_error,
        )
    else:
        target = target_cols[0]
        if isinstance(request.model_names, Mapping):
            raise ValueError(
                "A single-output model requires model_names as a list, not a mapping."
            )
        if isinstance(request.metric, Mapping):
            raise ValueError(
                "A single-output model requires metric as a string, not a mapping."
            )
        if isinstance(request.tuning_trials, Mapping):
            _validate_target_mapping(
                request.tuning_trials,
                target_cols,
                "tuning_trials",
            )
            tuning_trials = request.tuning_trials.get(target, 30)
        else:
            tuning_trials = request.tuning_trials
        result = registered.model.compare(
            model_names=request.model_names,
            model_params=request.model_params,
            method=request.method,
            n_splits=request.n_splits,
            metric=request.metric,
            tuning=request.tuning,
            tune_best=request.tune_best,
            tuning_trials=tuning_trials,
            tuning_verbose=request.tuning_verbose,
            continue_on_error=request.continue_on_error,
        )

    registered.model.comparison_result = result
    return _serialize_comparison(model_id, target_cols, result)


def get_comparison(self: Any, model_id: str) -> ModelComparisonResponse:
    """Return the latest comparison result for one registered model."""

    registered = self._get_registered(model_id)
    result = getattr(registered.model, "comparison_result", None)
    if result is None:
        raise ComparisonNotFoundError(model_id)
    return _serialize_comparison(
        model_id,
        list(registered.info.target_cols),
        result,
    )


def tune_best_comparison(
    self: Any,
    model_id: str,
    request: TuneBestModelRequest,
) -> ModelComparisonResponse:
    """Tune selected best candidates from the latest comparison result."""

    registered = self._get_registered(model_id)
    target_cols = list(registered.info.target_cols)
    result = getattr(registered.model, "comparison_result", None)
    if result is None:
        raise ComparisonNotFoundError(model_id)

    if len(target_cols) > 1:
        selected_targets = request.targets or None
        if isinstance(request.n_trials, Mapping):
            _validate_target_mapping(request.n_trials, target_cols, "n_trials")
        result.tune_best(
            targets=selected_targets,
            n_trials=request.n_trials,
            verbose=request.verbose,
            evaluate=request.evaluate,
        )
    else:
        target = target_cols[0]
        if request.targets and request.targets != [target]:
            raise ValueError(
                f"A single-output comparison only supports target {target!r}."
            )
        if isinstance(request.n_trials, Mapping):
            _validate_target_mapping(request.n_trials, target_cols, "n_trials")
            n_trials = request.n_trials.get(target, 30)
        else:
            n_trials = request.n_trials
        result.tune_best(
            n_trials=n_trials,
            verbose=request.verbose,
            evaluate=request.evaluate,
        )

    registered.model.comparison_result = result
    return _serialize_comparison(model_id, target_cols, result)


def install_comparison_service(service_cls: type[Any]) -> None:
    """Attach comparison operations to the in-memory model service."""

    service_cls.run_comparison = run_comparison
    service_cls.get_comparison = get_comparison
    service_cls.tune_best_comparison = tune_best_comparison


__all__ = [
    "ComparisonNotFoundError",
    "install_comparison_service",
]
