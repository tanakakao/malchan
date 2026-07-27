"""Service extensions for model comparison and best-model tuning."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import numpy as np
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


def _best_model(result: Any) -> Any | None:
    """Return the selected fitted model from a comparison-compatible result."""

    try:
        return getattr(result, "best_model", None)
    except (KeyError, TypeError):
        return None


def _actual_cv_values(model: Any) -> np.ndarray | None:
    """Return target values aligned to positional CV prediction indices."""

    prepare_y = getattr(model, "_prepare_cv_y", None)
    get_y = getattr(model, "_get_y", None)
    try:
        target = prepare_y(None) if callable(prepare_y) else get_y() if callable(get_y) else None
    except (AttributeError, TypeError, ValueError):
        return None
    if target is None:
        return None

    values = np.asarray(target).reshape(-1)
    if getattr(model, "task", None) == "classification":
        encoder = getattr(model, "le", None)
        inverse_transform = getattr(encoder, "inverse_transform", None)
        if callable(inverse_transform):
            try:
                values = np.asarray(inverse_transform(values.astype(int))).reshape(-1)
            except (TypeError, ValueError):
                pass
    return values


def _prediction_records(model: Any, value: Any) -> list[dict[str, Any]]:
    """Combine CV predictions with actual values for future Train/Test plots."""

    if value is None:
        return []
    frame = value if isinstance(value, pd.DataFrame) else pd.DataFrame(value)
    actual_values = _actual_cv_values(model)
    task = getattr(model, "task", None)
    target = str(getattr(model, "target_col", "target"))
    records: list[dict[str, Any]] = []

    class_labels: list[Any] = []
    if task == "classification":
        encoder = getattr(model, "le", None)
        inverse_transform = getattr(encoder, "inverse_transform", None)
        if callable(inverse_transform) and len(frame.columns):
            try:
                class_labels = list(inverse_transform(np.arange(len(frame.columns))))
            except (TypeError, ValueError):
                class_labels = []

    for row_index, row in frame.iterrows():
        record: dict[str, Any] = {"index": _json_safe(row_index)}
        try:
            position = int(row_index)
        except (TypeError, ValueError):
            position = -1
        if actual_values is not None and 0 <= position < len(actual_values):
            record["actual"] = _json_safe(actual_values[position])

        if task == "classification":
            numeric_values = pd.to_numeric(row, errors="coerce")
            for column, probability in numeric_values.items():
                label = str(column)
                prefix = f"{target}_"
                if label.startswith(prefix):
                    label = label[len(prefix):]
                record[f"probability_{label}"] = _json_safe(probability)
            valid = numeric_values.dropna()
            if not valid.empty:
                best_position = int(np.argmax(valid.to_numpy(dtype=float)))
                if len(class_labels) == len(frame.columns):
                    column_position = list(frame.columns).index(valid.index[best_position])
                    record["predicted"] = _json_safe(class_labels[column_position])
                else:
                    predicted_label = str(valid.index[best_position])
                    prefix = f"{target}_"
                    record["predicted"] = predicted_label.removeprefix(prefix)
        elif len(row):
            predicted = row[target] if target in row.index else row.iloc[0]
            record["predicted"] = _json_safe(predicted)

        records.append(record)
    return records


def _serialize_best_cv_predictions(result: Any) -> dict[str, list[dict[str, Any]]] | None:
    """Serialize Train/Test CV predictions for the selected best model."""

    model = _best_model(result)
    predictions = None if model is None else getattr(model, "cv_preds", None)
    if not isinstance(predictions, Mapping):
        return None
    serialized = {
        str(split): _prediction_records(model, frame)
        for split, frame in predictions.items()
    }
    return serialized if any(serialized.values()) else None


def _ensure_target_best_evaluation(result: Any) -> None:
    """Ensure the selected model retains both CV scores and CV predictions."""

    model = _best_model(result)
    if model is None:
        return
    if getattr(model, "cv_scores", None) is not None and getattr(model, "cv_preds", None) is not None:
        return
    cv_score = getattr(model, "cv_score", None)
    if not callable(cv_score):
        return
    cv_score(
        method=str(getattr(result, "method", "kfold")),
        n_splits=int(getattr(result, "n_splits", 5)),
    )


def _ensure_best_evaluation(result: Any) -> None:
    """Ensure best-model CV data for single- or multi-output comparison results."""

    child_results = getattr(result, "results", None)
    if isinstance(child_results, Mapping):
        for child_result in child_results.values():
            _ensure_target_best_evaluation(child_result)
        return
    _ensure_target_best_evaluation(result)


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
        best_cv_predictions=_serialize_best_cv_predictions(result),
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


def _activate_best_models(registered: Any, result: Any) -> None:
    """Promote selected best candidates for later prediction and inverse analysis."""

    child_results = getattr(result, "results", None)
    if child_results is None:
        best_model = getattr(result, "best_model", None)
        if best_model is None:
            raise RuntimeError("No successful best model is available to activate.")
        best_model.comparison_result = result
        registered.model = best_model
        target = registered.info.target_cols[0]
        registered.info.model_names_by_target[target] = [result.best_model_name]
        registered.info.model_names = [result.best_model_name]
        return

    model_map = getattr(registered.model, "models", None)
    if model_map is None:
        raise RuntimeError("The multi-output model does not expose target models.")
    for target, child_result in child_results.items():
        best_model = getattr(child_result, "best_model", None)
        if best_model is None:
            raise RuntimeError(
                f"No successful best model is available for target {target!r}."
            )
        model_map[target] = best_model
        registered.info.model_names_by_target[target] = [
            child_result.best_model_name
        ]
    registered.model.comparison_result = result


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

    _ensure_best_evaluation(result)
    registered.model.comparison_result = result
    if request.activate_best:
        _activate_best_models(registered, result)
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

    if request.evaluate or request.activate_best:
        _ensure_best_evaluation(result)
    registered.model.comparison_result = result
    if request.activate_best:
        _activate_best_models(registered, result)
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
