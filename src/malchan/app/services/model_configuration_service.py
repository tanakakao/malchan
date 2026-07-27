"""Model parameter metadata and registered-model evaluation services."""

from __future__ import annotations

import json
import math
from typing import Any

import pandas as pd
from optuna.distributions import (
    CategoricalDistribution,
    FloatDistribution,
    IntDistribution,
)

from malchan.app.schemas import (
    ModelEvaluationRequest,
    ModelEvaluationResponse,
    ModelParameterDefinition,
    ModelParameterSchemaResponse,
    TargetModelEvaluation,
)
from malchan.models.utils import (
    cls_default_params,
    get_param_grid_cls,
    get_param_grid_reg,
    reg_default_params,
)


def _json_value(value: Any) -> Any:
    """Return a JSON-compatible model parameter value."""

    if hasattr(value, "item"):
        value = value.item()
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    raise TypeError(f"{type(value).__name__} is not JSON serializable.")


def _frame_records(value: Any) -> list[dict[str, Any]]:
    """Convert a dataframe-like score result into JSON-safe records."""

    if value is None:
        return []
    frame = value if isinstance(value, pd.DataFrame) else pd.DataFrame(value)
    return json.loads(frame.to_json(orient="records", date_format="iso"))


def _numeric_default(
    default_value: Any,
    low: int | float,
    high: int | float,
    *,
    log: bool,
    integer: bool,
) -> int | float:
    """Resolve a usable default within a numeric distribution."""

    if isinstance(default_value, (int, float)) and not isinstance(default_value, bool):
        value = min(max(default_value, low), high)
    elif log and low > 0:
        value = math.sqrt(low * high)
    else:
        value = (low + high) / 2
    return int(round(value)) if integer else float(value)


def _parameter_definition(
    raw_name: str,
    distribution: Any,
    defaults: dict[str, Any],
) -> ModelParameterDefinition:
    """Convert one Optuna distribution into a Web control definition."""

    name = raw_name.removeprefix("predictor__")
    label = name.replace("_", " ")
    default_value = defaults.get(name)

    if isinstance(distribution, IntDistribution):
        return ModelParameterDefinition(
            name=name,
            label=label,
            control="integer",
            default_value=_numeric_default(
                default_value,
                distribution.low,
                distribution.high,
                log=distribution.log,
                integer=True,
            ),
            low=distribution.low,
            high=distribution.high,
            step=distribution.step,
            log=distribution.log,
        )

    if isinstance(distribution, FloatDistribution):
        return ModelParameterDefinition(
            name=name,
            label=label,
            control="float",
            default_value=_numeric_default(
                default_value,
                distribution.low,
                distribution.high,
                log=distribution.log,
                integer=False,
            ),
            low=distribution.low,
            high=distribution.high,
            step=distribution.step,
            log=distribution.log,
        )

    if isinstance(distribution, CategoricalDistribution):
        try:
            choices = [_json_value(choice) for choice in distribution.choices]
            resolved_default = _json_value(default_value)
            if resolved_default not in choices:
                resolved_default = choices[0] if choices else None
        except TypeError:
            try:
                resolved_default = _json_value(default_value)
            except TypeError:
                resolved_default = str(default_value)
            return ModelParameterDefinition(
                name=name,
                label=label,
                control="readonly",
                default_value=resolved_default,
                editable=False,
                note="複雑なPythonオブジェクトのため既定値を使用します。",
            )

        boolean_choices = bool(choices) and all(
            isinstance(choice, bool) for choice in choices
        )
        return ModelParameterDefinition(
            name=name,
            label=label,
            control="boolean" if boolean_choices else "categorical",
            default_value=resolved_default,
            choices=choices,
        )

    return ModelParameterDefinition(
        name=name,
        label=label,
        control="readonly",
        default_value=str(default_value),
        editable=False,
        note="Web画面で編集できないパラメータ形式です。",
    )


def get_model_parameter_schema(
    self: Any,
    task: str,
    model_name: str,
) -> ModelParameterSchemaResponse:
    """Return controls derived from the model's Optuna search space."""

    if task == "regression":
        grid = get_param_grid_reg(model_name)
        defaults = reg_default_params.get(model_name)
    elif task == "classification":
        grid = get_param_grid_cls(model_name)
        defaults = cls_default_params.get(model_name)
    else:
        raise ValueError("task must be 'regression' or 'classification'.")

    if defaults is None:
        raise ValueError(f"Unknown model name for {task}: {model_name}")
    parameters = [] if grid is None else [
        _parameter_definition(name, distribution, defaults)
        for name, distribution in grid.items()
    ]
    return ModelParameterSchemaResponse(
        task=task,
        model_name=model_name,
        parameters=parameters,
    )


def evaluate_model(
    self: Any,
    model_id: str,
    request: ModelEvaluationRequest,
) -> ModelEvaluationResponse:
    """Evaluate the currently registered model without selecting candidates."""

    registered = self._get_registered(model_id)
    target_tasks = dict(zip(registered.info.target_cols, registered.info.tasks))
    target_models = getattr(registered.model, "models", None)
    results: dict[str, TargetModelEvaluation] = {}

    for target in registered.info.target_cols:
        child_model = (
            target_models[target]
            if isinstance(target_models, dict) and target in target_models
            else registered.model
        )
        child_model.cv_score(
            method=request.method,
            n_splits=request.n_splits,
        )
        scores = getattr(child_model, "cv_scores", None)
        if not isinstance(scores, dict):
            raise RuntimeError(
                f"Cross-validation scores are unavailable for target {target!r}."
            )
        results[target] = TargetModelEvaluation(
            target=target,
            task=target_tasks[target],
            train=_frame_records(scores.get("train")),
            test=_frame_records(scores.get("test")),
        )

    return ModelEvaluationResponse(
        model_id=model_id,
        method=request.method,
        n_splits=request.n_splits,
        targets=results,
    )


def install_model_configuration_service(service_cls: type[Any]) -> None:
    """Attach model-configuration operations to the application service."""

    service_cls.get_model_parameter_schema = get_model_parameter_schema
    service_cls.evaluate_model = evaluate_model


__all__ = ["install_model_configuration_service"]
