"""Schemas for model parameter controls and cross-validation evaluation."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .models import TaskType

ParameterControlType = Literal[
    "integer",
    "float",
    "categorical",
    "boolean",
    "readonly",
]


class ModelParameterDefinition(BaseModel):
    """One editable model parameter exposed to the Web application."""

    name: str
    label: str
    control: ParameterControlType
    default_value: Any = None
    low: int | float | None = None
    high: int | float | None = None
    step: int | float | None = None
    log: bool = False
    choices: list[Any] = Field(default_factory=list)
    editable: bool = True
    note: str | None = None


class ModelParameterSchemaResponse(BaseModel):
    """Parameter controls derived from the model tuning search space."""

    task: TaskType
    model_name: str
    parameters: list[ModelParameterDefinition] = Field(default_factory=list)


class ModelEvaluationRequest(BaseModel):
    """Cross-validation settings for evaluating one registered model."""

    model_config = ConfigDict(extra="forbid")

    method: Literal["kfold", "loo"] = "kfold"
    n_splits: int = Field(default=5, ge=2)


class TargetModelEvaluation(BaseModel):
    """Cross-validation metrics and OOF predictions for one output target."""

    target: str
    task: TaskType
    train: list[dict[str, Any]] = Field(default_factory=list)
    test: list[dict[str, Any]] = Field(default_factory=list)
    oof: dict[str, float] = Field(default_factory=dict)
    oof_predictions: list[dict[str, Any]] = Field(default_factory=list)


class ModelEvaluationResponse(BaseModel):
    """Cross-validation metrics for a registered model."""

    model_id: str
    method: Literal["kfold", "loo"]
    n_splits: int
    targets: dict[str, TargetModelEvaluation]
