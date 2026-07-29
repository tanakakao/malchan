"""Schemas for trained-model diagrams and cached evaluation results."""

from typing import Literal

from pydantic import BaseModel, Field

from .model_configuration import ModelEvaluationResponse
from .models import TaskType


class TargetModelDiagram(BaseModel):
    """Scikit-learn style diagram for one target-specific estimator."""

    target: str
    task: TaskType
    model_names: list[str] = Field(default_factory=list)
    html: str
    renderer: Literal["sklearn", "text"] = "sklearn"


class ModelVisualizationResponse(BaseModel):
    """Trained-model diagrams and the latest cached validation result."""

    model_id: str
    targets: list[TargetModelDiagram]
    evaluation: ModelEvaluationResponse | None = None


__all__ = ["ModelVisualizationResponse", "TargetModelDiagram"]
