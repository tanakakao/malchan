"""Schemas for trained-model structures and cached evaluation results."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from .model_configuration import ModelEvaluationResponse
from .models import TaskType


EstimatorNodeKind = Literal[
    "pipeline",
    "branch",
    "ensemble",
    "transformer",
    "estimator",
    "passthrough",
    "dropped",
    "reference",
]


class EstimatorStructureNode(BaseModel):
    """Framework-independent node describing a fitted estimator structure."""

    name: str
    class_name: str
    kind: EstimatorNodeKind
    columns: list[str] = Field(default_factory=list)
    parameters: dict[str, str] = Field(default_factory=dict)
    children: list[EstimatorStructureNode] = Field(default_factory=list)


class TargetModelDiagram(BaseModel):
    """Native structure diagram for one target-specific estimator."""

    target: str
    task: TaskType
    model_names: list[str] = Field(default_factory=list)
    structure: EstimatorStructureNode
    html: str = ""
    renderer: Literal["native", "sklearn", "text"] = "native"


class ModelVisualizationResponse(BaseModel):
    """Trained-model structures and the latest cached validation result."""

    model_id: str
    targets: list[TargetModelDiagram]
    evaluation: ModelEvaluationResponse | None = None


__all__ = [
    "EstimatorNodeKind",
    "EstimatorStructureNode",
    "ModelVisualizationResponse",
    "TargetModelDiagram",
]
