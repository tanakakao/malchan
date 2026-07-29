"""Request and response schemas for the FastAPI application."""

from .comparison import (
    CompareModelsRequest,
    ModelComparisonResponse,
    TargetComparisonResponse,
    TuneBestModelRequest,
)
from .inverse_analysis import (
    InverseAnalysisRequest,
    InverseAnalysisResponse,
    InverseObjective,
    NumericSearchRange,
    SumConstraint,
)
from .model_configuration import (
    ModelEvaluationRequest,
    ModelEvaluationResponse,
    ModelParameterDefinition,
    ModelParameterSchemaResponse,
    TargetModelEvaluation,
)
from .model_visualization import (
    EstimatorNodeKind,
    EstimatorStructureNode,
    ModelVisualizationResponse,
    TargetModelDiagram,
)
from .models import (
    HealthResponse,
    ModelInfo,
    ModelListResponse,
    PredictRequest,
    PredictionResponse,
    TrainModelRequest,
)
from .visualization import PlotlyFigureResponse
from .xai import (
    LocalShapRequest,
    LocalShapResponse,
    LocalShapTargetResponse,
    RecomputeXaiRequest,
    XaiImportanceItem,
    XaiImportanceResponse,
    XaiPdpResponse,
    XaiPdpSeries,
    XaiShapResponse,
    XaiShapValuesResponse,
    XaiSummaryResponse,
    XaiTargetSummary,
)

__all__ = [
    "CompareModelsRequest",
    "EstimatorNodeKind",
    "EstimatorStructureNode",
    "HealthResponse",
    "InverseAnalysisRequest",
    "InverseAnalysisResponse",
    "InverseObjective",
    "LocalShapRequest",
    "LocalShapResponse",
    "LocalShapTargetResponse",
    "ModelComparisonResponse",
    "ModelEvaluationRequest",
    "ModelEvaluationResponse",
    "ModelInfo",
    "ModelListResponse",
    "ModelParameterDefinition",
    "ModelParameterSchemaResponse",
    "ModelVisualizationResponse",
    "NumericSearchRange",
    "PlotlyFigureResponse",
    "PredictRequest",
    "PredictionResponse",
    "RecomputeXaiRequest",
    "SumConstraint",
    "TargetComparisonResponse",
    "TargetModelDiagram",
    "TargetModelEvaluation",
    "TrainModelRequest",
    "TuneBestModelRequest",
    "XaiImportanceItem",
    "XaiImportanceResponse",
    "XaiPdpResponse",
    "XaiPdpSeries",
    "XaiShapResponse",
    "XaiShapValuesResponse",
    "XaiSummaryResponse",
    "XaiTargetSummary",
]
