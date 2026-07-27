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
    "HealthResponse",
    "InverseAnalysisRequest",
    "InverseAnalysisResponse",
    "InverseObjective",
    "ModelComparisonResponse",
    "ModelEvaluationRequest",
    "ModelEvaluationResponse",
    "ModelInfo",
    "ModelListResponse",
    "ModelParameterDefinition",
    "ModelParameterSchemaResponse",
    "NumericSearchRange",
    "PlotlyFigureResponse",
    "PredictRequest",
    "PredictionResponse",
    "RecomputeXaiRequest",
    "SumConstraint",
    "TargetComparisonResponse",
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
