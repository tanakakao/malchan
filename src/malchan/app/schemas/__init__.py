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
from .models import (
    HealthResponse,
    ModelInfo,
    ModelListResponse,
    PredictRequest,
    PredictionResponse,
    TrainModelRequest,
)
from .xai import (
    RecomputeXaiRequest,
    XaiImportanceItem,
    XaiImportanceResponse,
    XaiPdpResponse,
    XaiPdpSeries,
    XaiShapResponse,
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
    "ModelInfo",
    "ModelListResponse",
    "NumericSearchRange",
    "PredictRequest",
    "PredictionResponse",
    "RecomputeXaiRequest",
    "SumConstraint",
    "TargetComparisonResponse",
    "TrainModelRequest",
    "TuneBestModelRequest",
    "XaiImportanceItem",
    "XaiImportanceResponse",
    "XaiPdpResponse",
    "XaiPdpSeries",
    "XaiShapResponse",
    "XaiSummaryResponse",
    "XaiTargetSummary",
]
