"""Application services used by FastAPI and future web clients."""

from .comparison_service import (
    ComparisonNotFoundError,
    install_comparison_service,
)
from .model_service import InMemoryModelService, ModelNotFoundError

install_comparison_service(InMemoryModelService)

__all__ = [
    "ComparisonNotFoundError",
    "InMemoryModelService",
    "ModelNotFoundError",
]
