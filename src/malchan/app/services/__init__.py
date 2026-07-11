"""Application services used by FastAPI and future web clients."""

from .comparison_service import (
    ComparisonNotFoundError,
    install_comparison_service,
)
from .model_service import InMemoryModelService, ModelNotFoundError
from .xai_comparison_hook import install_xai_comparison_hooks
from .xai_pdp_service import install_xai_pdp_service
from .xai_service import XaiNotReadyError, install_xai_service

install_comparison_service(InMemoryModelService)
install_xai_service(InMemoryModelService)
install_xai_pdp_service(InMemoryModelService)
install_xai_comparison_hooks(InMemoryModelService)

__all__ = [
    "ComparisonNotFoundError",
    "InMemoryModelService",
    "ModelNotFoundError",
    "XaiNotReadyError",
]
