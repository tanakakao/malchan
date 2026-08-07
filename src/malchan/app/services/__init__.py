"""Application services used by FastAPI and future web clients."""

from .comparison_service import (
    ComparisonNotFoundError,
    install_comparison_service,
)
from .compositional_service import install_compositional_service
from .model_configuration_service import install_model_configuration_service
from .model_service import InMemoryModelService, ModelNotFoundError
from .model_bundle_service import (
    InvalidModelBundleError,
    ModelBundleTooLargeError,
    ModelBundleUnavailableError,
    install_model_bundle_service,
)
from .model_visualization_service import install_model_visualization_service
from .xai_comparison_hook import install_xai_comparison_hooks
from .xai_service import XaiNotReadyError, install_xai_service
from .xai_shap_service import install_xai_shap_service

install_compositional_service(InMemoryModelService)
install_comparison_service(InMemoryModelService)
install_model_configuration_service(InMemoryModelService)
install_model_visualization_service(InMemoryModelService)
install_xai_service(InMemoryModelService)
install_xai_shap_service(InMemoryModelService)
install_xai_comparison_hooks(InMemoryModelService)
install_model_bundle_service(InMemoryModelService)

__all__ = [
    "ComparisonNotFoundError",
    "InMemoryModelService",
    "InvalidModelBundleError",
    "ModelBundleTooLargeError",
    "ModelBundleUnavailableError",
    "ModelNotFoundError",
    "XaiNotReadyError",
]