"""Pipeline-level public API."""

from .single_output import SingleOutputMLModelPipeline
from .multi_output import MLModelPipeline
from .analysis_extensions import install_analysis_extensions
from .compositional_extensions import install_compositional_extensions

install_compositional_extensions(
    SingleOutputMLModelPipeline,
    MLModelPipeline,
)
install_analysis_extensions(
    SingleOutputMLModelPipeline,
    MLModelPipeline,
)

__all__ = ["SingleOutputMLModelPipeline", "MLModelPipeline"]