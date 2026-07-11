"""Pipeline-level public API."""

from .single_output import SingleOutputMLModelPipeline
from .multi_output import MLModelPipeline
from .analysis_extensions import install_analysis_extensions

install_analysis_extensions(
    SingleOutputMLModelPipeline,
    MLModelPipeline,
)

__all__ = ["SingleOutputMLModelPipeline", "MLModelPipeline"]
