"""Backward-compatible re-exports for high-level pipeline classes."""

from ..pipeline.single_output import SingleOutputMLModelPipeline
from ..pipeline.multi_output import MLModelPipeline

__all__ = ["SingleOutputMLModelPipeline", "MLModelPipeline"]
