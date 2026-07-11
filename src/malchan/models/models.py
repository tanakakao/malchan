"""Backward-compatible re-exports for high-level pipeline classes."""

from ..pipeline import MLModelPipeline, SingleOutputMLModelPipeline

__all__ = ["SingleOutputMLModelPipeline", "MLModelPipeline"]
