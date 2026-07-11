"""Public API for :mod:`malchan.models`."""

from . import compare
from ..pipeline import MLModelPipeline, SingleOutputMLModelPipeline
from .compare import ModelComparisonResult, MultiOutputModelComparisonResult

__all__ = [
    "MLModelPipeline",
    "ModelComparisonResult",
    "MultiOutputModelComparisonResult",
    "SingleOutputMLModelPipeline",
    "compare",
]
