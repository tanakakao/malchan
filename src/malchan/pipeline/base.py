"""Shared pipeline types.

This module is intentionally small for now.  Common base classes, protocols, or
shared validation helpers for high-level pipeline classes should be added here
as the implementation is moved out of ``malchan.models``.
"""

from typing import Protocol


class FittablePipeline(Protocol):
    """Protocol for high-level pipelines that expose a fit method."""

    def fit(self, *args, **kwargs):
        """Fit the pipeline."""
