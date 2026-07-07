"""Single-output model pipeline entrypoint.

The implementation currently lives in ``malchan.models.models`` and will be
moved here after the remaining old absolute imports are replaced.  Keeping this
module as the public entrypoint lets follow-up PRs move implementation details
without changing user-facing imports.
"""

from importlib import import_module
from typing import Any

__all__ = ["SingleOutputMLModelPipeline"]


def __getattr__(name: str) -> Any:
    """Resolve the single-output pipeline class lazily."""

    if name != "SingleOutputMLModelPipeline":
        raise AttributeError(f"module 'malchan.pipeline.single_output' has no attribute {name!r}")

    module = import_module("malchan.models.models")
    value = getattr(module, name)
    globals()[name] = value
    return value
