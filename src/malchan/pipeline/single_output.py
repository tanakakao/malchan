"""Single-output model pipeline public entrypoint.

``malchan.pipeline.single_output`` is the canonical public module for the
single-target model pipeline.  The heavy legacy implementation is still reused
internally until ``malchan.models.models`` is split further, but the public class
is now defined in this module instead of being exposed only through
``__getattr__`` indirection.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any, Dict

__all__ = ["SingleOutputMLModelPipeline"]


def _load_legacy_pipeline_class():
    """Load the current implementation class only when a pipeline is used."""

    module = import_module("malchan.models.models")
    return getattr(module, "SingleOutputMLModelPipeline")


class SingleOutputMLModelPipeline:
    """Public constructor for the single-output machine-learning pipeline.

    The implementation is delegated to the legacy class for now.  Keeping this
    wrapper as an actual class in ``malchan.pipeline.single_output`` lets the
    public API move first while preserving lazy loading and backwards-compatible
    runtime behavior.
    """

    def __new__(cls, *args: Any, **kwargs: Any):
        legacy_cls = _load_legacy_pipeline_class()
        return legacy_cls(*args, **kwargs)

    @classmethod
    def load_checkpoint(cls, state: Dict[str, Any]):
        """Restore a single-output pipeline from a checkpoint dictionary."""

        legacy_cls = _load_legacy_pipeline_class()
        return legacy_cls.load_checkpoint(state)
