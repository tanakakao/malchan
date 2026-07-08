"""Pipeline-level public API.

This package is the new home for high-level model pipeline objects.
The current implementation is still being moved out of ``malchan.models`` in
small steps, so exports are resolved lazily to keep ``import malchan.pipeline``
lightweight.
"""

from importlib import import_module
from typing import Any

_LAZY_EXPORTS = {
    "SingleOutputMLModelPipeline": "malchan.pipeline.single_output",
    "MLModelPipeline": "malchan.pipeline.multi_output",
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    """Resolve pipeline classes lazily."""

    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module 'malchan.pipeline' has no attribute {name!r}")

    module = import_module(_LAZY_EXPORTS[name])
    value = getattr(module, name)
    globals()[name] = value
    return value
