"""Public API for :mod:`malchan.models`.

Pipeline classes are routed through :mod:`malchan.pipeline` so importing
``malchan.models`` does not eagerly import the large legacy implementation in
``malchan.models.models``.  This keeps the package boundary lightweight while
pipeline implementations are moved out in smaller follow-up changes.
"""

from importlib import import_module
from typing import Any

_LAZY_EXPORTS = {
    "MLModelPipeline": "malchan.pipeline",
    "SingleOutputMLModelPipeline": "malchan.pipeline",
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    """Resolve public model exports lazily."""

    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module 'malchan.models' has no attribute {name!r}")

    module = import_module(_LAZY_EXPORTS[name])
    value = getattr(module, name)
    globals()[name] = value
    return value
