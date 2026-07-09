"""Model pipeline builders public API."""

from importlib import import_module
from typing import Any

_LAZY_EXPORTS = {
    "make_pipeline": "malchan.models.pipelines.pipeline",
    "make_predictor": "malchan.models.pipelines.predictor_pipeline",
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    """Resolve model pipeline helpers lazily.

    Args:
        name (str): Exported helper name to resolve.

    Returns:
        Any: Requested helper object.

    Raises:
        AttributeError: If ``name`` is not exported by this package.
    """
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module 'malchan.models.pipelines' has no attribute {name!r}")

    module = import_module(_LAZY_EXPORTS[name])
    value = getattr(module, name)
    globals()[name] = value
    return value
