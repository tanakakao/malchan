"""Application layer for future FastAPI and web UI entry points."""

from importlib import import_module
from typing import Any

_LAZY_EXPORTS = {
    "create_app": "malchan.app.api",
    "AppSettings": "malchan.app.core",
    "get_settings": "malchan.app.core",
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    """Resolve application-layer objects lazily.

    Args:
        name: Public object name requested from :mod:`malchan.app`.

    Returns:
        The requested public object.

    Raises:
        AttributeError: If ``name`` is not exported by this package.
    """

    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module 'malchan.app' has no attribute {name!r}")

    module = import_module(_LAZY_EXPORTS[name])
    value = getattr(module, name)
    globals()[name] = value
    return value
