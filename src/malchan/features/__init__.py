"""Feature-generation public API.

This package is the intended home for chemistry and materials feature builders.
The concrete implementation still lives inside preprocessing-related modules and
will be moved here in small follow-up PRs.
"""

from importlib import import_module
from typing import Any

_LAZY_EXPORTS = {
    "chemistry": "malchan.features.chemistry",
    "materials": "malchan.features.materials",
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    """Resolve feature subpackages lazily."""

    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module 'malchan.features' has no attribute {name!r}")

    module = import_module(_LAZY_EXPORTS[name])
    globals()[name] = module
    return module
