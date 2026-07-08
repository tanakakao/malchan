"""Preprocessing public API.

This package is the intended home for preprocessing pipeline builders.
The current implementation still lives under ``malchan.models.pipelines`` and
will be moved here in small follow-up PRs.
"""

from importlib import import_module
from typing import Any

_LAZY_EXPORTS = {
    "make_pipeline": "malchan.preprocessing.compose",
    "make_preprocess": "malchan.preprocessing.compose",
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    """Resolve preprocessing helpers lazily."""

    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module 'malchan.preprocessing' has no attribute {name!r}")

    module = import_module(_LAZY_EXPORTS[name])
    value = getattr(module, name)
    globals()[name] = value
    return value
