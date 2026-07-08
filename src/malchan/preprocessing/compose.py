"""Composable preprocessing and estimator pipeline entrypoints.

The current implementation is still located in ``malchan.models.pipelines``.
This module provides the new import location first, so the implementation can be
moved without changing user-facing imports in later PRs.
"""

from importlib import import_module
from typing import Any

__all__ = ["make_pipeline", "make_preprocess"]


def __getattr__(name: str) -> Any:
    """Resolve preprocessing builders lazily."""

    if name == "make_pipeline":
        module = import_module("malchan.models.pipelines.pipeline")
    elif name == "make_preprocess":
        module = import_module("malchan.models.pipelines.preprocess_pipeline")
    else:
        raise AttributeError(f"module 'malchan.preprocessing.compose' has no attribute {name!r}")

    value = getattr(module, name)
    globals()[name] = value
    return value
