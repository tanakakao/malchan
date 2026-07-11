"""Public inverse-analysis API."""

from typing import Any

from .models import inverse_analysis as _inverse_analysis


def inverse_analysis(model: Any, *args: Any, **kwargs: Any):
    """Run inverse analysis against a model or an application-layer adapter."""

    resolved_model = getattr(model, "_model", model)
    return _inverse_analysis(resolved_model, *args, **kwargs)


__all__ = ["inverse_analysis"]
