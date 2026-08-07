"""Service hook passing FastAPI compositional settings into model pipelines."""

from __future__ import annotations

from functools import wraps
from typing import Any

from malchan.pipeline.compositional_extensions import compositional_training_context


def install_compositional_service(service_cls: type[Any]) -> None:
    """Wrap ``train`` so nested pipeline calls inherit compositional settings."""

    if getattr(service_cls, "_compositional_service_installed", False):
        return

    original_train = service_cls.train

    @wraps(original_train)
    def train(self: Any, request: Any) -> Any:
        with compositional_training_context(
            compositional_groups=getattr(request, "compositional_groups", ()),
            compositional_method=getattr(request, "compositional_method", "ILR"),
            compositional_zero_replacement=getattr(
                request,
                "compositional_zero_replacement",
                1e-6,
            ),
            compositional_closure=getattr(request, "compositional_closure", True),
            compositional_alr_reference=getattr(
                request,
                "compositional_alr_reference",
                -1,
            ),
            compositional_scale_type=getattr(
                request,
                "compositional_scale_type",
                None,
            ),
        ):
            return original_train(self, request)

    service_cls.train = train
    service_cls._compositional_service_installed = True


__all__ = ["install_compositional_service"]
