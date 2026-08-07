"""Runtime extensions wiring compositional preprocessing into public pipelines.

The legacy pipeline classes intentionally keep their long-standing ``fit``
signatures.  This module accepts the new compositional keyword arguments at the
runtime boundary, stores them on fitted pipeline objects, and injects them into
the existing ``make_pipeline`` builder without changing positional APIs.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
from typing import Any


@dataclass(frozen=True, slots=True)
class CompositionalTrainingSettings:
    """Resolved compositional preprocessing settings for one training call."""

    groups: tuple[tuple[str, ...], ...] = ()
    method: str | None = "ILR"
    zero_replacement: float | None = 1e-6
    closure: bool = True
    alr_reference: int | str = -1
    scale_type: str | None = None


_DEFAULT_SETTINGS = CompositionalTrainingSettings()
_ACTIVE_SETTINGS: ContextVar[CompositionalTrainingSettings | None] = ContextVar(
    "malchan_compositional_training_settings",
    default=None,
)


def _normalize_groups(value: Any) -> tuple[tuple[str, ...], ...]:
    """Normalize list-like compositional groups for stable storage."""

    if value is None:
        return ()
    return tuple(tuple(group) for group in value)


def _settings_from_kwargs(kwargs: dict[str, Any]) -> CompositionalTrainingSettings:
    """Consume compositional fit kwargs, falling back to the active API context."""

    active = _ACTIVE_SETTINGS.get() or _DEFAULT_SETTINGS
    groups = _normalize_groups(kwargs.pop("compositional_groups", active.groups))
    method = kwargs.pop("compositional_method", active.method)
    zero_replacement = kwargs.pop(
        "compositional_zero_replacement",
        active.zero_replacement,
    )
    closure = kwargs.pop("compositional_closure", active.closure)
    alr_reference = kwargs.pop(
        "compositional_alr_reference",
        active.alr_reference,
    )
    scale_type = kwargs.pop("compositional_scale_type", active.scale_type)
    return CompositionalTrainingSettings(
        groups=groups,
        method=method,
        zero_replacement=zero_replacement,
        closure=bool(closure),
        alr_reference=alr_reference,
        scale_type=scale_type,
    )


def current_compositional_training_settings() -> CompositionalTrainingSettings:
    """Return settings active for the current synchronous training context."""

    return _ACTIVE_SETTINGS.get() or _DEFAULT_SETTINGS


@contextmanager
def compositional_training_context(
    *,
    compositional_groups: Any = (),
    compositional_method: str | None = "ILR",
    compositional_zero_replacement: float | None = 1e-6,
    compositional_closure: bool = True,
    compositional_alr_reference: int | str = -1,
    compositional_scale_type: str | None = None,
) -> Iterator[CompositionalTrainingSettings]:
    """Temporarily expose FastAPI compositional settings to nested pipeline fits."""

    settings = CompositionalTrainingSettings(
        groups=_normalize_groups(compositional_groups),
        method=compositional_method,
        zero_replacement=compositional_zero_replacement,
        closure=bool(compositional_closure),
        alr_reference=compositional_alr_reference,
        scale_type=compositional_scale_type,
    )
    token = _ACTIVE_SETTINGS.set(settings)
    try:
        yield settings
    finally:
        _ACTIVE_SETTINGS.reset(token)


def _apply_settings(model: Any, settings: CompositionalTrainingSettings) -> None:
    """Persist compositional settings on a pipeline instance."""

    model.compositional_groups = [list(group) for group in settings.groups]
    model.compositional_method = settings.method
    model.compositional_zero_replacement = settings.zero_replacement
    model.compositional_closure = settings.closure
    model.compositional_alr_reference = settings.alr_reference
    model.compositional_scale_type = settings.scale_type


def _single_make_pipeline(self: Any) -> Any:
    """Build the existing estimator pipeline with compositional arguments."""

    from ..models.pipelines import make_pipeline

    model_items = make_pipeline(
        model_names=self.model_names,
        task=self.task,
        num_cols=self._shared_attr("num_cols"),
        cat_cols=self._shared_attr("cat_cols"),
        smiles_cols=self._shared_attr("smiles_cols"),
        comp_cols=self._shared_attr("comp_cols"),
        num_impute_type=self.num_impute_type,
        num_scale_type=self.num_scale_type,
        cat_impute=self.cat_impute,
        fingerprints=self.fingerprints,
        comp_method=self.comp_method,
        comp_feats=self.comp_feats,
        poly=self.poly,
        poly_degree=self.poly_degree,
        poly_interaction_only=self.poly_interaction_only,
        decomposition=self.decomposition,
        decomposition_method=self.decomposition_method,
        n_components=self.dec_n_components,
        ensemble=self.ensemble,
        ens_type=self.ens_type,
        base_model=self.base_model,
        model_params=self.model_params,
        base_model_params=self.base_model_param,
        compositional_groups=getattr(self, "compositional_groups", ()),
        compositional_method=getattr(self, "compositional_method", "ILR"),
        compositional_zero_replacement=getattr(
            self,
            "compositional_zero_replacement",
            1e-6,
        ),
        compositional_closure=getattr(self, "compositional_closure", True),
        compositional_alr_reference=getattr(
            self,
            "compositional_alr_reference",
            -1,
        ),
        compositional_scale_type=getattr(self, "compositional_scale_type", None),
    )
    return model_items[0]


def install_compositional_extensions(
    single_output_cls: type[Any],
    multi_output_cls: type[Any],
) -> None:
    """Attach compositional fit support to the existing public pipeline classes."""

    if getattr(single_output_cls, "_compositional_extensions_installed", False):
        return

    original_single_fit = single_output_cls.fit
    original_single_fit_from_context = single_output_cls.fit_from_context
    original_single_make_pipeline = single_output_cls._make_pipeline
    original_multi_fit = multi_output_cls.fit

    @wraps(original_single_fit)
    def single_fit(self: Any, *args: Any, **kwargs: Any) -> Any:
        settings = _settings_from_kwargs(kwargs)
        _apply_settings(self, settings)
        return original_single_fit(self, *args, **kwargs)

    @wraps(original_single_fit_from_context)
    def single_fit_from_context(
        self: Any,
        context: Any,
        target_col: str,
        **kwargs: Any,
    ) -> Any:
        settings = _settings_from_kwargs(kwargs)
        _apply_settings(self, settings)
        return original_single_fit_from_context(
            self,
            context=context,
            target_col=target_col,
            **kwargs,
        )

    @wraps(original_single_make_pipeline)
    def single_make_pipeline(self: Any) -> Any:
        return _single_make_pipeline(self)

    @wraps(original_multi_fit)
    def multi_fit(self: Any, *args: Any, **kwargs: Any) -> Any:
        settings = _settings_from_kwargs(kwargs)
        _apply_settings(self, settings)
        with compositional_training_context(
            compositional_groups=settings.groups,
            compositional_method=settings.method,
            compositional_zero_replacement=settings.zero_replacement,
            compositional_closure=settings.closure,
            compositional_alr_reference=settings.alr_reference,
            compositional_scale_type=settings.scale_type,
        ):
            return original_multi_fit(self, *args, **kwargs)

    single_output_cls.fit = single_fit
    single_output_cls.fit_from_context = single_fit_from_context
    single_output_cls._make_pipeline = single_make_pipeline
    multi_output_cls.fit = multi_fit
    single_output_cls._compositional_extensions_installed = True
    multi_output_cls._compositional_extensions_installed = True


__all__ = [
    "CompositionalTrainingSettings",
    "compositional_training_context",
    "current_compositional_training_settings",
    "install_compositional_extensions",
]