"""Compositional-aware feature-importance aggregation for FastAPI XAI."""

from __future__ import annotations

from collections.abc import Mapping
from functools import wraps
from typing import Any

import numpy as np

from malchan.app.schemas import XaiImportanceItem, XaiImportanceResponse


def _target_model(service: Any, model_id: str, target: str) -> Any:
    """Return the fitted target-specific pipeline for one registered model."""

    registered = service._get_registered(model_id)
    model_map = getattr(registered.model, "models", None)
    if isinstance(model_map, Mapping):
        if target not in model_map:
            raise ValueError(f"Unknown XAI target {target!r}.")
        return model_map[target]
    if target not in registered.info.target_cols:
        raise ValueError(f"Unknown XAI target {target!r}.")
    return registered.model


def _shared_columns(model: Any, name: str) -> list[str]:
    """Read raw feature-column metadata from a model or shared context."""

    if hasattr(model, "_shared_attr"):
        try:
            value = model._shared_attr(name)
        except (AttributeError, TypeError):
            value = None
    else:
        value = getattr(model, name, None)
        context = getattr(model, "context", None)
        if value is None and context is not None:
            value = getattr(context, name, None)
    return [] if value is None else list(value)


def _compositional_groups(model: Any) -> list[list[str]]:
    """Return normalized compositional groups attached during model fitting."""

    groups = getattr(model, "compositional_groups", None) or []
    return [list(group) for group in groups]


def _sum_values(
    items: list[XaiImportanceItem],
    predicate: Any,
    *,
    magnitude: bool = False,
) -> float | None:
    """Aggregate matching importance values while preserving legacy semantics."""

    values = [item.value for item in items if predicate(item.feature)]
    if not values:
        return None
    array = np.asarray(values, dtype=float)
    if magnitude:
        return float(np.abs(array).sum())
    return float(array.sum())


def _aggregate_compositional_importance(
    child: Any,
    raw: XaiImportanceResponse,
    *,
    top_n: int | None,
) -> XaiImportanceResponse:
    """Aggregate transformed coordinates into raw-variable/group-level importance."""

    groups = _compositional_groups(child)
    grouped_columns = {column for group in groups for column in group}
    items: list[XaiImportanceItem] = []

    for column in _shared_columns(child, "num_cols"):
        if column in grouped_columns:
            continue
        value = _sum_values(raw.items, lambda feature, column=column: feature == column)
        if value is not None:
            items.append(XaiImportanceItem(feature=column, value=value))

    for index, group in enumerate(groups):
        prefix = f"compositional_{index}__"
        value = _sum_values(
            raw.items,
            lambda feature, prefix=prefix: feature.startswith(prefix),
            magnitude=True,
        )
        if value is not None:
            label = f"組成比: {' / '.join(group)}"
            items.append(XaiImportanceItem(feature=label, value=value))

    for column in _shared_columns(child, "cat_cols"):
        value = _sum_values(
            raw.items,
            lambda feature, column=column: (
                feature == column or feature.startswith(f"{column}_")
            ),
        )
        if value is not None:
            items.append(XaiImportanceItem(feature=column, value=value))

    smiles_value = _sum_values(
        raw.items,
        lambda feature: feature.startswith("smiles__") or feature.startswith("smiles_"),
    )
    if smiles_value is not None:
        items.append(XaiImportanceItem(feature="SMILES", value=smiles_value))

    composition_value = _sum_values(
        raw.items,
        lambda feature: feature.startswith("comp__") or feature.startswith("comp_"),
    )
    if composition_value is not None:
        items.append(XaiImportanceItem(feature="Composition", value=composition_value))

    items.sort(key=lambda item: abs(item.value), reverse=True)
    if top_n is not None:
        items = items[:top_n]

    return XaiImportanceResponse(
        model_id=raw.model_id,
        target=raw.target,
        method=raw.method,
        combined=True,
        items=items,
    )


def install_compositional_xai_service(service_cls: type[Any]) -> None:
    """Make combined XAI importance include compositional groups by default."""

    if getattr(service_cls, "_compositional_xai_service_installed", False):
        return

    original_get_xai_importance = service_cls.get_xai_importance

    @wraps(original_get_xai_importance)
    def get_xai_importance(
        self: Any,
        model_id: str,
        target: str,
        method: str,
        combined: bool = True,
        top_n: int | None = None,
    ) -> XaiImportanceResponse:
        child = _target_model(self, model_id, target)
        if not combined or not _compositional_groups(child):
            return original_get_xai_importance(
                self,
                model_id,
                target,
                method,
                combined=combined,
                top_n=top_n,
            )

        raw = original_get_xai_importance(
            self,
            model_id,
            target,
            method,
            combined=False,
            top_n=None,
        )
        return _aggregate_compositional_importance(
            child,
            raw,
            top_n=top_n,
        )

    service_cls.get_xai_importance = get_xai_importance
    service_cls._compositional_xai_service_installed = True


__all__ = ["install_compositional_xai_service"]
