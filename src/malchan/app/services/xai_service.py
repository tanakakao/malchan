"""Cached explainability service for registered models."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from malchan.app.schemas import (
    RecomputeXaiRequest,
    XaiImportanceItem,
    XaiImportanceResponse,
    XaiPdpResponse,
    XaiPdpSeries,
    XaiShapResponse,
    XaiSummaryResponse,
    XaiTargetSummary,
)


class XaiNotReadyError(LookupError):
    """Raised when cached explainability data is not available."""


def _state_store(service: Any) -> dict[str, dict[str, Any]]:
    """Return the process-local XAI state store for one service instance."""

    store = getattr(service, "_xai_states", None)
    if store is None:
        store = {}
        service._xai_states = store
    return store


def _target_models(registered: Any) -> dict[str, Any]:
    """Return target-specific child models for single or multi-output pipelines."""

    model_map = getattr(registered.model, "models", None)
    if isinstance(model_map, Mapping):
        return {
            target: model_map[target]
            for target in registered.info.target_cols
            if target in model_map
        }
    return {registered.info.target_cols[0]: registered.model}


def _shared_columns(model: Any, name: str) -> list[str]:
    """Read raw column metadata from a model or its shared context."""

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


def _aggregate_status(statuses: list[str]) -> str:
    """Combine target-level XAI states into one model-level state."""

    if not statuses or all(status == "not_requested" for status in statuses):
        return "not_requested"
    if all(status == "ready" for status in statuses):
        return "ready"
    if all(status == "unavailable" for status in statuses):
        return "unavailable"
    if any(status == "ready" for status in statuses):
        return "partial"
    if any(status == "computing" for status in statuses):
        return "computing"
    return "failed"


def _target_state(
    target: str,
    status: str,
    *,
    computed_at: datetime | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    """Create one internal target-state record."""

    return {
        "target": target,
        "status": status,
        "computed_at": computed_at,
        "error": error,
    }


def _compute_xai_for_registered(
    service: Any,
    model_id: str,
    targets: list[str] | None = None,
) -> XaiSummaryResponse:
    """Compute SHAP and ``get_xai`` once for selected registered targets."""

    registered = service._get_registered(model_id)
    target_models = _target_models(registered)
    selected = list(registered.info.target_cols) if not targets else list(targets)
    unknown = sorted(set(selected).difference(target_models))
    if unknown:
        raise ValueError(f"XAI targets contain unknown targets: {unknown}")

    store = _state_store(service)
    state = store.setdefault(
        model_id,
        {
            "requested": True,
            "targets": {
                target: _target_state(target, "not_requested")
                for target in registered.info.target_cols
            },
        },
    )
    state["requested"] = True

    for target in selected:
        child = target_models[target]
        state["targets"][target] = _target_state(target, "computing")
        registered.info.xai_status = "computing"
        shap_method = getattr(child, "shap", None)
        get_xai_method = getattr(child, "get_xai", None)
        if not callable(shap_method) or not callable(get_xai_method):
            state["targets"][target] = _target_state(
                target,
                "unavailable",
                error="The fitted model does not expose shap() and get_xai().",
            )
            continue

        try:
            shap_method()
            get_xai_method()
            cache = getattr(child, "importances", None)
            if not isinstance(cache, Mapping):
                raise RuntimeError("get_xai() did not create an importances cache.")
            state["targets"][target] = _target_state(
                target,
                "ready",
                computed_at=datetime.now(timezone.utc),
            )
        except Exception as exc:  # XAI failures must not invalidate the fitted model.
            state["targets"][target] = _target_state(
                target,
                "failed",
                computed_at=datetime.now(timezone.utc),
                error=f"{type(exc).__name__}: {exc}",
            )

    statuses = [
        state["targets"].get(target, _target_state(target, "not_requested"))["status"]
        for target in registered.info.target_cols
    ]
    state["status"] = _aggregate_status(statuses)
    registered.info.xai_status = state["status"]
    return get_xai_summary(service, model_id)


def _available_summary(
    target: str,
    child: Any,
    state: Mapping[str, Any],
) -> XaiTargetSummary:
    """Build target-level availability metadata from the cached model state."""

    cache = getattr(child, "importances", None)
    cache = cache if isinstance(cache, Mapping) else {}
    importance_methods = [
        method
        for method in ("model", "pfi", "shap")
        if cache.get(method) is not None
    ]
    shap_cache = cache.get("shap_pd")
    pdp_cache = cache.get("pd")
    shap_features = sorted(shap_cache) if isinstance(shap_cache, Mapping) else []
    pdp_features = sorted(pdp_cache) if isinstance(pdp_cache, Mapping) else []
    return XaiTargetSummary(
        target=target,
        status=state.get("status", "not_requested"),
        computed_at=state.get("computed_at"),
        error=state.get("error"),
        features=_shared_columns(child, "all_cols"),
        importance_methods=importance_methods,
        shap_features=shap_features,
        pdp_features=pdp_features,
    )


def get_xai_summary(service: Any, model_id: str) -> XaiSummaryResponse:
    """Return cached XAI availability without triggering computation."""

    registered = service._get_registered(model_id)
    store = _state_store(service)
    state = store.get(model_id)
    if state is None:
        state = {
            "requested": False,
            "status": "not_requested",
            "targets": {
                target: _target_state(target, "not_requested")
                for target in registered.info.target_cols
            },
        }
        store[model_id] = state
    target_models = _target_models(registered)
    summaries = {
        target: _available_summary(
            target,
            target_models[target],
            state["targets"].get(target, _target_state(target, "not_requested")),
        )
        for target in registered.info.target_cols
        if target in target_models
    }
    return XaiSummaryResponse(
        model_id=model_id,
        status=state.get("status", "not_requested"),
        targets=summaries,
    )


def recompute_xai(
    service: Any,
    model_id: str,
    request: RecomputeXaiRequest,
) -> XaiSummaryResponse:
    """Explicitly rebuild cached XAI data for selected targets."""

    return _compute_xai_for_registered(
        service,
        model_id,
        targets=request.targets or None,
    )


def refresh_xai_after_activation(
    service: Any,
    model_id: str,
) -> XaiSummaryResponse | None:
    """Recompute XAI after activating a compared model when originally requested."""

    state = _state_store(service).get(model_id)
    if state is None or not state.get("requested", False):
        return None
    return _compute_xai_for_registered(service, model_id)


def _cached_target(
    service: Any,
    model_id: str,
    target: str,
) -> tuple[Any, Mapping[str, Any]]:
    """Return a child model and its XAI cache without recomputing it."""

    registered = service._get_registered(model_id)
    target_models = _target_models(registered)
    if target not in target_models:
        raise ValueError(f"Unknown XAI target {target!r}.")
    child = target_models[target]
    cache = getattr(child, "importances", None)
    if not isinstance(cache, Mapping):
        raise XaiNotReadyError(
            f"Cached XAI is unavailable for target {target!r}. Train with compute_xai=true "
            "or call the recompute endpoint."
        )
    return child, cache


def _importance_names(child: Any, combined: bool) -> list[str]:
    """Resolve feature labels aligned with cached importance arrays."""

    if combined:
        names = [
            *_shared_columns(child, "num_cols"),
            *_shared_columns(child, "cat_cols"),
        ]
        if _shared_columns(child, "smiles_cols"):
            names.append("SMILES")
        if _shared_columns(child, "comp_cols"):
            names.append("Composition")
        return names
    frame = getattr(child, "df_prerpocessed", None)
    if isinstance(frame, pd.DataFrame):
        return [str(column) for column in frame.columns]
    return [str(column) for column in getattr(child, "feature_names", [])]


def _normalize_importance(values: Any, feature_count: int) -> np.ndarray:
    """Reduce model-specific importance shapes to one value per feature."""

    array = np.asarray(values, dtype=float)
    if array.ndim == 0:
        array = array.reshape(1)
    if array.ndim == 1:
        return array
    if array.shape[-1] == feature_count:
        axes = tuple(range(array.ndim - 1))
        return np.mean(np.abs(array), axis=axes)
    if array.shape[0] == feature_count:
        axes = tuple(range(1, array.ndim))
        return np.mean(np.abs(array), axis=axes)
    return array.reshape(-1)


def get_xai_importance(
    service: Any,
    model_id: str,
    target: str,
    method: str,
    combined: bool = True,
    top_n: int | None = None,
) -> XaiImportanceResponse:
    """Return cached model, PFI, or SHAP importance values."""

    if method not in {"model", "pfi", "shap"}:
        raise ValueError("method must be one of: model, pfi, shap.")
    child, cache = _cached_target(service, model_id, target)
    key = f"{method}_combine" if combined else method
    values = cache.get(key)
    if values is None and combined:
        combined = False
        values = cache.get(method)
    if values is None:
        raise XaiNotReadyError(
            f"Cached {method} importance is unavailable for target {target!r}."
        )

    names = _importance_names(child, combined)
    array = _normalize_importance(values, len(names))
    if len(names) != len(array):
        names = names[: len(array)] + [
            f"feature_{index}"
            for index in range(len(names), len(array))
        ]
    items = [
        XaiImportanceItem(feature=name, value=float(value))
        for name, value in zip(names, array)
        if np.isfinite(value)
    ]
    items.sort(key=lambda item: abs(item.value), reverse=True)
    if top_n is not None:
        items = items[:top_n]
    return XaiImportanceResponse(
        model_id=model_id,
        target=target,
        method=method,
        combined=combined,
        items=items,
    )


def get_xai_shap(
    service: Any,
    model_id: str,
    target: str,
    feature: str,
) -> XaiShapResponse:
    """Return cached SHAP scatter records for one raw feature."""

    _, cache = _cached_target(service, model_id, target)
    shap_cache = cache.get("shap_pd")
    if not isinstance(shap_cache, Mapping) or feature not in shap_cache:
        available = sorted(shap_cache) if isinstance(shap_cache, Mapping) else []
        raise ValueError(
            f"Unknown or unavailable SHAP feature {feature!r}. Available: {available}"
        )
    value = shap_cache[feature]
    if value is None:
        raise XaiNotReadyError(
            f"Cached SHAP scatter data is unavailable for feature {feature!r}."
        )
    frame = value if isinstance(value, pd.DataFrame) else pd.DataFrame(value)
    records = json.loads(frame.to_json(orient="records", date_format="iso"))
    value_columns = [
        str(column)
        for column in frame.columns
        if str(column).startswith("shap")
    ]
    return XaiShapResponse(
        model_id=model_id,
        target=target,
        feature=feature,
        value_columns=value_columns,
        records=records,
    )


def _json_scalar(value: Any) -> Any:
    """Convert NumPy scalar values used as PDP ticks into JSON-safe values."""

    if hasattr(value, "item"):
        return value.item()
    return value


def get_xai_pdp(
    service: Any,
    model_id: str,
    target: str,
    feature: str,
    include_ice: bool = False,
    max_ice: int = 30,
) -> XaiPdpResponse:
    """Return cached PDP means and optional ICE curves for one feature."""

    child, cache = _cached_target(service, model_id, target)
    pdp_cache = cache.get("pd")
    if not isinstance(pdp_cache, Mapping) or feature not in pdp_cache:
        available = sorted(pdp_cache) if isinstance(pdp_cache, Mapping) else []
        raise ValueError(
            f"Unknown or unavailable PDP feature {feature!r}. Available: {available}"
        )
    cached = pdp_cache[feature]
    if cached is None or len(cached) != 2:
        raise XaiNotReadyError(
            f"Cached PDP data is unavailable for feature {feature!r}."
        )

    values, ticks = cached
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim not in {2, 3}:
        raise ValueError(f"Unsupported cached PDP shape: {array.shape}")

    if array.ndim == 2:
        mean_values = array.mean(axis=1, keepdims=True)
        labels = [target]
    else:
        mean_values = array.mean(axis=1)
        target_items = getattr(child, "target_items", None)
        labels = (
            []
            if target_items is None
            else [str(value) for value in list(target_items)]
        )
        if len(labels) != mean_values.shape[1]:
            labels = [f"output_{index}" for index in range(mean_values.shape[1])]

    series: list[XaiPdpSeries] = []
    for output_index, label in enumerate(labels):
        ice_values = None
        if include_ice:
            if array.ndim == 2:
                ice_values = array[:, :max_ice].T.tolist()
            else:
                ice_values = array[:, :max_ice, output_index].T.tolist()
        series.append(
            XaiPdpSeries(
                name=label,
                pd_values=mean_values[:, output_index].tolist(),
                ice_values=ice_values,
            )
        )

    tick_values = np.asarray(ticks).tolist()
    return XaiPdpResponse(
        model_id=model_id,
        target=target,
        feature=feature,
        x_values=[_json_scalar(value) for value in tick_values],
        series=series,
    )


def install_xai_service(service_cls: type[Any]) -> None:
    """Attach cached XAI operations and wrap model lifecycle methods."""

    if getattr(service_cls, "_xai_service_installed", False):
        return
    original_train = service_cls.train
    original_delete = service_cls.delete

    def train_with_xai(self: Any, request: Any) -> Any:
        info = original_train(self, request)
        registered = self._get_registered(info.model_id)
        store = _state_store(self)
        store[info.model_id] = {
            "requested": bool(request.compute_xai),
            "status": "not_requested",
            "targets": {
                target: _target_state(target, "not_requested")
                for target in info.target_cols
            },
        }
        if request.compute_xai:
            _compute_xai_for_registered(self, info.model_id)
        else:
            registered.info.xai_status = "not_requested"
        return registered.info

    def delete_with_xai(self: Any, model_id: str) -> None:
        original_delete(self, model_id)
        _state_store(self).pop(model_id, None)

    service_cls.train = train_with_xai
    service_cls.delete = delete_with_xai
    service_cls.get_xai_summary = get_xai_summary
    service_cls.recompute_xai = recompute_xai
    service_cls.get_xai_importance = get_xai_importance
    service_cls.get_xai_shap = get_xai_shap
    service_cls.get_xai_pdp = get_xai_pdp
    service_cls.refresh_xai_after_activation = refresh_xai_after_activation
    service_cls._xai_service_installed = True


__all__ = [
    "XaiNotReadyError",
    "install_xai_service",
]
