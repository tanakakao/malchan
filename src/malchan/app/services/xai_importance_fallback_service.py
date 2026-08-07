"""Recover feature importance even when full XAI computation partially fails."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from functools import wraps
from typing import Any


def _target_models(registered: Any) -> dict[str, Any]:
    """Return target-specific fitted pipelines."""

    model_map = getattr(registered.model, "models", None)
    if isinstance(model_map, Mapping):
        return {
            target: model_map[target]
            for target in registered.info.target_cols
            if target in model_map
        }
    return {
        target: registered.model
        for target in registered.info.target_cols
    }


def _try_importance(child: Any, method_name: str) -> tuple[Any | None, str | None]:
    """Call one importance method without invalidating the fitted model."""

    method = getattr(child, method_name, None)
    if not callable(method):
        return None, f"{method_name} is unavailable."
    try:
        return method(), None
    except Exception as exc:  # explainability must not invalidate training.
        return None, f"{type(exc).__name__}: {exc}"


def _recover_child_importance(child: Any) -> tuple[list[str], list[str]]:
    """Populate independent importance caches from whatever is still available."""

    existing = getattr(child, "importances", None)
    cache = dict(existing) if isinstance(existing, Mapping) else {}
    errors: list[str] = []

    if cache.get("model") is None:
        value, error = _try_importance(child, "model_importance")
        if value is not None:
            cache["model"] = value
        elif error is not None:
            errors.append(f"model: {error}")

    if cache.get("pfi") is None:
        value, error = _try_importance(child, "pfi_importance")
        if value is not None:
            cache["pfi"] = value
        elif error is not None:
            errors.append(f"pfi: {error}")

    if cache.get("shap") is None and getattr(child, "shap_values", None) is not None:
        value, error = _try_importance(child, "shap_importance")
        if value is not None:
            cache["shap"] = value
        elif error is not None:
            errors.append(f"shap: {error}")

    child.importances = cache
    available = [
        method
        for method in ("model", "pfi", "shap")
        if cache.get(method) is not None
    ]
    return available, errors


def _aggregate_status(statuses: list[str]) -> str:
    """Resolve model-level XAI status after importance recovery."""

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


def _recover_registered_importance(
    service: Any,
    model_id: str,
    targets: list[str] | None = None,
) -> None:
    """Recover per-target importance and make it immediately consumable by Web XAI."""

    registered = service._get_registered(model_id)
    target_models = _target_models(registered)
    selected = list(registered.info.target_cols) if not targets else list(targets)
    unknown = sorted(set(selected).difference(target_models))
    if unknown:
        raise ValueError(f"XAI targets contain unknown targets: {unknown}")

    store = getattr(service, "_xai_states", None)
    if not isinstance(store, dict):
        return
    state = store.get(model_id)
    if not isinstance(state, dict):
        return

    target_states = state.setdefault("targets", {})
    now = datetime.now(timezone.utc)
    for target in selected:
        available, fallback_errors = _recover_child_importance(target_models[target])
        if not available:
            continue

        target_state = target_states.setdefault(
            target,
            {
                "target": target,
                "status": "not_requested",
                "computed_at": None,
                "error": None,
            },
        )
        previous_error = target_state.get("error")
        target_state["status"] = "ready"
        target_state["computed_at"] = now
        if previous_error and fallback_errors:
            target_state["error"] = (
                f"{previous_error} Importance fallback: {'; '.join(fallback_errors)}"
            )
        elif previous_error:
            target_state["error"] = previous_error
        elif fallback_errors:
            target_state["error"] = f"Importance fallback: {'; '.join(fallback_errors)}"

    statuses = [
        target_states.get(target, {}).get("status", "not_requested")
        for target in registered.info.target_cols
    ]
    state["status"] = _aggregate_status(statuses)
    registered.info.xai_status = state["status"]


def install_xai_importance_fallback_service(service_cls: type[Any]) -> None:
    """Ensure default importance remains available after partial XAI failures."""

    if getattr(service_cls, "_xai_importance_fallback_installed", False):
        return

    original_train = service_cls.train
    original_recompute_xai = service_cls.recompute_xai

    @wraps(original_train)
    def train(self: Any, request: Any) -> Any:
        info = original_train(self, request)
        if getattr(request, "compute_xai", False):
            _recover_registered_importance(self, info.model_id)
            return self._get_registered(info.model_id).info
        return info

    @wraps(original_recompute_xai)
    def recompute_xai(self: Any, model_id: str, request: Any) -> Any:
        original_recompute_xai(self, model_id, request)
        targets = list(getattr(request, "targets", None) or []) or None
        _recover_registered_importance(self, model_id, targets=targets)
        return self.get_xai_summary(model_id)

    service_cls.train = train
    service_cls.recompute_xai = recompute_xai
    service_cls._xai_importance_fallback_installed = True


__all__ = [
    "install_xai_importance_fallback_service",
]
