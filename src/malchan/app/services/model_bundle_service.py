"""Trusted in-memory model artifact export and import services."""

from __future__ import annotations

import pickle
import re
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any

from malchan import __version__
from malchan.app.schemas import (
    ModelBundleImportResponse,
    ModelEvaluationResponse,
    ModelInfo,
)

from .model_service import _RegisteredModel

MODEL_ARTIFACT_FORMAT = "malchan-model-artifact"
MODEL_ARTIFACT_VERSION = 1
MODEL_ARTIFACT_SUFFIX = ".malchan"
_DEFAULT_MAX_BUNDLE_BYTES = 256 * 1024 * 1024


class ModelBundleUnavailableError(RuntimeError):
    """Backward-compatible error type for older API integrations."""


class InvalidModelBundleError(ValueError):
    """Raised when a model artifact is malformed or incompatible."""


class ModelBundleTooLargeError(ValueError):
    """Raised when a model artifact exceeds the configured in-memory limit."""


def _max_bundle_bytes(service: Any) -> int:
    """Return the configured maximum artifact size."""

    value = int(getattr(service, "_model_bundle_max_bytes", _DEFAULT_MAX_BUNDLE_BYTES))
    return max(1, value)


def _shared_value(model: Any, name: str) -> Any:
    """Read metadata from a model, shared context, or first child model."""

    shared_attr = getattr(model, "_shared_attr", None)
    if callable(shared_attr):
        try:
            value = shared_attr(name)
        except (AttributeError, KeyError, TypeError):
            value = None
        if value is not None:
            return value

    value = getattr(model, name, None)
    context = getattr(model, "context", None)
    if value is None and context is not None:
        value = getattr(context, name, None)
    if value is not None:
        return value

    children = getattr(model, "models", None)
    if isinstance(children, dict):
        for child in children.values():
            child_value = _shared_value(child, name)
            if child_value is not None:
                return child_value
    return None


def _column_metadata(model: Any) -> dict[str, list[str]]:
    """Return raw input-column groups needed to restore the Web workbench."""

    return {
        name: [] if (value := _shared_value(model, name)) is None else list(value)
        for name in ("num_cols", "cat_cols", "smiles_cols", "comp_cols")
    }


def _evaluation_payload(service: Any, model_id: str) -> dict[str, Any] | None:
    """Return one cached cross-validation result as a serializable dictionary."""

    cache = getattr(service, "_model_evaluation_cache", None)
    if not isinstance(cache, dict):
        return None
    evaluation = cache.get(model_id)
    if evaluation is None:
        return None
    if isinstance(evaluation, ModelEvaluationResponse):
        return evaluation.model_dump(mode="python")
    return ModelEvaluationResponse.model_validate(evaluation).model_dump(mode="python")


def _xai_payload(service: Any, model_id: str) -> dict[str, Any] | None:
    """Return cached XAI availability state when present."""

    store = getattr(service, "_xai_states", None)
    if not isinstance(store, dict) or model_id not in store:
        return None
    state = store[model_id]
    return deepcopy(state) if isinstance(state, dict) else None


def _bundle_filename(info: ModelInfo) -> str:
    """Create an attachment filename containing no path-sensitive characters."""

    target = "-".join(info.target_cols[:2]) or "model"
    safe_target = re.sub(r"[^A-Za-z0-9._-]+", "-", target).strip("-._") or "model"
    safe_id = re.sub(r"[^A-Za-z0-9_-]+", "", info.model_id)[:12] or "model"
    return f"malchan-{safe_target}-{safe_id}{MODEL_ARTIFACT_SUFFIX}"


def _build_model_artifact(service: Any, model_id: str) -> dict[str, Any]:
    """Build the canonical versioned artifact envelope."""

    registered = service._get_registered(model_id)
    return {
        "format": MODEL_ARTIFACT_FORMAT,
        "artifact_version": MODEL_ARTIFACT_VERSION,
        "malchan_version": __version__,
        "model": registered.model,
        "metadata": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "original_model_id": registered.info.model_id,
        },
        "state": {
            "info": registered.info.model_dump(mode="python"),
            "columns": _column_metadata(registered.model),
            "evaluation": _evaluation_payload(service, model_id),
            "xai_state": _xai_payload(service, model_id),
        },
    }


def _serialize_model_artifact(service: Any, model_id: str) -> tuple[bytes, str]:
    """Serialize one registered model as a trusted pickle-backed artifact."""

    artifact = _build_model_artifact(service, model_id)
    try:
        bundle = pickle.dumps(artifact, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as exc:
        raise InvalidModelBundleError(
            f"学習済みモデルをシリアライズできませんでした: {type(exc).__name__}: {exc}"
        ) from exc
    if len(bundle) > _max_bundle_bytes(service):
        raise ModelBundleTooLargeError(
            "モデルファイルがMALCHAN_MODEL_BUNDLE_MAX_MBの上限を超えています。"
        )
    info = ModelInfo.model_validate(artifact["state"]["info"])
    return bundle, _bundle_filename(info)


def _normalize_loaded_artifact(payload: Any) -> dict[str, Any]:
    """Validate one deserialized canonical model artifact."""

    if not isinstance(payload, dict):
        raise InvalidModelBundleError("malchanモデルファイルの内容が不正です。")
    if payload.get("format") != MODEL_ARTIFACT_FORMAT:
        raise InvalidModelBundleError("対応していないモデルファイル形式です。")
    try:
        artifact_version = int(payload.get("artifact_version", -1))
    except (TypeError, ValueError) as exc:
        raise InvalidModelBundleError("モデルファイルの形式バージョンが不正です。") from exc
    if artifact_version != MODEL_ARTIFACT_VERSION:
        raise InvalidModelBundleError(
            f"モデルファイルの形式バージョン{artifact_version!r}には対応していません。"
        )

    metadata = payload.get("metadata")
    state = payload.get("state")
    if not isinstance(metadata, dict) or not isinstance(state, dict):
        raise InvalidModelBundleError(
            "モデルファイルのmetadataまたはstateが不正です。"
        )
    return {
        **payload,
        "metadata": dict(metadata),
        "state": dict(state),
    }


def _deserialize_model_artifact(
    service: Any,
    bundle: bytes,
) -> dict[str, Any]:
    """Load a pickle-backed artifact after bounded byte reception."""

    if not isinstance(bundle, bytes | bytearray):
        raise InvalidModelBundleError("モデルファイルはバイト列で指定してください。")
    raw = bytes(bundle)
    if len(raw) > _max_bundle_bytes(service):
        raise ModelBundleTooLargeError(
            "モデルファイルがMALCHAN_MODEL_BUNDLE_MAX_MBの上限を超えています。"
        )
    if not raw:
        raise InvalidModelBundleError("モデルファイルが空です。")

    try:
        payload = pickle.loads(raw)
    except Exception as exc:
        raise InvalidModelBundleError(
            "モデルを復元できませんでした。信頼できるmalchan生成ファイルで、"
            "作成時と互換性のあるmalchanおよび依存ライブラリが必要です。"
        ) from exc
    return _normalize_loaded_artifact(payload)


def configure_model_bundles(
    self: Any,
    max_bytes: int = _DEFAULT_MAX_BUNDLE_BYTES,
) -> None:
    """Configure the maximum in-memory model artifact size."""

    self._model_bundle_max_bytes = max(1, int(max_bytes))


def export_model_bundle(self: Any, model_id: str) -> tuple[bytes, str]:
    """Return a downloadable model artifact and attachment filename."""

    return _serialize_model_artifact(self, model_id)


def import_model_bundle(
    self: Any,
    bundle: bytes,
) -> ModelBundleImportResponse:
    """Restore one trusted downloaded model into the process-local registry."""

    artifact = _deserialize_model_artifact(self, bundle)
    metadata = artifact["metadata"]
    state = artifact["state"]
    try:
        info = ModelInfo.model_validate(state["info"])
        model = artifact["model"]
    except (KeyError, TypeError, ValueError) as exc:
        raise InvalidModelBundleError("モデルファイルに必要な情報がありません。") from exc
    if not callable(getattr(model, "predict", None)):
        raise InvalidModelBundleError("復元されたオブジェクトは予測モデルではありません。")

    columns = state.get("columns") or {}
    if not isinstance(columns, dict):
        raise InvalidModelBundleError("モデルファイルの列情報が不正です。")
    normalized_columns: dict[str, list[str]] = {}
    for name in ("num_cols", "cat_cols", "smiles_cols", "comp_cols"):
        values = columns.get(name, [])
        if not isinstance(values, list) or not all(
            isinstance(value, str) for value in values
        ):
            raise InvalidModelBundleError(f"モデルファイルの{name}が不正です。")
        normalized_columns[name] = list(dict.fromkeys(values))

    evaluation_payload = state.get("evaluation")
    try:
        restored_evaluation = (
            None
            if evaluation_payload is None
            else ModelEvaluationResponse.model_validate(evaluation_payload)
        )
    except (TypeError, ValueError) as exc:
        raise InvalidModelBundleError("モデルファイルの精度検証結果が不正です。") from exc
    xai_state = state.get("xai_state")
    restored_xai_state = deepcopy(xai_state) if isinstance(xai_state, dict) else None

    with self._lock:
        new_model_id = None
        for _ in range(100):
            try:
                candidate = str(self._id_factory())
            except StopIteration as exc:
                raise RuntimeError("読み込みモデルへIDを割り当てられませんでした。") from exc
            if candidate not in self._models:
                new_model_id = candidate
                break
        if new_model_id is None:
            raise RuntimeError("読み込みモデルへ一意なIDを割り当てられませんでした。")

        restored_info = info.model_copy(update={"model_id": new_model_id})
        self._models[new_model_id] = _RegisteredModel(model=model, info=restored_info)
        if restored_evaluation is not None:
            cache = getattr(self, "_model_evaluation_cache", None)
            if cache is None:
                cache = {}
                self._model_evaluation_cache = cache
            cache[new_model_id] = restored_evaluation.model_copy(
                update={"model_id": new_model_id}
            )
        if restored_xai_state is not None:
            store = getattr(self, "_xai_states", None)
            if store is None:
                store = {}
                self._xai_states = store
            store[new_model_id] = restored_xai_state

    return ModelBundleImportResponse(
        model=restored_info,
        original_model_id=str(metadata.get("original_model_id") or info.model_id),
        **normalized_columns,
    )


def install_model_bundle_service(service_cls: type[Any]) -> None:
    """Attach trusted download and upload operations to the model service."""

    if getattr(service_cls, "_model_bundle_service_installed", False):
        return
    service_cls.configure_model_bundles = configure_model_bundles
    service_cls.export_model_bundle = export_model_bundle
    service_cls.import_model_bundle = import_model_bundle
    service_cls._model_bundle_service_installed = True


__all__ = [
    "InvalidModelBundleError",
    "MODEL_ARTIFACT_FORMAT",
    "MODEL_ARTIFACT_SUFFIX",
    "MODEL_ARTIFACT_VERSION",
    "ModelBundleTooLargeError",
    "ModelBundleUnavailableError",
    "install_model_bundle_service",
]
