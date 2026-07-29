"""Signed in-memory model bundle export and import services."""

from __future__ import annotations

import hashlib
import hmac
import json
import pickle
import re
import struct
import sys
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

_BUNDLE_MAGIC = b"MALCHAN-MODEL-BUNDLE\x00"
_BUNDLE_FORMAT_VERSION = 1
_HEADER_LENGTH = struct.Struct(">I")
_SIGNATURE_SIZE = hashlib.sha256().digest_size
_MAX_HEADER_BYTES = 64 * 1024
_DEFAULT_MAX_BUNDLE_BYTES = 256 * 1024 * 1024


class ModelBundleUnavailableError(RuntimeError):
    """Raised when secure model-bundle operations are not configured."""


class InvalidModelBundleError(ValueError):
    """Raised when a model bundle is malformed, modified, or incompatible."""


class ModelBundleTooLargeError(ValueError):
    """Raised when a model bundle exceeds the configured in-memory limit."""


def _secret_bytes(service: Any) -> bytes:
    """Return the configured HMAC key and reject weak or missing secrets."""

    value = getattr(service, "_model_bundle_secret", None)
    if value is None:
        raise ModelBundleUnavailableError(
            "モデルのダウンロード・読み込みを使用するには、"
            "MALCHAN_MODEL_BUNDLE_SECRETへ32文字以上の秘密値を設定してください。"
        )
    secret = value if isinstance(value, bytes) else str(value).encode("utf-8")
    if len(secret) < 32:
        raise ModelBundleUnavailableError(
            "MALCHAN_MODEL_BUNDLE_SECRETは32バイト以上にしてください。"
        )
    return secret


def _max_bundle_bytes(service: Any) -> int:
    """Return the configured maximum bundle size."""

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
    return f"malchan-{safe_target}-{safe_id}.malchan"


def _build_bundle(service: Any, model_id: str) -> tuple[bytes, str]:
    """Serialize and sign one registered model entirely in memory."""

    secret = _secret_bytes(service)
    registered = service._get_registered(model_id)
    columns = _column_metadata(registered.model)
    payload_object = {
        "format_version": _BUNDLE_FORMAT_VERSION,
        "model": registered.model,
        "info": registered.info.model_dump(mode="python"),
        "columns": columns,
        "evaluation": _evaluation_payload(service, model_id),
        "xai_state": _xai_payload(service, model_id),
    }
    try:
        payload = pickle.dumps(payload_object, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as exc:
        raise InvalidModelBundleError(
            f"学習済みモデルをシリアライズできませんでした: {type(exc).__name__}: {exc}"
        ) from exc

    header = {
        "format": "malchan-model-bundle",
        "format_version": _BUNDLE_FORMAT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "malchan_version": __version__,
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
        "original_model_id": registered.info.model_id,
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
    }
    header_bytes = json.dumps(
        header,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    if len(header_bytes) > _MAX_HEADER_BYTES:
        raise InvalidModelBundleError("モデルバンドルのヘッダーが大きすぎます。")

    signed = b"".join(
        (
            _BUNDLE_MAGIC,
            _HEADER_LENGTH.pack(len(header_bytes)),
            header_bytes,
            payload,
        )
    )
    signature = hmac.new(secret, signed, hashlib.sha256).digest()
    bundle = signed + signature
    if len(bundle) > _max_bundle_bytes(service):
        raise ModelBundleTooLargeError(
            "モデルファイルがMALCHAN_MODEL_BUNDLE_MAX_MBの上限を超えています。"
        )
    return bundle, _bundle_filename(registered.info)


def _parse_bundle(service: Any, bundle: bytes) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify one signed bundle before deserializing its trusted payload."""

    secret = _secret_bytes(service)
    if not isinstance(bundle, bytes | bytearray):
        raise InvalidModelBundleError("モデルファイルはバイト列で指定してください。")
    raw = bytes(bundle)
    if len(raw) > _max_bundle_bytes(service):
        raise ModelBundleTooLargeError(
            "モデルファイルがMALCHAN_MODEL_BUNDLE_MAX_MBの上限を超えています。"
        )

    minimum_size = len(_BUNDLE_MAGIC) + _HEADER_LENGTH.size + 2 + _SIGNATURE_SIZE
    if len(raw) < minimum_size or not raw.startswith(_BUNDLE_MAGIC):
        raise InvalidModelBundleError("malchanモデルファイルの形式ではありません。")

    header_length_offset = len(_BUNDLE_MAGIC)
    header_length = _HEADER_LENGTH.unpack_from(raw, header_length_offset)[0]
    if header_length < 2 or header_length > _MAX_HEADER_BYTES:
        raise InvalidModelBundleError("モデルファイルのヘッダー長が不正です。")

    header_start = header_length_offset + _HEADER_LENGTH.size
    header_end = header_start + header_length
    payload_end = len(raw) - _SIGNATURE_SIZE
    if header_end >= payload_end:
        raise InvalidModelBundleError("モデルファイルが途中で切れています。")

    signed = raw[:payload_end]
    supplied_signature = raw[payload_end:]
    expected_signature = hmac.new(secret, signed, hashlib.sha256).digest()
    if not hmac.compare_digest(supplied_signature, expected_signature):
        raise InvalidModelBundleError(
            "モデルファイルの署名を確認できません。改ざん、秘密鍵の相違、"
            "または別環境で作成されたファイルの可能性があります。"
        )

    try:
        header = json.loads(raw[header_start:header_end].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InvalidModelBundleError("モデルファイルのヘッダーが不正です。") from exc
    if header.get("format") != "malchan-model-bundle":
        raise InvalidModelBundleError("対応していないモデルファイル形式です。")
    if header.get("format_version") != _BUNDLE_FORMAT_VERSION:
        raise InvalidModelBundleError(
            f"モデルファイルの形式バージョン{header.get('format_version')!r}には対応していません。"
        )

    payload = raw[header_end:payload_end]
    if not hmac.compare_digest(
        str(header.get("payload_sha256", "")),
        hashlib.sha256(payload).hexdigest(),
    ):
        raise InvalidModelBundleError("モデルファイルの内容ハッシュが一致しません。")

    try:
        payload_object = pickle.loads(payload)
    except Exception as exc:
        raise InvalidModelBundleError(
            "モデルを復元できませんでした。作成時と同じmalchanおよび依存ライブラリが必要です。"
        ) from exc
    if not isinstance(payload_object, dict):
        raise InvalidModelBundleError("モデルファイルの内容が不正です。")
    if payload_object.get("format_version") != _BUNDLE_FORMAT_VERSION:
        raise InvalidModelBundleError("モデルファイル内部の形式バージョンが一致しません。")
    return header, payload_object


def configure_model_bundles(
    self: Any,
    secret: str | bytes | None,
    max_bytes: int = _DEFAULT_MAX_BUNDLE_BYTES,
) -> None:
    """Configure signed bundle operations without persisting the secret or models."""

    self._model_bundle_secret = secret
    self._model_bundle_max_bytes = max(1, int(max_bytes))


def export_model_bundle(self: Any, model_id: str) -> tuple[bytes, str]:
    """Return a signed downloadable model bundle and attachment filename."""

    return _build_bundle(self, model_id)


def import_model_bundle(self: Any, bundle: bytes) -> ModelBundleImportResponse:
    """Verify and restore one downloaded model into the process-local registry."""

    header, payload = _parse_bundle(self, bundle)
    try:
        info = ModelInfo.model_validate(payload["info"])
        model = payload["model"]
    except (KeyError, TypeError, ValueError) as exc:
        raise InvalidModelBundleError("モデルファイルに必要な情報がありません。") from exc
    if not callable(getattr(model, "predict", None)):
        raise InvalidModelBundleError("復元されたオブジェクトは予測モデルではありません。")

    columns = payload.get("columns") or {}
    if not isinstance(columns, dict):
        raise InvalidModelBundleError("モデルファイルの列情報が不正です。")
    normalized_columns: dict[str, list[str]] = {}
    for name in ("num_cols", "cat_cols", "smiles_cols", "comp_cols"):
        values = columns.get(name, [])
        if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
            raise InvalidModelBundleError(f"モデルファイルの{name}が不正です。")
        normalized_columns[name] = list(dict.fromkeys(values))

    with self._lock:
        new_model_id = None
        for _ in range(100):
            candidate = str(self._id_factory())
            if candidate not in self._models:
                new_model_id = candidate
                break
        if new_model_id is None:
            raise RuntimeError("読み込みモデルへ一意なIDを割り当てられませんでした。")
        restored_info = info.model_copy(update={"model_id": new_model_id})
        self._models[new_model_id] = _RegisteredModel(model=model, info=restored_info)

    evaluation_payload = payload.get("evaluation")
    if evaluation_payload is not None:
        evaluation = ModelEvaluationResponse.model_validate(evaluation_payload).model_copy(
            update={"model_id": new_model_id}
        )
        cache = getattr(self, "_model_evaluation_cache", None)
        if cache is None:
            cache = {}
            self._model_evaluation_cache = cache
        cache[new_model_id] = evaluation

    xai_state = payload.get("xai_state")
    if isinstance(xai_state, dict):
        store = getattr(self, "_xai_states", None)
        if store is None:
            store = {}
            self._xai_states = store
        store[new_model_id] = deepcopy(xai_state)

    return ModelBundleImportResponse(
        model=restored_info,
        original_model_id=str(header.get("original_model_id") or info.model_id),
        **normalized_columns,
    )


def install_model_bundle_service(service_cls: type[Any]) -> None:
    """Attach signed download and upload operations to the model service."""

    if getattr(service_cls, "_model_bundle_service_installed", False):
        return
    service_cls.configure_model_bundles = configure_model_bundles
    service_cls.export_model_bundle = export_model_bundle
    service_cls.import_model_bundle = import_model_bundle
    service_cls._model_bundle_service_installed = True


__all__ = [
    "InvalidModelBundleError",
    "ModelBundleTooLargeError",
    "ModelBundleUnavailableError",
    "install_model_bundle_service",
]
