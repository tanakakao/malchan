"""Parsing helpers for JSON and optional Pydantic structured outputs."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any


def _strip_json_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def parse_json_payload(payload: Any) -> Any:
    """Convert an LLM response or JSON-like value to Python data."""

    if isinstance(payload, Mapping):
        return dict(payload)
    if isinstance(payload, (list, tuple)):
        return list(payload)
    if not isinstance(payload, str):
        raise TypeError(
            "Expected a JSON string, mapping, list, or tuple. "
            f"Got {type(payload).__name__}."
        )
    return json.loads(_strip_json_fence(payload))


def schema_from_response_model(response_model: Any) -> dict[str, Any]:
    """Return JSON Schema from a mapping or a Pydantic-compatible model class."""

    if isinstance(response_model, Mapping):
        return dict(response_model)
    builder = getattr(response_model, "model_json_schema", None)
    if callable(builder):
        schema = builder()
        if not isinstance(schema, Mapping):
            raise TypeError("model_json_schema() must return a mapping.")
        return dict(schema)
    raise TypeError(
        "response_model must be a JSON Schema mapping or expose "
        "model_json_schema()."
    )


def validate_structured_payload(payload: Any, response_model: Any) -> Any:
    """Validate parsed data with a Pydantic-compatible model when available.

    A raw JSON Schema mapping is returned as parsed Python data because malchan
    intentionally avoids a mandatory ``jsonschema`` dependency. Semantic model
    and preprocessing validation remains the responsibility of
    :class:`malchan.llm.SuggestionValidator`.
    """

    parsed = parse_json_payload(payload)
    if isinstance(response_model, Mapping):
        return parsed

    validator = getattr(response_model, "model_validate", None)
    if callable(validator):
        return validator(parsed)

    json_validator = getattr(response_model, "model_validate_json", None)
    if callable(json_validator):
        serialized = payload if isinstance(payload, str) else json.dumps(parsed)
        return json_validator(_strip_json_fence(serialized))

    raise TypeError(
        "response_model must expose model_validate() or model_validate_json()."
    )


def response_model_name(response_model: Any, default: str = "malchan_response") -> str:
    """Return a provider-safe schema name with at most 64 characters."""

    raw_name = getattr(response_model, "__name__", None) or default
    normalized = re.sub(r"[^A-Za-z0-9_-]+", "_", str(raw_name)).strip("_")
    return (normalized or default)[:64]
