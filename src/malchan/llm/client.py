"""OpenAI and Gemini adapters for structured configuration planning."""

from __future__ import annotations

import os
import ssl
import warnings
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .configs import LLMConfig, coerce_llm_config
from .parser import (
    response_model_name,
    schema_from_response_model,
    validate_structured_payload,
)


@dataclass(slots=True)
class LLMResponse:
    """Normalized response returned by a provider adapter."""

    text: str
    parsed: Any | None = None
    raw: Any | None = None
    provider: str | None = None
    model: str | None = None
    usage: dict[str, Any] | None = None
    request_id: str | None = None


class BaseLLMClient(ABC):
    """Provider-independent interface used by malchan planners."""

    @abstractmethod
    def generate_json(
        self,
        prompt: str,
        *,
        schema: dict[str, Any] | None = None,
        schema_name: str = "malchan_response",
        strict: bool | None = None,
    ) -> LLMResponse:
        """Generate a JSON response, optionally constrained by JSON Schema."""

        raise NotImplementedError

    def generate_structured(
        self,
        prompt: str,
        response_model: Any,
        *,
        schema_name: str | None = None,
        strict: bool | None = None,
    ) -> LLMResponse:
        """Generate and validate a Pydantic-compatible structured response."""

        resolved_name = schema_name or response_model_name(response_model)
        response = self.generate_json(
            prompt,
            schema=schema_from_response_model(response_model),
            schema_name=resolved_name,
            strict=strict,
        )
        response.parsed = validate_structured_payload(response.text, response_model)
        return response

    def close(self) -> None:
        """Release provider resources when the SDK exposes a close method."""

        client = getattr(self, "client", None)
        close = getattr(client, "close", None)
        if callable(close):
            close()


def _resolve_api_key(config: LLMConfig) -> str | None:
    if config.api_key:
        return config.api_key
    return os.environ.get(config.resolved_api_key_env())


def _resolve_ca_bundle_path(config: LLMConfig) -> Path | None:
    raw_path = config.ca_bundle_path
    if raw_path is None:
        for env_name in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"):
            value = os.environ.get(env_name)
            if value:
                raw_path = value
                break
    if raw_path is None:
        return None

    path = Path(os.path.expandvars(str(raw_path))).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"CA bundle file was not found: {path}")
    return path


def _build_ssl_context(config: LLMConfig) -> ssl.SSLContext | None:
    ca_bundle_path = _resolve_ca_bundle_path(config)
    if config.ssl_verify and ca_bundle_path is None:
        return None

    context = ssl.create_default_context()
    if ca_bundle_path is not None:
        context.load_verify_locations(cafile=str(ca_bundle_path))
    if not config.ssl_verify:
        warnings.warn(
            "LLMConfig.ssl_verify=False disables HTTPS certificate verification. "
            "Use ca_bundle_path or SSL_CERT_FILE for normal operation.",
            RuntimeWarning,
            stacklevel=2,
        )
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
    return context


def _build_openai_http_client(config: LLMConfig) -> Any | None:
    context = _build_ssl_context(config)
    if context is None and config.proxy is None:
        return None
    try:
        import httpx
    except ImportError as exc:  # pragma: no cover - optional runtime path
        raise _missing_dependency("OpenAI HTTP configuration", "httpx") from exc

    kwargs: dict[str, Any] = {}
    if context is not None:
        kwargs["verify"] = context
    if config.proxy is not None:
        kwargs["proxy"] = config.proxy
    return httpx.Client(**kwargs)


def _gemini_http_options(config: LLMConfig) -> dict[str, Any]:
    options: dict[str, Any] = {}
    if config.api_version:
        options["api_version"] = config.api_version
    if config.base_url:
        options["base_url"] = config.base_url
    if config.timeout is not None:
        options["timeout"] = int(config.timeout * 1000)
    if config.max_retries:
        options["retry_options"] = {"attempts": config.max_retries + 1}

    context = _build_ssl_context(config)
    client_args: dict[str, Any] = {}
    if context is not None:
        client_args["verify"] = context
    if config.proxy is not None:
        client_args["proxy"] = config.proxy
    if client_args:
        options["client_args"] = dict(client_args)
        options["async_client_args"] = dict(client_args)
    return options


def _usage_to_dict(usage: Any) -> dict[str, Any] | None:
    if usage is None:
        return None
    if hasattr(usage, "model_dump"):
        dumped = usage.model_dump(exclude_none=True)
        return dict(dumped) if isinstance(dumped, Mapping) else None
    if isinstance(usage, Mapping):
        return dict(usage)

    names = (
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "prompt_tokens",
        "completion_tokens",
        "prompt_token_count",
        "candidates_token_count",
        "total_token_count",
    )
    data = {name: getattr(usage, name) for name in names if hasattr(usage, name)}
    return data or None


def _request_id(response: Any) -> str | None:
    for name in ("_request_id", "request_id", "id"):
        value = getattr(response, name, None)
        if value:
            return str(value)
    sdk_response = getattr(response, "sdk_http_response", None)
    headers = getattr(sdk_response, "headers", None)
    if isinstance(headers, Mapping):
        for key in ("x-request-id", "x-goog-request-id"):
            if headers.get(key):
                return str(headers[key])
    return None


def _missing_dependency(provider: str, package: str) -> ImportError:
    return ImportError(
        f"LLM provider {provider!r} requires optional dependency {package!r}. "
        "Install with `pip install -e .[llm]` or install the package directly."
    )


def _merge_request_kwargs(
    base: dict[str, Any],
    extra: Mapping[str, Any],
    *,
    reserved: set[str],
) -> dict[str, Any]:
    conflicts = sorted(set(extra).intersection(reserved))
    if conflicts:
        raise ValueError(
            "LLMConfig.request_kwargs must not override managed fields: "
            f"{conflicts}"
        )
    base.update(dict(extra))
    return base


class OpenAIClient(BaseLLMClient):
    """OpenAI adapter using the Responses API with a compatibility fallback."""

    def __init__(self, config: LLMConfig, *, client: Any | None = None) -> None:
        self.config = config
        if client is not None:
            self.client = client
            return

        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise _missing_dependency("openai", "openai") from exc

        api_key = _resolve_api_key(config)
        if not api_key:
            raise ValueError(
                "OpenAI API key was not found. Set "
                f"{config.resolved_api_key_env()} or pass LLMConfig.api_key."
            )

        kwargs: dict[str, Any] = {
            "api_key": api_key,
            "max_retries": config.max_retries,
        }
        if config.timeout is not None:
            kwargs["timeout"] = config.timeout
        if config.base_url:
            kwargs["base_url"] = config.base_url
        http_client = _build_openai_http_client(config)
        if http_client is not None:
            kwargs["http_client"] = http_client
        for key in {"api_key", "max_retries", "timeout", "base_url", "http_client"}:
            if key in config.client_kwargs:
                raise ValueError(f"LLMConfig.client_kwargs must not override {key!r}.")
        kwargs.update(config.client_kwargs)
        self.client = OpenAI(**kwargs)

    def generate_json(
        self,
        prompt: str,
        *,
        schema: dict[str, Any] | None = None,
        schema_name: str = "malchan_response",
        strict: bool | None = None,
    ) -> LLMResponse:
        resolved_strict = self.config.strict_schema if strict is None else strict
        responses_api = getattr(self.client, "responses", None)
        if responses_api is not None and hasattr(responses_api, "create"):
            return self._generate_with_responses(
                prompt,
                schema=schema,
                schema_name=schema_name,
                strict=resolved_strict,
            )
        return self._generate_with_chat_completions(
            prompt,
            schema=schema,
            schema_name=schema_name,
            strict=resolved_strict,
        )

    def _generate_with_responses(
        self,
        prompt: str,
        *,
        schema: dict[str, Any] | None,
        schema_name: str,
        strict: bool,
    ) -> LLMResponse:
        format_payload: dict[str, Any]
        if schema is None:
            format_payload = {"type": "json_object"}
        else:
            format_payload = {
                "type": "json_schema",
                "name": schema_name[:64],
                "schema": schema,
                "strict": strict,
            }
        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "instructions": "Return valid JSON only. Do not use markdown fences.",
            "input": prompt,
            "text": {"format": format_payload},
            "max_output_tokens": self.config.max_output_tokens,
            "store": self.config.store,
        }
        if self.config.temperature is not None:
            kwargs["temperature"] = self.config.temperature
        kwargs = _merge_request_kwargs(
            kwargs,
            self.config.request_kwargs,
            reserved={
                "model",
                "instructions",
                "input",
                "text",
                "max_output_tokens",
                "store",
                "temperature",
            },
        )
        response = self.client.responses.create(**kwargs)
        text = getattr(response, "output_text", "") or ""
        return LLMResponse(
            text=text,
            raw=response,
            provider="openai",
            model=self.config.model,
            usage=_usage_to_dict(getattr(response, "usage", None)),
            request_id=_request_id(response),
        )

    def _generate_with_chat_completions(
        self,
        prompt: str,
        *,
        schema: dict[str, Any] | None,
        schema_name: str,
        strict: bool,
    ) -> LLMResponse:
        response_format: dict[str, Any]
        if schema is None:
            response_format = {"type": "json_object"}
        else:
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name[:64],
                    "schema": schema,
                    "strict": strict,
                },
            }
        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": "Return valid JSON only."},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": self.config.max_output_tokens,
            "response_format": response_format,
        }
        if self.config.temperature is not None:
            kwargs["temperature"] = self.config.temperature
        kwargs = _merge_request_kwargs(
            kwargs,
            self.config.request_kwargs,
            reserved={
                "model",
                "messages",
                "max_tokens",
                "response_format",
                "temperature",
            },
        )
        response = self.client.chat.completions.create(**kwargs)
        text = response.choices[0].message.content or ""
        return LLMResponse(
            text=text,
            raw=response,
            provider="openai",
            model=self.config.model,
            usage=_usage_to_dict(getattr(response, "usage", None)),
            request_id=_request_id(response),
        )


class GeminiClient(BaseLLMClient):
    """Google Gen AI SDK adapter using ``models.generate_content``."""

    def __init__(self, config: LLMConfig, *, client: Any | None = None) -> None:
        self.config = config
        if client is not None:
            self.client = client
            return

        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise _missing_dependency("gemini", "google-genai") from exc

        api_key = _resolve_api_key(config)
        if not api_key:
            raise ValueError(
                "Gemini API key was not found. Set "
                f"{config.resolved_api_key_env()} or pass LLMConfig.api_key."
            )
        kwargs: dict[str, Any] = {"api_key": api_key}
        http_options = _gemini_http_options(config)
        if http_options:
            kwargs["http_options"] = types.HttpOptions(**http_options)
        for key in {"api_key", "http_options"}:
            if key in config.client_kwargs:
                raise ValueError(f"LLMConfig.client_kwargs must not override {key!r}.")
        kwargs.update(config.client_kwargs)
        self.client = genai.Client(**kwargs)

    def generate_json(
        self,
        prompt: str,
        *,
        schema: dict[str, Any] | None = None,
        schema_name: str = "malchan_response",
        strict: bool | None = None,
    ) -> LLMResponse:
        del schema_name, strict  # Gemini validates the supplied schema directly.
        config_payload: dict[str, Any] = {
            "response_mime_type": "application/json",
            "max_output_tokens": self.config.max_output_tokens,
        }
        if self.config.temperature is not None:
            config_payload["temperature"] = self.config.temperature
        if schema is not None:
            config_payload["response_schema"] = schema
        config_payload = _merge_request_kwargs(
            config_payload,
            self.config.request_kwargs,
            reserved={
                "response_mime_type",
                "response_schema",
                "max_output_tokens",
                "temperature",
            },
        )
        response = self.client.models.generate_content(
            model=self.config.model,
            contents=prompt,
            config=config_payload,
        )
        text = getattr(response, "text", "") or ""
        usage = getattr(response, "usage_metadata", None)
        return LLMResponse(
            text=text,
            raw=response,
            provider="gemini",
            model=self.config.model,
            usage=_usage_to_dict(usage),
            request_id=_request_id(response),
        )


def make_llm_client(
    config: LLMConfig | Mapping[str, Any] | None,
    *,
    client: Any | None = None,
) -> BaseLLMClient:
    """Construct a provider adapter from a dataclass or mapping."""

    resolved = coerce_llm_config(config)
    if resolved is None:
        raise ValueError("llm_config is required when planner_response is not supplied.")
    if resolved.normalized_provider == "openai":
        return OpenAIClient(resolved, client=client)
    if resolved.normalized_provider == "gemini":
        return GeminiClient(resolved, client=client)
    raise ValueError(f"Unknown LLM provider: {resolved.provider!r}.")
