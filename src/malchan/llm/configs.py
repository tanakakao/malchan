"""Configuration objects for provider-independent LLM planning."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

LLMProvider = Literal["openai", "gemini"] | str
RawDataPolicy = Literal["summary_only", "allow_samples"]

DEFAULT_API_KEY_ENV: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
}

_SECRET_FRAGMENTS = ("api_key", "authorization", "password", "secret", "token")


def _redact_secrets(value: Any, *, key: str | None = None) -> Any:
    """Return a recursively redacted copy suitable for logs and API responses."""

    if key is not None and any(fragment in key.lower() for fragment in _SECRET_FRAGMENTS):
        return "***" if value not in (None, "") else value
    if isinstance(value, Mapping):
        return {
            str(item_key): _redact_secrets(item, key=str(item_key))
            for item_key, item in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_redact_secrets(item) for item in value]
    return value


@dataclass(slots=True)
class LLMConfig:
    """Provider-neutral settings used to construct one LLM client.

    Args:
        model: Provider-side model name. The caller must select this explicitly
            so malchan does not silently depend on a time-sensitive default.
        provider: ``"openai"`` or ``"gemini"``.
        api_key_env: Environment variable containing the API key. Provider
            defaults are used when omitted.
        api_key: Direct API key. Environment variables are preferred because
            direct values are easier to leak into notebooks or serialized data.
        temperature: Sampling temperature. ``None`` omits the provider option.
        max_output_tokens: Maximum generated tokens.
        timeout: Request timeout in seconds. ``None`` uses the SDK default.
        max_retries: Provider-client retry count.
        ssl_verify: Whether HTTPS certificate verification is enabled.
        ca_bundle_path: PEM CA bundle used for corporate certificate chains.
        proxy: Optional explicit proxy URL. Environment proxy variables remain
            supported by both SDKs when this is omitted.
        base_url: Optional provider endpoint override.
        api_version: Optional Gemini API version.
        strict_schema: Whether provider-side strict JSON Schema enforcement is
            requested when supported.
        store: Whether OpenAI Responses API application state may be stored.
        client_kwargs: Extra provider-client constructor arguments.
        request_kwargs: Extra generation request arguments.
    """

    model: str
    provider: LLMProvider = "openai"
    api_key_env: str | None = None
    api_key: str | None = field(default=None, repr=False)
    temperature: float | None = 0.1
    max_output_tokens: int = 4096
    timeout: float | None = 60.0
    max_retries: int = 2
    ssl_verify: bool = True
    ca_bundle_path: str | None = None
    proxy: str | None = field(default=None, repr=False)
    base_url: str | None = None
    api_version: str | None = None
    strict_schema: bool = False
    store: bool = False
    client_kwargs: dict[str, Any] = field(default_factory=dict)
    request_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        provider = self.normalized_provider
        if provider not in DEFAULT_API_KEY_ENV:
            raise ValueError(f"Unknown LLM provider: {self.provider!r}.")
        if not str(self.model).strip():
            raise ValueError("LLMConfig.model must not be empty.")
        if self.temperature is not None and self.temperature < 0:
            raise ValueError("temperature must be non-negative or None.")
        if self.max_output_tokens < 1:
            raise ValueError("max_output_tokens must be at least 1.")
        if self.timeout is not None and self.timeout <= 0:
            raise ValueError("timeout must be positive or None.")
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative.")

    @property
    def normalized_provider(self) -> str:
        """Return a lowercase provider identifier."""

        return str(self.provider).strip().lower()

    def resolved_api_key_env(self) -> str:
        """Return the configured or provider-default API key variable name."""

        return self.api_key_env or DEFAULT_API_KEY_ENV[self.normalized_provider]

    def safe_dict(self) -> dict[str, Any]:
        """Return a secret-redacted dictionary for logs and persisted metadata."""

        data = asdict(self)
        if data.get("proxy"):
            data["proxy"] = "***"
        return _redact_secrets(data)


@dataclass(slots=True)
class LLMContextConfig:
    """Domain and operational context supplied to the configuration planner."""

    feature_descriptions: Mapping[str, str] = field(default_factory=dict)
    target_descriptions: Mapping[str, str] = field(default_factory=dict)
    domain_notes: Sequence[str] = field(default_factory=list)
    priorities: Mapping[str, float] = field(default_factory=dict)
    constraints: Sequence[str] = field(default_factory=list)
    max_training_seconds: float | None = None
    raw_data_policy: RawDataPolicy = "summary_only"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.max_training_seconds is not None and self.max_training_seconds <= 0:
            raise ValueError("max_training_seconds must be positive or None.")
        if self.raw_data_policy not in {"summary_only", "allow_samples"}:
            raise ValueError(f"Unknown raw_data_policy: {self.raw_data_policy!r}.")
        invalid_priorities = {
            name: value
            for name, value in self.priorities.items()
            if not isinstance(value, (int, float)) or not 0.0 <= float(value) <= 1.0
        }
        if invalid_priorities:
            raise ValueError(
                "priority values must be numeric values between 0 and 1: "
                f"{invalid_priorities}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-friendly planner context."""

        return {
            "feature_descriptions": dict(self.feature_descriptions),
            "target_descriptions": dict(self.target_descriptions),
            "domain_notes": list(self.domain_notes),
            "priorities": {name: float(value) for name, value in self.priorities.items()},
            "constraints": list(self.constraints),
            "max_training_seconds": self.max_training_seconds,
            "raw_data_policy": self.raw_data_policy,
            "metadata": _redact_secrets(dict(self.metadata)),
        }


def coerce_llm_config(value: LLMConfig | Mapping[str, Any] | None) -> LLMConfig | None:
    """Normalize a mapping, dataclass, or ``None`` to :class:`LLMConfig`."""

    if value is None or isinstance(value, LLMConfig):
        return value
    return LLMConfig(**dict(value))


def coerce_llm_context(
    value: LLMContextConfig | Mapping[str, Any] | None,
) -> LLMContextConfig:
    """Normalize optional planner context."""

    if value is None:
        return LLMContextConfig()
    if isinstance(value, LLMContextConfig):
        return value
    return LLMContextConfig(**dict(value))
