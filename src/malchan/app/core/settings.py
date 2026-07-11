"""Settings used by the FastAPI and web application layers."""

import os
from dataclasses import dataclass
from functools import lru_cache

DEFAULT_APP_NAME = "malchan"
DEFAULT_API_PREFIX = "/api"


@dataclass(frozen=True, slots=True)
class AppSettings:
    """Runtime settings for the malchan application layer.

    Attributes:
        app_name: Human-readable application name.
        api_prefix: URL prefix for API routes.
        debug: Whether development debug mode is enabled.
    """

    app_name: str = DEFAULT_APP_NAME
    api_prefix: str = DEFAULT_API_PREFIX
    debug: bool = False


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """Load application settings from environment variables.

    Returns:
        Immutable application settings that can be reused by API and web code.
    """

    debug_value = os.getenv("MALCHAN_DEBUG", "").strip().lower()
    return AppSettings(
        app_name=os.getenv("MALCHAN_APP_NAME", DEFAULT_APP_NAME),
        api_prefix=os.getenv("MALCHAN_API_PREFIX", DEFAULT_API_PREFIX),
        debug=debug_value in {"1", "true", "yes", "on"},
    )
