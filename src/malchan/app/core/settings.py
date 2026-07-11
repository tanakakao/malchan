"""Settings used by the FastAPI and web application layers."""

import os
from dataclasses import dataclass
from functools import lru_cache

DEFAULT_APP_NAME = "malchan"
DEFAULT_API_PREFIX = "/api"
DEFAULT_CORS_ORIGINS = (
    "http://127.0.0.1:5173",
    "http://localhost:5173",
)


def _env_flag(name: str, default: bool) -> bool:
    """Read one conventional boolean environment variable."""

    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True, slots=True)
class AppSettings:
    """Runtime settings for the malchan application layer.

    Attributes:
        app_name: Human-readable application name.
        api_prefix: URL prefix for API routes.
        debug: Whether development debug mode is enabled.
        cors_origins: Allowed browser origins for the Vite development server.
        serve_frontend: Whether to serve a built React application when found.
        frontend_dist: Optional explicit path to the React build directory.
    """

    app_name: str = DEFAULT_APP_NAME
    api_prefix: str = DEFAULT_API_PREFIX
    debug: bool = False
    cors_origins: tuple[str, ...] = DEFAULT_CORS_ORIGINS
    serve_frontend: bool = True
    frontend_dist: str | None = None


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """Load application settings from environment variables."""

    origins_value = os.getenv("MALCHAN_CORS_ORIGINS")
    origins = (
        tuple(item.strip() for item in origins_value.split(",") if item.strip())
        if origins_value is not None
        else DEFAULT_CORS_ORIGINS
    )
    return AppSettings(
        app_name=os.getenv("MALCHAN_APP_NAME", DEFAULT_APP_NAME),
        api_prefix=os.getenv("MALCHAN_API_PREFIX", DEFAULT_API_PREFIX),
        debug=_env_flag("MALCHAN_DEBUG", False),
        cors_origins=origins,
        serve_frontend=_env_flag("MALCHAN_SERVE_FRONTEND", True),
        frontend_dist=os.getenv("MALCHAN_FRONTEND_DIST") or None,
    )
