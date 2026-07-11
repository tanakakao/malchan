"""FastAPI application factory for malchan."""

from typing import Any

from malchan.app.core import AppSettings, get_settings


def create_app(settings: AppSettings | None = None) -> Any:
    """Create and configure the FastAPI application.

    Args:
        settings: Optional settings object. When omitted, environment-backed
            settings are loaded with :func:`malchan.app.core.get_settings`.

    Returns:
        A configured :class:`fastapi.FastAPI` application instance.

    Raises:
        RuntimeError: If FastAPI is not installed in the current environment.
    """

    try:
        from fastapi import FastAPI
    except ImportError as exc:
        raise RuntimeError("FastAPI support requires installing malchan with the 'web' extra.") from exc

    resolved_settings = settings or get_settings()
    app = FastAPI(title=resolved_settings.app_name, debug=resolved_settings.debug)

    @app.get(f"{resolved_settings.api_prefix}/health", tags=["system"])
    def health_check() -> dict[str, str]:
        """Return a lightweight health-check response.

        Returns:
            Service status metadata for load balancers and smoke tests.
        """

        return {"status": "ok", "service": resolved_settings.app_name}

    return app
