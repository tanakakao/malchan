"""FastAPI application factory for malchan."""

from typing import Any

from malchan import __version__
from malchan.app.core import AppSettings, get_settings


def create_app(settings: AppSettings | None = None, model_service: Any | None = None) -> Any:
    """Create and configure the FastAPI application.

    Args:
        settings: Optional settings object. Environment-backed settings are used
            when omitted.
        model_service: Optional service implementation for dependency injection.

    Returns:
        A configured :class:`fastapi.FastAPI` application instance.

    Raises:
        RuntimeError: If FastAPI is not installed in the current environment.
    """

    try:
        from fastapi import FastAPI
    except ImportError as exc:
        raise RuntimeError("FastAPI support requires installing malchan with the 'web' extra.") from exc

    from malchan.app.api.routes import create_api_router
    from malchan.app.services import InMemoryModelService

    resolved_settings = settings or get_settings()
    resolved_service = model_service or InMemoryModelService()
    app = FastAPI(
        title=resolved_settings.app_name,
        debug=resolved_settings.debug,
        version=__version__,
    )
    app.state.model_service = resolved_service
    app.include_router(
        create_api_router(
            service=resolved_service,
            app_name=resolved_settings.app_name,
        ),
        prefix=resolved_settings.api_prefix,
    )
    return app
