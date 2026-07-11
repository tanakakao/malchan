"""FastAPI application factory for malchan."""

from typing import Any

from malchan import __version__
from malchan.app.core import AppSettings, get_settings


def create_app(settings: AppSettings | None = None, model_service: Any | None = None) -> Any:
    """Create and configure the FastAPI and optional React application."""

    try:
        from fastapi import FastAPI
        from fastapi.middleware.cors import CORSMiddleware
    except ImportError as exc:
        raise RuntimeError(
            "FastAPI support requires installing malchan with the 'web' extra."
        ) from exc

    from malchan.app.api.routes import create_api_router
    from malchan.app.services import InMemoryModelService
    from malchan.app.web import mount_web_ui

    resolved_settings = settings or get_settings()
    resolved_service = model_service or InMemoryModelService()
    app = FastAPI(
        title=resolved_settings.app_name,
        debug=resolved_settings.debug,
        version=__version__,
    )
    if resolved_settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(resolved_settings.cors_origins),
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    app.state.model_service = resolved_service
    app.include_router(
        create_api_router(
            service=resolved_service,
            app_name=resolved_settings.app_name,
        ),
        prefix=resolved_settings.api_prefix,
    )
    if resolved_settings.serve_frontend:
        app.state.frontend_dist = mount_web_ui(
            app,
            configured_path=resolved_settings.frontend_dist,
        )
    else:
        app.state.frontend_dist = None
    return app
