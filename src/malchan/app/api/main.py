"""FastAPI application factory for malchan."""

from typing import TYPE_CHECKING, Any

from malchan import __version__
from malchan.app.core import AppSettings, get_settings

if TYPE_CHECKING:
    from fastapi import FastAPI


def create_app(
    settings: AppSettings | None = None,
    model_service: Any | None = None,
    *,
    title: str | None = None,
    version: str | None = None,
) -> "FastAPI":
    """Create and configure the FastAPI and optional React application.

    The factory can be imported from :mod:`malchan.app` and used by Uvicorn,
    Python scripts, tests, or an in-process ``TestClient`` in Jupyter.

    Args:
        settings: Runtime settings controlling the API prefix, CORS, debug mode,
            optional React frontend mounting, and signed model bundles.
        model_service: Optional model service implementation. A process-local
            :class:`~malchan.app.services.InMemoryModelService` is created when
            omitted.
        title: Optional OpenAPI application title. This overrides
            ``settings.app_name`` without mutating the settings object.
        version: Optional OpenAPI and health-endpoint version. The installed
            package version is used when omitted.

    Returns:
        Configured FastAPI application.

    Raises:
        RuntimeError: If the FastAPI optional dependencies are not installed.
    """

    try:
        from fastapi import FastAPI
        from fastapi.middleware.cors import CORSMiddleware
    except ImportError as exc:
        raise RuntimeError(
            "FastAPI support requires installing malchan with the 'api' or 'web' extra."
        ) from exc

    from malchan.app.api.routes import create_api_router
    from malchan.app.services import InMemoryModelService
    from malchan.app.web import mount_web_ui

    resolved_settings = settings or get_settings()
    resolved_service = model_service or InMemoryModelService()
    configure_model_bundles = getattr(resolved_service, "configure_model_bundles", None)
    if callable(configure_model_bundles):
        configure_model_bundles(
            resolved_settings.model_bundle_secret,
            resolved_settings.model_bundle_max_bytes,
        )
    resolved_title = title or resolved_settings.app_name
    resolved_version = version or __version__

    app = FastAPI(
        title=resolved_title,
        debug=resolved_settings.debug,
        version=resolved_version,
    )
    if resolved_settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(resolved_settings.cors_origins),
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    app.state.settings = resolved_settings
    app.state.model_service = resolved_service
    app.include_router(
        create_api_router(
            service=resolved_service,
            app_name=resolved_title,
            app_version=resolved_version,
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
