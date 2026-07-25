"""Tests for creating the FastAPI application from Python and notebooks."""

from fastapi.testclient import TestClient

from malchan import __version__
from malchan.app import AppSettings, create_app


def test_create_app_supports_in_process_notebook_usage() -> None:
    """Create a customized app and call it without starting a network server."""

    settings = AppSettings(
        api_prefix="/notebook-api",
        cors_origins=(),
        serve_frontend=False,
    )
    app = create_app(
        settings=settings,
        title="malchan Notebook API",
        version="0.2.0-notebook",
    )

    with TestClient(app) as client:
        response = client.get("/notebook-api/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "malchan Notebook API",
        "version": "0.2.0-notebook",
    }
    assert app.title == "malchan Notebook API"
    assert app.version == "0.2.0-notebook"
    assert app.state.settings is settings
    assert app.state.frontend_dist is None


def test_create_app_uses_package_version_by_default() -> None:
    """Use the installed package version when no override is supplied."""

    app = create_app(
        settings=AppSettings(cors_origins=(), serve_frontend=False),
    )

    with TestClient(app) as client:
        response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json()["version"] == __version__
