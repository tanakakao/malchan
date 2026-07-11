import importlib.util

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("fastapi") is None
    or importlib.util.find_spec("httpx") is None,
    reason="Web UI tests require the web and test extras.",
)


def test_create_app_serves_built_react_frontend_without_shadowing_api(tmp_path) -> None:
    """The root static mount should serve React while preserving API routes."""

    from fastapi.testclient import TestClient

    from malchan.app import AppSettings, create_app

    (tmp_path / "index.html").write_text(
        "<!doctype html><title>malchan React</title><div id='root'></div>",
        encoding="utf-8",
    )
    (tmp_path / "asset.txt").write_text("asset", encoding="utf-8")
    app = create_app(
        settings=AppSettings(
            frontend_dist=str(tmp_path),
            cors_origins=(),
        )
    )
    client = TestClient(app)

    assert client.get("/").status_code == 200
    assert "malchan React" in client.get("/").text
    assert client.get("/asset.txt").text == "asset"
    assert client.get("/api/health").json()["status"] == "ok"
    assert app.state.frontend_dist == tmp_path.resolve()


def test_create_app_can_disable_frontend_mount() -> None:
    """API-only deployments should be able to disable the React mount."""

    from fastapi.testclient import TestClient

    from malchan.app import AppSettings, create_app

    app = create_app(
        settings=AppSettings(
            serve_frontend=False,
            cors_origins=(),
        )
    )
    client = TestClient(app)

    assert client.get("/api/health").status_code == 200
    assert client.get("/").status_code == 404
    assert app.state.frontend_dist is None
