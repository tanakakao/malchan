import importlib.util

import pytest

from malchan.app.progress import (
    get_progress_snapshot,
    mark_target,
    progress_scope,
    report_dimension,
    set_target_plan,
)


def test_progress_store_tracks_target_trial_and_fold() -> None:
    progress_id = "progress-target-trial-fold-001"

    with progress_scope(progress_id, operation="POST /api/models"):
        set_target_plan(["strength", "cost", "density", "hardness"])
        mark_target("cost")
        report_dimension("trial", 18, 50, label="Optuna")
        report_dimension("fold", 3, 5, label="CV fold")

        snapshot = get_progress_snapshot(progress_id)
        assert snapshot is not None
        assert snapshot["status"] == "running"
        assert snapshot["dimensions"]["target"] == {
            "current": 2,
            "total": 4,
            "label": "cost",
            "detail": "",
        }
        assert snapshot["dimensions"]["trial"]["current"] == 18
        assert snapshot["dimensions"]["trial"]["total"] == 50
        assert snapshot["dimensions"]["fold"]["current"] == 3
        assert snapshot["dimensions"]["fold"]["total"] == 5

    completed = get_progress_snapshot(progress_id)
    assert completed is not None
    assert completed["status"] == "success"
    assert completed["completed_at"] is not None


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("fastapi") is None
    or importlib.util.find_spec("httpx") is None,
    reason="Progress API test requires the web and test extras.",
)


def test_progress_endpoint_retains_completed_request_snapshot() -> None:
    from fastapi.testclient import TestClient

    from malchan.app import create_app

    client = TestClient(create_app())
    progress_id = "progress-health-request-001"
    response = client.get(
        "/api/health",
        headers={"X-Malchan-Progress-ID": progress_id},
    )
    assert response.status_code == 200

    progress = client.get(f"/api/progress/{progress_id}")
    assert progress.status_code == 200
    payload = progress.json()
    assert payload["status"] == "success"
    assert payload["operation"] == "GET /api/health"
