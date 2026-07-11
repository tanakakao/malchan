import importlib.util

import pandas as pd
import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("fastapi") is None
    or importlib.util.find_spec("httpx") is None,
    reason="FastAPI API tests require the web and test extras.",
)


class FakePipeline:
    """Small single-output pipeline double used by API and service tests."""

    def __init__(self) -> None:
        """Initialize the recorded fit arguments."""

        self.fit_kwargs = None

    def fit(self, **kwargs) -> None:
        """Record arguments passed by the model service."""

        self.fit_kwargs = kwargs

    def predict(self, X, proba=False, idx2item=False):
        """Return deterministic predictions for the supplied rows."""

        assert list(X.columns) == ["x"]
        return pd.DataFrame({"y": [float(value) * 2 for value in X["x"]]})


class FakeMultiPipeline:
    """Small multi-output pipeline double used by API and service tests."""

    def __init__(self) -> None:
        """Initialize the recorded fit arguments."""

        self.fit_kwargs = None

    def fit(self, **kwargs) -> None:
        """Record arguments passed by the model service."""

        self.fit_kwargs = kwargs

    def predict(self, X, proba=False, idx2item=False):
        """Return deterministic predictions for both requested targets."""

        assert list(X.columns) == ["x"]
        return pd.DataFrame(
            {
                "strength": [float(value) * 2 for value in X["x"]],
                "cost": [float(value) + 1 for value in X["x"]],
            }
        )


def _train_payload() -> dict:
    """Return a valid single-output regression request."""

    return {
        "data": [
            {"x": 1.0, "y": 2.0},
            {"x": 2.0, "y": 4.0},
        ],
        "target_col": "y",
        "task": "regression",
        "num_cols": ["x"],
        "cat_cols": [],
        "model_names": ["dummy"],
    }


def _multi_train_payload() -> dict:
    """Return a valid two-objective regression request."""

    return {
        "data": [
            {"x": 1.0, "strength": 2.0, "cost": 2.0},
            {"x": 2.0, "strength": 4.0, "cost": 3.0},
        ],
        "target_cols": ["strength", "cost"],
        "tasks": ["regression", "regression"],
        "num_cols": ["x"],
        "cat_cols": [],
        "model_names": ["shared-model"],
        "model_names_by_target": {
            "strength": ["random-forest"],
            "cost": ["linear-model"],
        },
        "model_params_by_target": {
            "strength": {"n_estimators": 10},
        },
    }


def _make_client(
    model_factory=FakePipeline,
    multi_model_factory=None,
    id_factory=None,
):
    """Create a test client with an injected in-memory model service."""

    from fastapi.testclient import TestClient

    from malchan.app import create_app
    from malchan.app.services import InMemoryModelService

    service = InMemoryModelService(
        model_factory=model_factory,
        multi_model_factory=multi_model_factory,
        id_factory=id_factory,
    )
    return TestClient(create_app(model_service=service))


def test_model_service_trains_and_predicts() -> None:
    """The service should preserve the existing single-output workflow."""

    from malchan.app.schemas import PredictRequest, TrainModelRequest
    from malchan.app.services import InMemoryModelService

    pipeline = FakePipeline()
    service = InMemoryModelService(
        model_factory=lambda: pipeline,
        id_factory=lambda: "model-1",
    )

    info = service.train(TrainModelRequest.model_validate(_train_payload()))
    predictions = service.predict(
        "model-1",
        PredictRequest(data=[{"x": 3.0}]),
    )

    assert info.model_id == "model-1"
    assert info.target_col == "y"
    assert info.target_cols == ["y"]
    assert pipeline.fit_kwargs["target_col"] == "y"
    assert predictions == [{"y": 6.0}]


def test_model_service_trains_and_predicts_multiple_targets() -> None:
    """The service should route multi-target requests to MLModelPipeline."""

    from malchan.app.schemas import PredictRequest, TrainModelRequest
    from malchan.app.services import InMemoryModelService

    pipeline = FakeMultiPipeline()
    service = InMemoryModelService(
        model_factory=lambda: (_ for _ in ()).throw(
            AssertionError("Single-output factory was used.")
        ),
        multi_model_factory=lambda: pipeline,
        id_factory=lambda: "multi-model",
    )

    info = service.train(
        TrainModelRequest.model_validate(_multi_train_payload())
    )
    predictions = service.predict(
        "multi-model",
        PredictRequest(data=[{"x": 3.0}]),
    )

    assert info.target_col is None
    assert info.target_cols == ["strength", "cost"]
    assert info.tasks == ["regression", "regression"]
    assert info.model_names_by_target == {
        "strength": ["random-forest"],
        "cost": ["linear-model"],
    }
    assert pipeline.fit_kwargs["target_cols"] == ["strength", "cost"]
    assert pipeline.fit_kwargs["tasks"] == ["regression", "regression"]
    assert pipeline.fit_kwargs["model_names"] == [
        ["random-forest"],
        ["linear-model"],
    ]
    assert pipeline.fit_kwargs["model_params"] == [
        {"n_estimators": 10},
        None,
    ]
    assert predictions == [{"strength": 6.0, "cost": 4.0}]


def test_create_app_exposes_model_lifecycle() -> None:
    """FastAPI should expose training, listing, inference, and deletion."""

    client = _make_client(id_factory=lambda: "model-1")

    health = client.get("/api/health")
    trained = client.post("/api/models", json=_train_payload())
    predicted = client.post(
        "/api/models/model-1/predict",
        json={"data": [{"x": 3.0}]},
    )
    listed = client.get("/api/models")
    deleted = client.delete("/api/models/model-1")

    assert health.status_code == 200
    assert health.json()["version"] == "0.1.0"
    assert trained.status_code == 201
    assert predicted.json() == {
        "model_id": "model-1",
        "predictions": [{"y": 6.0}],
    }
    assert listed.json()["models"][0]["model_id"] == "model-1"
    assert deleted.status_code == 204
    assert client.get("/api/models/model-1").status_code == 404


def test_create_app_exposes_multi_output_prediction() -> None:
    """FastAPI should return all objective predictions in each record."""

    client = _make_client(
        multi_model_factory=FakeMultiPipeline,
        id_factory=lambda: "multi-model",
    )

    trained = client.post("/api/models", json=_multi_train_payload())
    predicted = client.post(
        "/api/models/multi-model/predict",
        json={"data": [{"x": 3.0}]},
    )

    assert trained.status_code == 201
    assert trained.json()["target_cols"] == ["strength", "cost"]
    assert predicted.json() == {
        "model_id": "multi-model",
        "predictions": [{"strength": 6.0, "cost": 4.0}],
    }


def test_train_request_rejects_missing_columns() -> None:
    """Request validation should report rows missing declared columns."""

    client = _make_client()
    payload = _train_payload()
    payload["data"][1].pop("x")

    response = client.post("/api/models", json=payload)

    assert response.status_code == 422
    assert "missing required columns" in response.text


def test_train_request_rejects_misaligned_multi_output_tasks() -> None:
    """Every multi-output target should have an aligned task."""

    client = _make_client(multi_model_factory=FakeMultiPipeline)
    payload = _multi_train_payload()
    payload["tasks"] = ["regression"]

    response = client.post("/api/models", json=payload)

    assert response.status_code == 422
    assert "same length" in response.text


def test_unknown_model_is_404() -> None:
    """Inference should return 404 for unknown model identifiers."""

    client = _make_client()

    response = client.post(
        "/api/models/missing/predict",
        json={"data": [{"x": 1.0}]},
    )

    assert response.status_code == 404
