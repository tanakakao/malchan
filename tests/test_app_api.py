import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

import pandas as pd
from fastapi.testclient import TestClient

from malchan.app import create_app
from malchan.app.schemas import PredictRequest, TrainModelRequest
from malchan.app.services import InMemoryModelService


class FakePipeline:
    """Small pipeline double used by API and service tests."""

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


def test_model_service_trains_and_predicts() -> None:
    """The service should connect API schemas to the model pipeline."""

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
    assert pipeline.fit_kwargs["target_col"] == "y"
    assert predictions == [{"y": 6.0}]


def test_create_app_exposes_model_lifecycle() -> None:
    """FastAPI should expose training, listing, inference, and deletion."""

    service = InMemoryModelService(
        model_factory=FakePipeline,
        id_factory=lambda: "model-1",
    )
    client = TestClient(create_app(model_service=service))

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


def test_train_request_rejects_missing_columns() -> None:
    """Request validation should report rows missing declared columns."""

    client = TestClient(
        create_app(
            model_service=InMemoryModelService(model_factory=FakePipeline)
        )
    )
    payload = _train_payload()
    payload["data"][1].pop("x")

    response = client.post("/api/models", json=payload)

    assert response.status_code == 422
    assert "missing required columns" in response.text


def test_unknown_model_is_404() -> None:
    """Inference should return 404 for unknown model identifiers."""

    client = TestClient(
        create_app(
            model_service=InMemoryModelService(model_factory=FakePipeline)
        )
    )

    response = client.post(
        "/api/models/missing/predict",
        json={"data": [{"x": 1.0}]},
    )

    assert response.status_code == 404
