"""Tests for trained-model diagrams and cached validation results."""

import importlib.util

import pandas as pd
import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("fastapi") is None
    or importlib.util.find_spec("httpx") is None,
    reason="Model visualization API tests require the web and test extras.",
)


class DiagramPipeline:
    """Small fitted sklearn pipeline exposed through the malchan wrapper shape."""

    def __init__(self) -> None:
        self.model = None
        self.cv_scores = None
        self.X = None
        self.num_cols = []
        self.cat_cols = []
        self.smiles_cols = []
        self.comp_cols = []
        self.target_col = None

    def fit(self, **kwargs) -> None:
        from sklearn.linear_model import LinearRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        frame = kwargs["df"]
        self.X = frame[kwargs["num_cols"]].copy()
        self.num_cols = list(kwargs["num_cols"])
        self.cat_cols = list(kwargs["cat_cols"])
        self.smiles_cols = list(kwargs["smiles_cols"])
        self.comp_cols = list(kwargs["comp_cols"])
        self.target_col = kwargs["target_col"]
        self.model = Pipeline(
            [
                ("scale", StandardScaler()),
                ("predictor", LinearRegression()),
            ]
        ).fit(self.X, frame[self.target_col])

    def cv_score(self, method: str, n_splits: int) -> None:
        del method, n_splits
        self.cv_scores = {
            "train": pd.DataFrame([{"R2": 0.98}, {"R2": 0.96}]),
            "test": pd.DataFrame([{"R2": 0.91}, {"R2": 0.89}]),
        }

    def predict(self, X, proba=False, idx2item=False):
        del proba, idx2item
        return pd.DataFrame({self.target_col: self.model.predict(X)})


def _payload() -> dict:
    return {
        "data": [
            {"x": 1.0, "y": 2.0},
            {"x": 2.0, "y": 4.0},
            {"x": 3.0, "y": 6.0},
        ],
        "target_col": "y",
        "task": "regression",
        "num_cols": ["x"],
        "cat_cols": [],
        "model_names": ["LinearRegression"],
    }


def test_model_visualization_returns_sklearn_pipeline_html_and_cached_scores() -> None:
    from fastapi.testclient import TestClient

    from malchan.app import create_app
    from malchan.app.services import InMemoryModelService

    service = InMemoryModelService(
        model_factory=DiagramPipeline,
        id_factory=lambda: "diagram-model",
    )
    client = TestClient(create_app(model_service=service))

    trained = client.post("/api/models", json=_payload())
    before = client.get("/api/models/diagram-model/visualization")
    evaluated = client.post(
        "/api/models/diagram-model/evaluate",
        json={"method": "kfold", "n_splits": 2},
    )
    after = client.get("/api/models/diagram-model/visualization")

    assert trained.status_code == 201
    assert before.status_code == 200
    assert before.json()["evaluation"] is None
    target = before.json()["targets"][0]
    assert target["target"] == "y"
    assert target["renderer"] == "sklearn"
    assert "Pipeline" in target["html"]
    assert "StandardScaler" in target["html"]
    assert "LinearRegression" in target["html"]

    assert evaluated.status_code == 200
    cached = after.json()["evaluation"]
    assert cached["model_id"] == "diagram-model"
    assert cached["targets"]["y"]["train"] == [{"R2": 0.98}, {"R2": 0.96}]
    assert cached["targets"]["y"]["test"] == [{"R2": 0.91}, {"R2": 0.89}]


def test_model_visualization_returns_404_for_unknown_model() -> None:
    from fastapi.testclient import TestClient

    from malchan.app import create_app
    from malchan.app.services import InMemoryModelService

    client = TestClient(create_app(model_service=InMemoryModelService()))

    response = client.get("/api/models/unknown/visualization")

    assert response.status_code == 404
