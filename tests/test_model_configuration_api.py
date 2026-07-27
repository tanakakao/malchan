import importlib.util
import sys
from types import ModuleType

import pandas as pd
import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("fastapi") is None
    or importlib.util.find_spec("httpx") is None,
    reason="FastAPI model-configuration tests require the web and test extras.",
)


class FloatDistribution:
    """Small Optuna-compatible float distribution used by the API test."""

    def __init__(self, low, high, *, step=None, log=False):
        self.low = low
        self.high = high
        self.step = step
        self.log = log


class CategoricalDistribution:
    """Small Optuna-compatible categorical distribution used by the API test."""

    def __init__(self, choices):
        self.choices = choices


class FakeEvaluationPipeline:
    """Pipeline double recording CV settings without comparing model families."""

    def __init__(self) -> None:
        self.cv_scores = None
        self.cv_calls = []

    def fit(self, **kwargs) -> None:
        """Accept application training arguments."""

        self.fit_kwargs = kwargs

    def cv_score(self, method="kfold", n_splits=5, X=None, y=None) -> None:
        """Expose deterministic train and validation metrics."""

        self.cv_calls.append({"method": method, "n_splits": n_splits})
        self.cv_scores = {
            "train": pd.DataFrame([{"RMSE": 0.4, "R2": 0.95}]),
            "test": pd.DataFrame([{"RMSE": 0.6, "R2": 0.90}]),
        }


class FakeMultiEvaluationPipeline:
    """Unused multi-output factory required by the application service."""

    def fit(self, **kwargs) -> None:
        self.fit_kwargs = kwargs


def _make_client(model_id="evaluation-model"):
    """Create a FastAPI client with a CV-capable model double."""

    from fastapi.testclient import TestClient

    from malchan.app import create_app
    from malchan.app.services import InMemoryModelService

    service = InMemoryModelService(
        model_factory=FakeEvaluationPipeline,
        multi_model_factory=FakeMultiEvaluationPipeline,
        id_factory=lambda: model_id,
    )
    return TestClient(create_app(model_service=service))


def _train_payload() -> dict:
    """Return a minimal single-output training request."""

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
        "model_names": ["Ridge"],
    }


def test_model_parameter_endpoint_converts_tuning_space_to_controls(monkeypatch) -> None:
    """The Web API should expose ranges and choices without predictor prefixes."""

    fake_utils = ModuleType("malchan.models.utils")
    fake_utils.reg_default_params = {"Ridge": {"alpha": 1.0, "solver": "auto"}}
    fake_utils.cls_default_params = {}
    fake_utils.get_param_grid_reg = lambda model_name: {
        "predictor__alpha": FloatDistribution(1e-3, 1e3, log=True),
        "predictor__solver": CategoricalDistribution(["auto", "svd"]),
    }
    fake_utils.get_param_grid_cls = lambda model_name: None
    monkeypatch.setitem(sys.modules, "malchan.models.utils", fake_utils)

    client = _make_client()
    response = client.get(
        "/api/model-parameters",
        params={"task": "regression", "model_name": "Ridge"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["task"] == "regression"
    assert payload["model_name"] == "Ridge"
    parameters = {item["name"]: item for item in payload["parameters"]}
    assert parameters["alpha"] == {
        "name": "alpha",
        "label": "alpha",
        "control": "float",
        "default_value": 1.0,
        "low": 0.001,
        "high": 1000.0,
        "step": None,
        "log": True,
        "choices": [],
        "editable": True,
        "note": None,
    }
    assert parameters["solver"]["control"] == "categorical"
    assert parameters["solver"]["choices"] == ["auto", "svd"]
    assert parameters["solver"]["default_value"] == "auto"


def test_evaluate_endpoint_scores_registered_model_without_comparison() -> None:
    """CV evaluation should return metrics for the selected registered model."""

    client = _make_client()
    trained = client.post("/api/models", json=_train_payload())
    assert trained.status_code == 201

    response = client.post(
        "/api/models/evaluation-model/evaluate",
        json={"method": "kfold", "n_splits": 3},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["model_id"] == "evaluation-model"
    assert payload["method"] == "kfold"
    assert payload["n_splits"] == 3
    assert payload["targets"]["y"] == {
        "target": "y",
        "task": "regression",
        "train": [{"RMSE": 0.4, "R2": 0.95}],
        "test": [{"RMSE": 0.6, "R2": 0.9}],
    }
