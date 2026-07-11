import importlib.util
from types import SimpleNamespace

import pandas as pd
import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("fastapi") is None
    or importlib.util.find_spec("httpx") is None,
    reason="FastAPI activation tests require the web and test extras.",
)


class ActivatedBestModel:
    """Best-model placeholder used after comparison activation."""

    def predict(self, X, proba=False, idx2item=False):
        """Return a value distinct from the original registered model."""

        return pd.DataFrame({"y": [99.0] * len(X)})


class ActivatingComparisonResult:
    """Minimal comparison result exposing an activatable best model."""

    def __init__(self) -> None:
        """Create one successful candidate result."""

        self.metric = "RMSE"
        self.higher_is_better = False
        self.ranking = pd.DataFrame(
            [
                {
                    "rank": 1,
                    "model_name": "activated-model",
                    "target": "y",
                    "task": "regression",
                    "test_RMSE": 0.5,
                }
            ]
        )
        self.failures = {}
        self.best_model_name = "activated-model"
        self.best_model = ActivatedBestModel()
        self.best_params = {"depth": 4}
        self.best_is_tuned = True
        self.best_cv_scores = {
            "train": pd.DataFrame([{"RMSE": 0.2}]),
            "test": pd.DataFrame([{"RMSE": 0.5}]),
        }


class OriginalPipeline:
    """Original registered model returning a different prediction."""

    def fit(self, **kwargs) -> None:
        """Accept application training arguments."""

        self.fit_kwargs = kwargs

    def predict(self, X, proba=False, idx2item=False):
        """Return the original model prediction."""

        return pd.DataFrame({"y": [1.0] * len(X)})

    def compare(self, **kwargs):
        """Return an activatable best-model result."""

        return ActivatingComparisonResult()


def test_compare_can_activate_best_model_for_later_prediction() -> None:
    """activate_best should promote the selected model in the registry."""

    from fastapi.testclient import TestClient

    from malchan.app import create_app
    from malchan.app.services import InMemoryModelService

    service = InMemoryModelService(
        model_factory=OriginalPipeline,
        id_factory=lambda: "activation-model",
    )
    client = TestClient(create_app(model_service=service))
    client.post(
        "/api/models",
        json={
            "data": [
                {"x": 1.0, "y": 2.0},
                {"x": 2.0, "y": 4.0},
            ],
            "target_col": "y",
            "task": "regression",
            "num_cols": ["x"],
            "cat_cols": [],
            "model_names": ["seed-model"],
        },
    )

    comparison = client.post(
        "/api/models/activation-model/compare",
        json={
            "model_names": ["activated-model"],
            "n_splits": 2,
            "activate_best": True,
        },
    )
    prediction = client.post(
        "/api/models/activation-model/predict",
        json={"data": [{"x": 3.0}]},
    )
    metadata = client.get("/api/models/activation-model")

    assert comparison.status_code == 200
    assert prediction.json()["predictions"] == [{"y": 99.0}]
    assert metadata.json()["model_names"] == ["activated-model"]
    assert service.get_comparison("activation-model").targets["y"].best_is_tuned
