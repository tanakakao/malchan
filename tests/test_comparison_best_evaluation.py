import importlib.util

import pandas as pd
import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("fastapi") is None
    or importlib.util.find_spec("httpx") is None,
    reason="FastAPI comparison evaluation tests require the web and test extras.",
)


class EvaluatedBestModel:
    """Best-model double retaining CV metrics and plot-ready predictions."""

    def __init__(self) -> None:
        self.target_col = "y"
        self.task = "regression"
        self.cv_scores = None
        self.cv_preds = None
        self.cv_calls = []

    def _prepare_cv_y(self, y=None):
        """Return actual values aligned to positional CV prediction indices."""

        return pd.Series([2.0, 4.0], name="y")

    def cv_score(self, method="kfold", n_splits=5, X=None, y=None) -> None:
        """Record evaluation settings and create Train/Test diagnostics."""

        self.cv_calls.append({"method": method, "n_splits": n_splits})
        self.cv_scores = {
            "train": pd.DataFrame([{"RMSE": 0.2, "R2": 0.98}]),
            "test": pd.DataFrame([{"RMSE": 0.5, "R2": 0.91}]),
        }
        self.cv_preds = {
            "train": pd.DataFrame({"y": [1.9, 4.1]}, index=[0, 1]),
            "test": pd.DataFrame({"y": [2.1, 3.9]}, index=[0, 1]),
        }


class EvaluationComparisonResult:
    """Comparison result exposing one selected best model."""

    def __init__(self) -> None:
        self.metric = "RMSE"
        self.higher_is_better = False
        self.ranking = pd.DataFrame(
            [
                {
                    "rank": 1,
                    "model_name": "best-model",
                    "target": "y",
                    "task": "regression",
                    "train_RMSE": 0.2,
                    "test_RMSE": 0.5,
                }
            ]
        )
        self.failures = {}
        self.best_model_name = "best-model"
        self.best_model = EvaluatedBestModel()
        self.best_params = {"depth": 4}
        self.best_is_tuned = False
        self.method = "kfold"
        self.n_splits = 3

    @property
    def best_cv_scores(self):
        """Return the selected model's latest CV metrics."""

        return self.best_model.cv_scores


class ComparisonEvaluationPipeline:
    """Registered seed model returning the evaluation comparison result."""

    latest_result = None

    def fit(self, **kwargs) -> None:
        self.fit_kwargs = kwargs

    def predict(self, X, proba=False, idx2item=False):
        return pd.DataFrame({"y": [0.0] * len(X)})

    def compare(self, **kwargs):
        result = EvaluationComparisonResult()
        result.method = kwargs["method"]
        result.n_splits = kwargs["n_splits"]
        type(self).latest_result = result
        return result


def test_comparison_returns_best_model_scores_and_plot_data() -> None:
    """Comparison should evaluate the best model and retain Train/Test records."""

    from fastapi.testclient import TestClient

    from malchan.app import create_app
    from malchan.app.services import InMemoryModelService

    service = InMemoryModelService(
        model_factory=ComparisonEvaluationPipeline,
        id_factory=lambda: "comparison-evaluation-model",
    )
    client = TestClient(create_app(model_service=service))
    trained = client.post(
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
            "compute_xai": False,
        },
    )
    response = client.post(
        "/api/models/comparison-evaluation-model/compare",
        json={
            "model_names": ["best-model"],
            "method": "kfold",
            "n_splits": 3,
        },
    )

    assert trained.status_code == 201
    assert response.status_code == 200
    result = ComparisonEvaluationPipeline.latest_result
    assert result is not None
    assert result.best_model.cv_calls == [{"method": "kfold", "n_splits": 3}]

    target = response.json()["targets"]["y"]
    assert target["best_cv_scores"] == {
        "train": [{"RMSE": 0.2, "R2": 0.98}],
        "test": [{"RMSE": 0.5, "R2": 0.91}],
    }
    assert target["best_cv_predictions"] == {
        "train": [
            {"index": 0, "actual": 2.0, "predicted": 1.9},
            {"index": 1, "actual": 4.0, "predicted": 4.1},
        ],
        "test": [
            {"index": 0, "actual": 2.0, "predicted": 2.1},
            {"index": 1, "actual": 4.0, "predicted": 3.9},
        ],
    }
