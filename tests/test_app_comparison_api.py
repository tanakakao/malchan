import importlib.util
from types import SimpleNamespace

import pandas as pd
import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("fastapi") is None
    or importlib.util.find_spec("httpx") is None,
    reason="FastAPI comparison tests require the web and test extras.",
)


class FakeTargetComparison:
    """Small comparison result supporting deferred best-model tuning."""

    def __init__(self, target: str, best_name: str) -> None:
        """Initialize an untuned comparison result."""

        self.target = target
        self.metric = "RMSE"
        self.higher_is_better = False
        self.ranking = pd.DataFrame(
            [
                {
                    "rank": 1,
                    "model_name": best_name,
                    "target": target,
                    "task": "regression",
                    "test_RMSE": 1.0,
                },
                {
                    "rank": 2,
                    "model_name": "other-model",
                    "target": target,
                    "task": "regression",
                    "test_RMSE": 2.0,
                },
            ]
        )
        self.failures = {}
        self.best_model_name = best_name
        self.best_params = None
        self.best_is_tuned = False
        self.best_cv_scores = None
        self.tuning_calls = []

    def tune_best(self, *, n_trials=30, verbose=0, evaluate=True):
        """Record tuning settings and expose tuned response metadata."""

        self.tuning_calls.append(
            {
                "n_trials": n_trials,
                "verbose": verbose,
                "evaluate": evaluate,
            }
        )
        self.best_is_tuned = True
        self.best_params = {"optimized": True, "n_trials": n_trials}
        self.best_cv_scores = (
            {
                "train": pd.DataFrame([{"RMSE": 0.4}]),
                "test": pd.DataFrame([{"RMSE": 0.6}]),
            }
            if evaluate
            else None
        )
        return SimpleNamespace(target=self.target, n_trials=n_trials)


class FakeMultiComparison:
    """Multi-output comparison result delegating tuning per target."""

    def __init__(self) -> None:
        """Create one result for each objective."""

        self.results = {
            "strength": FakeTargetComparison("strength", "strength-model"),
            "cost": FakeTargetComparison("cost", "cost-model"),
        }

    def tune_best(
        self,
        *,
        targets=None,
        n_trials=30,
        verbose=0,
        evaluate=True,
    ):
        """Tune selected targets with shared or target-specific trial counts."""

        selected = list(self.results) if targets is None else list(targets)
        tuned = {}
        for target in selected:
            target_trials = (
                n_trials.get(target, 30)
                if isinstance(n_trials, dict)
                else n_trials
            )
            tuned[target] = self.results[target].tune_best(
                n_trials=target_trials,
                verbose=verbose,
                evaluate=evaluate,
            )
        return tuned


class FakeComparisonPipeline:
    """Single-output model double exposing the model-native compare method."""

    def __init__(self) -> None:
        """Initialize recorded settings."""

        self.compare_kwargs = None
        self.comparison_result = None

    def fit(self, **kwargs) -> None:
        """Accept training arguments used by the application service."""

        self.fit_kwargs = kwargs

    def predict(self, X, proba=False, idx2item=False):
        """Return deterministic predictions."""

        return pd.DataFrame({"y": [float(value) * 2 for value in X["x"]]})

    def compare(self, **kwargs):
        """Return a comparison and optionally tune its selected best model."""

        self.compare_kwargs = kwargs
        result = FakeTargetComparison("y", "best-model")
        if kwargs["tune_best"]:
            result.tune_best(
                n_trials=kwargs["tuning_trials"],
                verbose=kwargs["tuning_verbose"],
                evaluate=True,
            )
        self.comparison_result = result
        return result


class FakeMultiComparisonPipeline:
    """Multi-output model double exposing comparison and target tuning."""

    def __init__(self) -> None:
        """Initialize recorded settings."""

        self.compare_kwargs = None
        self.comparison_result = None

    def fit(self, **kwargs) -> None:
        """Accept multi-output training arguments."""

        self.fit_kwargs = kwargs

    def predict(self, X, proba=False, idx2item=False):
        """Return deterministic multi-output predictions."""

        return pd.DataFrame(
            {
                "strength": [float(value) * 2 for value in X["x"]],
                "cost": [float(value) + 1 for value in X["x"]],
            }
        )

    def compare(self, **kwargs):
        """Return one comparison result per output target."""

        self.compare_kwargs = kwargs
        result = FakeMultiComparison()
        if kwargs["tune_best"]:
            result.tune_best(
                n_trials=kwargs["tuning_trials"],
                verbose=kwargs["tuning_verbose"],
                evaluate=True,
            )
        self.comparison_result = result
        return result


def _single_train_payload() -> dict:
    """Return a minimal single-output API training request."""

    return {
        "data": [
            {"x": 1.0, "y": 2.0},
            {"x": 2.0, "y": 4.0},
        ],
        "target_col": "y",
        "task": "regression",
        "num_cols": ["x"],
        "cat_cols": [],
        "model_names": ["seed-model"],
    }


def _multi_train_payload() -> dict:
    """Return a minimal multi-output API training request."""

    return {
        "data": [
            {"x": 1.0, "strength": 2.0, "cost": 2.0},
            {"x": 2.0, "strength": 4.0, "cost": 3.0},
        ],
        "target_cols": ["strength", "cost"],
        "tasks": ["regression", "regression"],
        "num_cols": ["x"],
        "cat_cols": [],
        "model_names": ["seed-model"],
    }


def _make_client(
    model_factory=FakeComparisonPipeline,
    multi_model_factory=FakeMultiComparisonPipeline,
    model_id="comparison-model",
):
    """Create a FastAPI test client with comparison-capable pipelines."""

    from fastapi.testclient import TestClient

    from malchan.app import create_app
    from malchan.app.services import InMemoryModelService

    service = InMemoryModelService(
        model_factory=model_factory,
        multi_model_factory=multi_model_factory,
        id_factory=lambda: model_id,
    )
    return TestClient(create_app(model_service=service))


def test_compare_endpoint_can_tune_best_automatically() -> None:
    """POST /compare should expose the two-stage tuning workflow."""

    client = _make_client()
    client.post("/api/models", json=_single_train_payload())

    response = client.post(
        "/api/models/comparison-model/compare",
        json={
            "model_names": ["model-a", "model-b"],
            "n_splits": 2,
            "tune_best": True,
            "tuning_trials": 17,
        },
    )

    assert response.status_code == 200
    target = response.json()["targets"]["y"]
    assert target["best_model_name"] == "best-model"
    assert target["best_is_tuned"] is True
    assert target["best_params"] == {"optimized": True, "n_trials": 17}
    assert target["best_cv_scores"]["test"] == [{"RMSE": 0.6}]
    assert target["ranking"][0]["test_RMSE"] == 1.0


def test_comparison_can_be_tuned_later_and_retrieved() -> None:
    """A later tune-best request should update the stored comparison state."""

    client = _make_client()
    client.post("/api/models", json=_single_train_payload())
    compared = client.post(
        "/api/models/comparison-model/compare",
        json={"model_names": ["model-a", "model-b"], "n_splits": 2},
    )
    tuned = client.post(
        "/api/models/comparison-model/comparison/tune-best",
        json={"n_trials": 23, "evaluate": True},
    )
    retrieved = client.get("/api/models/comparison-model/comparison")

    assert compared.json()["targets"]["y"]["best_is_tuned"] is False
    assert tuned.status_code == 200
    assert tuned.json()["targets"]["y"]["best_params"] == {
        "optimized": True,
        "n_trials": 23,
    }
    assert retrieved.json() == tuned.json()


def test_tune_best_requires_existing_comparison() -> None:
    """Tuning before comparison should return an explicit conflict response."""

    client = _make_client()
    client.post("/api/models", json=_single_train_payload())

    response = client.post(
        "/api/models/comparison-model/comparison/tune-best",
        json={"n_trials": 10},
    )

    assert response.status_code == 409
    assert "Run model comparison" in response.text


def test_multi_output_comparison_supports_per_target_deferred_tuning() -> None:
    """The API should tune only selected outputs with per-target trials."""

    client = _make_client(model_id="multi-model")
    client.post("/api/models", json=_multi_train_payload())
    client.post(
        "/api/models/multi-model/compare",
        json={
            "model_names": {
                "strength": ["model-a", "model-b"],
                "cost": ["model-c", "model-d"],
            },
            "metric": {"strength": "R2", "cost": "RMSE"},
            "n_splits": 2,
        },
    )

    response = client.post(
        "/api/models/multi-model/comparison/tune-best",
        json={
            "targets": ["strength"],
            "n_trials": {"strength": 80},
            "evaluate": False,
        },
    )

    assert response.status_code == 200
    targets = response.json()["targets"]
    assert targets["strength"]["best_is_tuned"] is True
    assert targets["strength"]["best_params"]["n_trials"] == 80
    assert targets["strength"]["best_cv_scores"] is None
    assert targets["cost"]["best_is_tuned"] is False
