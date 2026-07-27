import importlib.util

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("fastapi") is None
    or importlib.util.find_spec("httpx") is None,
    reason="XAI activation tests require the web and test extras.",
)


class ActivatedXaiBestModel:
    """Activated best model exposing CV diagnostics and XAI methods."""

    def __init__(self) -> None:
        self.target_col = "y"
        self.task = "regression"
        self.num_cols = ["x"]
        self.cat_cols = []
        self.smiles_cols = []
        self.comp_cols = []
        self.all_cols = ["x"]
        self.feature_names = ["x"]
        self.df_prerpocessed = pd.DataFrame({"x": [1.0, 2.0]})
        self.cv_scores = {
            "train": pd.DataFrame([{"RMSE": 0.2}]),
            "test": pd.DataFrame([{"RMSE": 0.5}]),
        }
        self.cv_preds = {
            "train": pd.DataFrame({"y": [1.9, 4.1]}, index=[0, 1]),
            "test": pd.DataFrame({"y": [2.1, 3.9]}, index=[0, 1]),
        }
        self.shap_calls = 0
        self.get_xai_calls = 0
        self.importances = None

    def _prepare_cv_y(self, y=None):
        return pd.Series([2.0, 4.0], name="y")

    def predict(self, X, proba=False, idx2item=False):
        return pd.DataFrame({"y": [99.0] * len(X)})

    def shap(self) -> None:
        """Create raw SHAP state for Explain visualizations."""

        self.shap_calls += 1
        self.shap_values = np.array([[0.1], [0.2]])
        self.base_values = np.array([2.0, 2.0])
        self.X_sample = self.df_prerpocessed.copy()

    def get_xai(self) -> None:
        """Create importance, SHAP scatter, and PDP caches."""

        self.get_xai_calls += 1
        values = np.array([1.0])
        self.importances = {
            "model": values,
            "pfi": values,
            "shap": values,
            "model_combine": values,
            "pfi_combine": values,
            "shap_combine": values,
            "shap_pd": {
                "x": pd.DataFrame({"x": [1.0, 2.0], "shap": [0.1, 0.2]})
            },
            "pd": {
                "x": (
                    np.array([[1.0, 2.0], [2.0, 3.0]]),
                    np.array([1.0, 2.0]),
                )
            },
        }


class ActivatedXaiComparisonResult:
    """Comparison result selecting the XAI-capable best model."""

    def __init__(self, best_model) -> None:
        self.metric = "RMSE"
        self.higher_is_better = False
        self.ranking = pd.DataFrame(
            [
                {
                    "rank": 1,
                    "model_name": "best-model",
                    "target": "y",
                    "task": "regression",
                    "test_RMSE": 0.5,
                }
            ]
        )
        self.failures = {}
        self.best_model_name = "best-model"
        self.best_model = best_model
        self.best_params = {"depth": 4}
        self.best_is_tuned = False
        self.method = "kfold"
        self.n_splits = 2

    @property
    def best_cv_scores(self):
        return self.best_model.cv_scores


class XaiDisabledComparisonSeed:
    """Seed model trained without XAI and replaced after comparison."""

    latest_best = None

    def fit(self, **kwargs) -> None:
        self.fit_kwargs = kwargs

    def predict(self, X, proba=False, idx2item=False):
        return pd.DataFrame({"y": [1.0] * len(X)})

    def compare(self, **kwargs):
        best = ActivatedXaiBestModel()
        type(self).latest_best = best
        return ActivatedXaiComparisonResult(best)


def test_activating_best_model_forces_xai_after_xai_disabled_seed() -> None:
    """Best activation should prepare Explain caches even when seed XAI was skipped."""

    from fastapi.testclient import TestClient

    from malchan.app import create_app
    from malchan.app.services import InMemoryModelService

    service = InMemoryModelService(
        model_factory=XaiDisabledComparisonSeed,
        id_factory=lambda: "forced-xai-model",
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
    compared = client.post(
        "/api/models/forced-xai-model/compare",
        json={
            "model_names": ["best-model"],
            "n_splits": 2,
            "activate_best": True,
        },
    )
    summary = client.get("/api/models/forced-xai-model/xai")
    metadata = client.get("/api/models/forced-xai-model")

    assert trained.status_code == 201
    assert trained.json()["xai_status"] == "not_requested"
    assert compared.status_code == 200

    best = XaiDisabledComparisonSeed.latest_best
    assert best is not None
    assert best.shap_calls == 1
    assert best.get_xai_calls == 1

    assert metadata.status_code == 200
    assert metadata.json()["model_names"] == ["best-model"]
    assert metadata.json()["xai_status"] == "ready"

    assert summary.status_code == 200
    target = summary.json()["targets"]["y"]
    assert summary.json()["status"] == "ready"
    assert target["status"] == "ready"
    assert set(target["importance_methods"]) == {"model", "pfi", "shap"}
    assert target["shap_features"] == ["x"]
    assert target["pdp_features"] == ["x"]
