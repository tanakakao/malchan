import importlib.util
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("fastapi") is None
    or importlib.util.find_spec("httpx") is None,
    reason="XAI activation tests require the web and test extras.",
)


class ActivatableXaiPipeline:
    """Pipeline double that returns a new best model during comparison."""

    def __init__(self) -> None:
        """Initialize XAI counters and model metadata."""

        self.shap_calls = 0
        self.get_xai_calls = 0
        self.fit_kwargs = None
        self.best_candidate = None
        self.target_col = None
        self.task = None
        self.num_cols = []
        self.cat_cols = []
        self.smiles_cols = []
        self.comp_cols = []
        self.all_cols = []
        self.feature_names = []
        self.df_prerpocessed = None
        self.importances = None

    def fit(self, **kwargs) -> None:
        """Record training metadata used to construct the compared candidate."""

        self.fit_kwargs = kwargs
        self.target_col = kwargs["target_col"]
        self.task = kwargs["task"]
        self.num_cols = list(kwargs["num_cols"])
        self.cat_cols = list(kwargs["cat_cols"])
        self.smiles_cols = list(kwargs["smiles_cols"])
        self.comp_cols = list(kwargs["comp_cols"])
        self.all_cols = [*self.num_cols, *self.cat_cols]
        self.feature_names = list(self.all_cols)
        self.df_prerpocessed = kwargs["df"][self.all_cols].copy()

    def shap(self) -> None:
        """Record an expensive SHAP calculation."""

        self.shap_calls += 1

    def get_xai(self) -> None:
        """Build a compact cache for the active model."""

        self.get_xai_calls += 1
        self.importances = {
            "model": np.array([1.0]),
            "pfi": np.array([1.0]),
            "shap": np.array([1.0]),
            "model_combine": np.array([1.0]),
            "pfi_combine": np.array([1.0]),
            "shap_combine": np.array([1.0]),
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

    def compare(self, **kwargs):
        """Return a newly fitted best model for registry activation."""

        best = ActivatableXaiPipeline()
        best.fit(**self.fit_kwargs)
        self.best_candidate = best
        return SimpleNamespace(
            metric="RMSE",
            higher_is_better=False,
            ranking=pd.DataFrame(
                [
                    {
                        "rank": 1,
                        "model_name": "best-model",
                        "target": self.target_col,
                        "task": self.task,
                        "test_RMSE": 0.5,
                    }
                ]
            ),
            failures={},
            best_model_name="best-model",
            best_model=best,
            best_params={"depth": 4},
            best_is_tuned=False,
            best_cv_scores={
                "train": pd.DataFrame([{"RMSE": 0.2}]),
                "test": pd.DataFrame([{"RMSE": 0.5}]),
            },
        )


class SimpleMultiPipeline:
    """Multi-output double without XAI methods or child-model mapping."""

    def fit(self, **kwargs) -> None:
        """Accept multi-output training arguments."""

        self.target_cols = list(kwargs["target_cols"])
        self.fit_kwargs = kwargs


def _single_payload() -> dict:
    """Return a single-output training payload with XAI enabled."""

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
        "compute_xai": True,
    }


def _multi_payload() -> dict:
    """Return a multi-output request used for compatibility validation."""

    return {
        "data": [
            {"x": 1.0, "strength": 2.0, "cost": 3.0},
            {"x": 2.0, "strength": 4.0, "cost": 5.0},
        ],
        "target_cols": ["strength", "cost"],
        "tasks": ["regression", "regression"],
        "num_cols": ["x"],
        "cat_cols": [],
        "model_names": ["seed-model"],
        "compute_xai": True,
    }


def test_activating_best_model_recomputes_xai_for_new_model() -> None:
    """The active best model must not retain the original model's XAI cache."""

    from fastapi.testclient import TestClient

    from malchan.app import create_app
    from malchan.app.services import InMemoryModelService

    original = ActivatableXaiPipeline()
    service = InMemoryModelService(
        model_factory=lambda: original,
        id_factory=lambda: "activation-xai-model",
    )
    client = TestClient(create_app(model_service=service))

    trained = client.post("/api/models", json=_single_payload())
    compared = client.post(
        "/api/models/activation-xai-model/compare",
        json={
            "model_names": ["best-model"],
            "n_splits": 2,
            "activate_best": True,
        },
    )
    summary = client.get("/api/models/activation-xai-model/xai")

    assert trained.status_code == 201
    assert compared.status_code == 200
    assert original.shap_calls == 1
    assert original.get_xai_calls == 1
    assert original.best_candidate is not None
    assert original.best_candidate.shap_calls == 1
    assert original.best_candidate.get_xai_calls == 1
    assert summary.json()["status"] == "ready"


def test_xai_enabled_training_keeps_simple_multi_output_models_usable() -> None:
    """XAI-unaware multi-output doubles should be registered as unavailable."""

    from malchan.app.schemas import TrainModelRequest
    from malchan.app.services import InMemoryModelService

    service = InMemoryModelService(
        multi_model_factory=SimpleMultiPipeline,
        id_factory=lambda: "simple-multi",
    )

    info = service.train(TrainModelRequest.model_validate(_multi_payload()))
    summary = service.get_xai_summary("simple-multi")

    assert info.xai_status == "unavailable"
    assert summary.status == "unavailable"
    assert set(summary.targets) == {"strength", "cost"}
    assert all(item.status == "unavailable" for item in summary.targets.values())
