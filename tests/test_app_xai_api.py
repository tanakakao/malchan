import importlib.util

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("fastapi") is None
    or importlib.util.find_spec("httpx") is None,
    reason="Cached XAI API tests require the web and test extras.",
)


class FakeXaiPipeline:
    """Small fitted pipeline exposing deterministic XAI caches."""

    def __init__(self) -> None:
        """Initialize call counters and empty model metadata."""

        self.shap_calls = 0
        self.get_xai_calls = 0
        self.X = None
        self.y = None
        self.num_cols = []
        self.cat_cols = []
        self.smiles_cols = []
        self.comp_cols = []
        self.all_cols = []
        self.feature_names = []
        self.df_prerpocessed = None
        self.target_col = None
        self.task = None
        self.target_items = None
        self.importances = None

    def fit(self, **kwargs) -> None:
        """Store training data and metadata used by cached serializers."""

        self.target_col = kwargs["target_col"]
        self.task = kwargs["task"]
        self.num_cols = list(kwargs["num_cols"])
        self.cat_cols = list(kwargs["cat_cols"])
        self.smiles_cols = list(kwargs["smiles_cols"])
        self.comp_cols = list(kwargs["comp_cols"])
        self.all_cols = [
            *self.num_cols,
            *self.cat_cols,
            *self.smiles_cols,
            *self.comp_cols,
        ]
        self.X = kwargs["df"][self.all_cols].copy()
        self.y = kwargs["df"][[self.target_col]].copy()
        self.feature_names = list(self.all_cols)
        self.df_prerpocessed = self.X.copy()
        if self.task == "classification":
            self.target_items = np.array(["NG", "OK"], dtype=object)

    def shap(self) -> None:
        """Record one expensive SHAP calculation."""

        self.shap_calls += 1

    def get_xai(self) -> None:
        """Build deterministic importance, SHAP, and PDP caches."""

        self.get_xai_calls += 1
        shap_frame = pd.DataFrame(
            {
                "x1": [1.0, 2.0],
                "shap": [0.1, 0.4],
            }
        )
        if self.task == "classification":
            pd_values = np.array(
                [
                    [[0.8, 0.2], [0.7, 0.3]],
                    [[0.5, 0.5], [0.4, 0.6]],
                    [[0.2, 0.8], [0.1, 0.9]],
                ]
            )
        else:
            pd_values = np.array(
                [
                    [1.0, 3.0],
                    [2.0, 4.0],
                    [3.0, 5.0],
                ]
            )
        self.importances = {
            "model": np.array([0.25, 0.75]),
            "pfi": np.array([0.4, 0.6]),
            "shap": np.array([0.3, 0.7]),
            "model_combine": np.array([0.25, 0.75]),
            "pfi_combine": np.array([0.4, 0.6]),
            "shap_combine": np.array([0.3, 0.7]),
            "shap_pd": {
                "x1": shap_frame,
                "x2": pd.DataFrame(
                    {"x2": [2.0, 4.0], "shap": [-0.2, 0.3]}
                ),
            },
            "pd": {
                "x1": (pd_values, np.array([0.0, 1.0, 2.0])),
                "x2": (pd_values, np.array([0.0, 1.0, 2.0])),
            },
        }

    def predict(self, X, proba=False, idx2item=False):
        """Return simple predictions for compatibility with the model service."""

        if self.task == "classification" and proba:
            return pd.DataFrame(
                {
                    f"{self.target_col}_NG": [0.25] * len(X),
                    f"{self.target_col}_OK": [0.75] * len(X),
                }
            )
        return pd.DataFrame({self.target_col: [1.0] * len(X)})


def _payload(*, task="regression", compute_xai=True) -> dict:
    """Return a valid XAI-enabled training request."""

    targets = [2.0, 4.0] if task == "regression" else ["NG", "OK"]
    return {
        "data": [
            {"x1": 1.0, "x2": 2.0, "y": targets[0]},
            {"x1": 2.0, "x2": 4.0, "y": targets[1]},
        ],
        "target_col": "y",
        "task": task,
        "num_cols": ["x1", "x2"],
        "cat_cols": [],
        "model_names": ["fake"],
        "compute_xai": compute_xai,
    }


def _make_client(pipeline: FakeXaiPipeline, model_id: str = "xai-model"):
    """Create a FastAPI client using one deterministic pipeline instance."""

    from fastapi.testclient import TestClient

    from malchan.app import create_app
    from malchan.app.services import InMemoryModelService

    service = InMemoryModelService(
        model_factory=lambda: pipeline,
        id_factory=lambda: model_id,
    )
    return TestClient(create_app(model_service=service)), service


def test_training_precomputes_xai_once_and_get_endpoints_use_cache() -> None:
    """Normal XAI reads must never rerun SHAP or get_xai."""

    pipeline = FakeXaiPipeline()
    client, _ = _make_client(pipeline)

    trained = client.post("/api/models", json=_payload())
    summary = client.get("/api/models/xai-model/xai")
    summary_again = client.get("/api/models/xai-model/xai")
    importance = client.get(
        "/api/models/xai-model/xai/y/importance",
        params={"method": "shap", "combined": True, "top_n": 2},
    )
    shap = client.get(
        "/api/models/xai-model/xai/y/shap",
        params={"feature": "x1"},
    )
    pdp = client.get(
        "/api/models/xai-model/xai/y/pdp",
        params={"feature": "x1", "include_ice": True, "max_ice": 1},
    )

    assert trained.status_code == 201
    assert trained.json()["xai_status"] == "ready"
    assert summary.status_code == 200
    assert summary.json()["status"] == "ready"
    assert summary.json()["targets"]["y"]["shap_features"] == ["x1", "x2"]
    assert summary_again.json() == summary.json()
    assert importance.json()["items"] == [
        {"feature": "x2", "value": 0.7},
        {"feature": "x1", "value": 0.3},
    ]
    assert shap.json()["value_columns"] == ["shap"]
    assert shap.json()["records"][1] == {"x1": 2.0, "shap": 0.4}
    assert pdp.json()["x_values"] == [0.0, 1.0, 2.0]
    assert pdp.json()["series"][0]["pd_values"] == [2.0, 3.0, 4.0]
    assert pdp.json()["series"][0]["ice_values"] == [[1.0, 2.0, 3.0]]
    assert pipeline.shap_calls == 1
    assert pipeline.get_xai_calls == 1


def test_recompute_endpoint_is_the_only_normal_way_to_rerun_xai() -> None:
    """An explicit recompute request should execute the expensive workflow again."""

    pipeline = FakeXaiPipeline()
    client, _ = _make_client(pipeline)
    client.post("/api/models", json=_payload())

    response = client.post(
        "/api/models/xai-model/xai/recompute",
        json={"targets": ["y"]},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "ready"
    assert pipeline.shap_calls == 2
    assert pipeline.get_xai_calls == 2


def test_xai_can_be_disabled_then_computed_explicitly() -> None:
    """compute_xai=false should skip work until the recompute endpoint is called."""

    pipeline = FakeXaiPipeline()
    client, _ = _make_client(pipeline)

    trained = client.post("/api/models", json=_payload(compute_xai=False))
    unavailable = client.get("/api/models/xai-model/xai/y/importance")
    recomputed = client.post(
        "/api/models/xai-model/xai/recompute",
        json={},
    )

    assert trained.json()["xai_status"] == "not_requested"
    assert unavailable.status_code == 409
    assert recomputed.status_code == 200
    assert recomputed.json()["status"] == "ready"
    assert pipeline.shap_calls == 1
    assert pipeline.get_xai_calls == 1


def test_classification_pdp_uses_numpy_target_items_without_truth_value_error() -> None:
    """Class labels stored as a NumPy array should serialize safely."""

    pipeline = FakeXaiPipeline()
    client, _ = _make_client(pipeline, model_id="classification-model")
    client.post(
        "/api/models",
        json=_payload(task="classification"),
    )

    response = client.get(
        "/api/models/classification-model/xai/y/pdp",
        params={"feature": "x1"},
    )

    assert response.status_code == 200
    assert [series["name"] for series in response.json()["series"]] == ["NG", "OK"]
    assert response.json()["series"][0]["pd_values"] == [0.75, 0.45, 0.15]
    assert response.json()["series"][1]["pd_values"] == [0.25, 0.55, 0.85]
