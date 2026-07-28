"""FastAPI ensemble-training request and service integration tests."""

import importlib.util

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("fastapi") is None
    or importlib.util.find_spec("httpx") is None,
    reason="FastAPI API tests require the web and test extras.",
)


class RecordingPipeline:
    """Record single-output fit arguments received from the model service."""

    def __init__(self) -> None:
        self.fit_kwargs = None

    def fit(self, **kwargs) -> None:
        self.fit_kwargs = kwargs


class RecordingMultiPipeline:
    """Record multi-output fit arguments received from the model service."""

    def __init__(self) -> None:
        self.fit_kwargs = None

    def fit(self, **kwargs) -> None:
        self.fit_kwargs = kwargs


def _single_payload() -> dict:
    """Return a minimal single-output regression request."""

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
        "model_names": ["model-a"],
        "compute_xai": False,
    }


def _multi_payload() -> dict:
    """Return a minimal two-target regression request."""

    return {
        "data": [
            {"x": 1.0, "strength": 2.0, "cost": 3.0},
            {"x": 2.0, "strength": 4.0, "cost": 4.0},
            {"x": 3.0, "strength": 6.0, "cost": 5.0},
        ],
        "target_cols": ["strength", "cost"],
        "tasks": ["regression", "regression"],
        "num_cols": ["x"],
        "cat_cols": [],
        "model_names_by_target": {
            "strength": ["model-a", "model-b"],
            "cost": ["model-c", "model-d"],
        },
        "ensemble": True,
        "ens_type": "アンサンブル",
        "model_params_by_target": {
            "strength": [{"alpha": 1.0}, {"alpha": 2.0}],
            "cost": [{"depth": 3}, {"depth": 4}],
        },
        "compute_xai": False,
    }


def _make_client(
    pipeline: RecordingPipeline | None = None,
    multi_pipeline: RecordingMultiPipeline | None = None,
):
    """Create a TestClient backed by recording pipeline doubles."""

    from fastapi.testclient import TestClient

    from malchan.app import create_app
    from malchan.app.services import InMemoryModelService

    single = pipeline or RecordingPipeline()
    multi = multi_pipeline or RecordingMultiPipeline()
    service = InMemoryModelService(
        model_factory=lambda: single,
        multi_model_factory=lambda: multi,
        id_factory=lambda: "ensemble-model",
    )
    return TestClient(create_app(model_service=service)), single, multi


def test_fastapi_forwards_voting_ensemble_settings() -> None:
    """Voting settings and per-member parameters should reach the pipeline."""

    client, pipeline, _ = _make_client()
    payload = _single_payload()
    payload.update(
        {
            "model_names": ["model-a", "model-b"],
            "ensemble": True,
            "ens_type": "アンサンブル",
            "model_params": [{"alpha": 1.0}, {"alpha": 2.0}],
        }
    )

    response = client.post("/api/models", json=payload)

    assert response.status_code == 201
    assert pipeline.fit_kwargs["ensemble"] is True
    assert pipeline.fit_kwargs["ens_type"] == "アンサンブル"
    assert pipeline.fit_kwargs["model_names"] == ["model-a", "model-b"]
    assert pipeline.fit_kwargs["model_params"] == [
        {"alpha": 1.0},
        {"alpha": 2.0},
    ]


def test_fastapi_normalizes_single_bagging_model_parameters() -> None:
    """A one-model parameter dictionary should become an aligned member list."""

    client, pipeline, _ = _make_client()
    payload = _single_payload()
    payload.update(
        {
            "ensemble": True,
            "ens_type": "バギング",
            "model_params": {"max_depth": 3},
        }
    )

    response = client.post("/api/models", json=payload)

    assert response.status_code == 201
    assert pipeline.fit_kwargs["ensemble"] is True
    assert pipeline.fit_kwargs["ens_type"] == "バギング"
    assert pipeline.fit_kwargs["model_params"] == [{"max_depth": 3}]


def test_fastapi_forwards_multi_output_ensemble_settings() -> None:
    """Multi-output training should preserve ensemble settings for every target."""

    pipeline = RecordingMultiPipeline()
    client, _, _ = _make_client(multi_pipeline=pipeline)

    response = client.post("/api/models", json=_multi_payload())

    assert response.status_code == 201
    assert pipeline.fit_kwargs["ensembles"] is True
    assert pipeline.fit_kwargs["ens_types"] == "アンサンブル"
    assert pipeline.fit_kwargs["model_names"] == [
        ["model-a", "model-b"],
        ["model-c", "model-d"],
    ]
    assert pipeline.fit_kwargs["model_params"] == [
        [{"alpha": 1.0}, {"alpha": 2.0}],
        [{"depth": 3}, {"depth": 4}],
    ]


@pytest.mark.parametrize(
    ("patch", "expected_message"),
    [
        ({"ensemble": True}, "ens_type is required"),
        (
            {
                "ensemble": True,
                "ens_type": "アンサンブル",
            },
            "requires at least two model names",
        ),
        (
            {
                "ensemble": True,
                "ens_type": "スタッキング",
                "model_names": ["model-a", "model-b"],
            },
            "base_model is required",
        ),
        (
            {
                "ensemble": True,
                "ens_type": "アンサンブル",
                "model_names": ["model-a", "model-b"],
                "model_params": {"alpha": 1.0},
            },
            "model_params must be a list aligned",
        ),
        (
            {
                "ensemble": True,
                "ens_type": "アンサンブル",
                "model_names": ["model-a", "model-b"],
                "model_params": [{"alpha": 1.0}],
            },
            "same length as model names",
        ),
    ],
)
def test_fastapi_rejects_invalid_ensemble_combinations(
    patch: dict,
    expected_message: str,
) -> None:
    """Invalid ensemble combinations should fail as request validation errors."""

    client, _, _ = _make_client()
    payload = _single_payload()
    payload.update(patch)

    response = client.post("/api/models", json=payload)

    assert response.status_code == 422
    assert expected_message in response.text


def test_fastapi_rejects_unknown_ensemble_type() -> None:
    """OpenAPI/Pydantic should restrict ensemble type to implemented values."""

    client, _, _ = _make_client()
    payload = _single_payload()
    payload.update(
        {
            "ensemble": True,
            "ens_type": "unknown",
            "model_names": ["model-a", "model-b"],
        }
    )

    response = client.post("/api/models", json=payload)

    assert response.status_code == 422
    assert "アンサンブル" in response.text
    assert "スタッキング" in response.text
    assert "バギング" in response.text
    assert "ブースティング" in response.text
