"""FastAPI integration tests for compositional log-ratio preprocessing."""

import importlib.util

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("fastapi") is None
    or importlib.util.find_spec("httpx") is None,
    reason="FastAPI API tests require the web and test extras.",
)


class RecordingPipeline:
    """Capture compositional context received during single-output fitting."""

    def __init__(self) -> None:
        self.settings = None
        self.fit_kwargs = None

    def fit(self, **kwargs) -> None:
        from malchan.pipeline.compositional_extensions import (
            current_compositional_training_settings,
        )

        self.fit_kwargs = kwargs
        self.settings = current_compositional_training_settings()


class RecordingMultiPipeline(RecordingPipeline):
    """Capture compositional context received during multi-output fitting."""


def _records() -> list[dict]:
    return [
        {"a": 0.2, "b": 0.3, "c": 0.5, "temperature": 100.0, "strength": 1.0, "cost": 4.0},
        {"a": 0.4, "b": 0.2, "c": 0.4, "temperature": 200.0, "strength": 2.0, "cost": 3.0},
        {"a": 0.3, "b": 0.4, "c": 0.3, "temperature": 300.0, "strength": 3.0, "cost": 2.0},
    ]


def _single_payload() -> dict:
    return {
        "data": _records(),
        "target_col": "strength",
        "task": "regression",
        "num_cols": ["a", "b", "c", "temperature"],
        "cat_cols": [],
        "model_names": ["Ridge"],
        "compositional_groups": [["a", "b", "c"]],
        "compositional_method": "ilr",
        "compositional_zero_replacement": 1e-5,
        "compositional_closure": True,
        "compositional_scale_type": "StandardScaler",
        "compute_xai": False,
    }


def _make_client(
    pipeline: RecordingPipeline | None = None,
    multi_pipeline: RecordingMultiPipeline | None = None,
):
    from fastapi.testclient import TestClient

    from malchan.app import create_app
    from malchan.app.services import InMemoryModelService

    single = pipeline or RecordingPipeline()
    multi = multi_pipeline or RecordingMultiPipeline()
    service = InMemoryModelService(
        model_factory=lambda: single,
        multi_model_factory=lambda: multi,
        id_factory=lambda: "composition-model",
    )
    return TestClient(create_app(model_service=service)), single, multi


def test_fastapi_forwards_compositional_settings_to_single_output_training() -> None:
    client, pipeline, _ = _make_client()

    response = client.post("/api/models", json=_single_payload())

    assert response.status_code == 201
    assert pipeline.settings.groups == (("a", "b", "c"),)
    assert pipeline.settings.method == "ILR"
    assert pipeline.settings.zero_replacement == pytest.approx(1e-5)
    assert pipeline.settings.closure is True
    assert pipeline.settings.scale_type == "StandardScaler"
    assert response.json()["feature_columns"] == ["a", "b", "c", "temperature"]


def test_fastapi_forwards_compositional_settings_to_multi_output_training() -> None:
    pipeline = RecordingMultiPipeline()
    client, _, _ = _make_client(multi_pipeline=pipeline)
    payload = _single_payload()
    payload.pop("target_col")
    payload.pop("task")
    payload.pop("model_names")
    payload["target_cols"] = ["strength", "cost"]
    payload["tasks"] = ["regression", "regression"]
    payload["model_names_by_target"] = {
        "strength": ["Ridge"],
        "cost": ["Ridge"],
    }
    payload["compositional_method"] = "clr"

    response = client.post("/api/models", json=payload)

    assert response.status_code == 201
    assert pipeline.settings.groups == (("a", "b", "c"),)
    assert pipeline.settings.method == "CLR"


@pytest.mark.parametrize(
    ("patch", "expected_message"),
    [
        ({"compositional_groups": [["a"]]}, "at least two columns"),
        (
            {"compositional_groups": [["a", "b"], ["b", "c"]]},
            "cannot belong to multiple compositional groups",
        ),
        (
            {"compositional_groups": [["a", "unknown"]]},
            "must use columns listed in num_cols",
        ),
        (
            {"compositional_groups": [["a", "b"]], "compositional_method": None},
            "compositional_method is required",
        ),
        (
            {"compositional_zero_replacement": 1.0},
            "less than 1",
        ),
        (
            {
                "compositional_groups": [["a", "b"]],
                "compositional_method": "ALR",
                "compositional_alr_reference": 3,
            },
            "out of range",
        ),
    ],
)
def test_fastapi_rejects_invalid_compositional_settings(
    patch: dict,
    expected_message: str,
) -> None:
    client, pipeline, _ = _make_client()
    payload = _single_payload()
    payload.update(patch)

    response = client.post("/api/models", json=payload)

    assert response.status_code == 422
    assert expected_message in response.text
    assert pipeline.fit_kwargs is None