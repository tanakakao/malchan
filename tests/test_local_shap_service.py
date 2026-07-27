from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


class FakePreprocess:
    """Minimal fitted preprocessor used by the local-SHAP service test."""

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        return frame[["x"]].to_numpy(dtype=float)


class FakeChildModel:
    """Target model exposing the fitted pipeline contract used by SHAP."""

    task = "regression"
    all_cols = ["x"]
    target_items = None

    def __init__(self) -> None:
        self.model = {
            "preprocess": FakePreprocess(),
            "predictor": object(),
        }
        self.df_prerpocessed = pd.DataFrame({"x": [0.0, 1.0]})
        self.shap_values = np.asarray([[999.0]])
        self.base_values = np.asarray([999.0])
        self.X_sample = pd.DataFrame({"x": [999.0]})


class FakeService:
    """Registered-model service double."""

    def __init__(self) -> None:
        self.child = FakeChildModel()
        self.registered = SimpleNamespace(
            model=self.child,
            info=SimpleNamespace(
                target_cols=["property"],
                feature_columns=["x"],
            ),
        )

    def _get_registered(self, model_id: str):
        assert model_id == "model-1"
        return self.registered


def test_local_shap_uses_training_background_and_only_evaluates_submitted_rows(
    monkeypatch,
) -> None:
    """Selected rows are explained against training data without replacing XAI caches."""

    from malchan.app.schemas import LocalShapRequest
    from malchan.app.services.xai_shap_service import compute_local_shap

    explainability = pytest.importorskip("malchan.models.explainability")
    captured_background = []
    captured_evaluation = []

    class FakeExplainer:
        def __call__(self, frame, **kwargs):
            captured_evaluation.append(frame.copy())
            return SimpleNamespace(
                values=frame.to_numpy(dtype=float) * 0.5,
                base_values=np.zeros(len(frame), dtype=float),
            )

    def fake_get_shap_values(predictor, frame):
        captured_background.append(frame.copy())
        return (
            np.zeros((len(frame), frame.shape[1])),
            np.zeros(len(frame), dtype=float),
            FakeExplainer(),
            frame.copy(),
        )

    monkeypatch.setattr(explainability, "get_shap_values", fake_get_shap_values)
    service = FakeService()
    response = compute_local_shap(
        service,
        "model-1",
        LocalShapRequest(data=[{"x": 2.0}, {"x": 4.0}]),
    )

    assert len(captured_background) == 1
    assert captured_background[0]["x"].tolist() == [0.0, 1.0]
    assert len(captured_evaluation) == 1
    assert captured_evaluation[0]["x"].tolist() == [2.0, 4.0]
    assert response.row_count == 2
    target = response.targets["property"]
    assert target.features == ["x"]
    assert target.shap_values["property"] == [[1.0], [2.0]]
    assert target.base_values["property"] == [0.0, 0.0]

    # Request-scoped explanations must not overwrite Explain-page caches.
    assert service.child.shap_values.tolist() == [[999.0]]
    assert service.child.base_values.tolist() == [999.0]
    assert service.child.X_sample["x"].tolist() == [999.0]


def test_local_shap_reuses_explainer_but_recalculates_values(monkeypatch) -> None:
    """Custom predictions may reuse the explainer while evaluating every new input."""

    from malchan.app.schemas import LocalShapRequest
    from malchan.app.services.xai_shap_service import compute_local_shap

    explainability = pytest.importorskip("malchan.models.explainability")
    build_count = 0
    evaluated = []

    class FakeExplainer:
        def __call__(self, frame, **kwargs):
            evaluated.append(frame["x"].tolist())
            return SimpleNamespace(
                values=frame.to_numpy(dtype=float),
                base_values=np.zeros(len(frame), dtype=float),
            )

    def fake_get_shap_values(predictor, frame):
        nonlocal build_count
        build_count += 1
        return np.zeros((len(frame), 1)), np.zeros(len(frame)), FakeExplainer(), frame

    monkeypatch.setattr(explainability, "get_shap_values", fake_get_shap_values)
    service = FakeService()
    compute_local_shap(service, "model-1", LocalShapRequest(data=[{"x": 2.0}]))
    compute_local_shap(service, "model-1", LocalShapRequest(data=[{"x": 5.0}]))

    assert build_count == 1
    assert evaluated == [[2.0], [5.0]]


def test_local_shap_rejects_missing_features() -> None:
    """Prediction-file rows must include every feature required by the model."""

    from malchan.app.schemas import LocalShapRequest
    from malchan.app.services.xai_shap_service import compute_local_shap

    with pytest.raises(ValueError, match="missing required columns"):
        compute_local_shap(
            FakeService(),
            "model-1",
            LocalShapRequest(data=[{"other": 1.0}]),
        )
