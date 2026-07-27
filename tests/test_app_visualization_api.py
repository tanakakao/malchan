import importlib.util
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("fastapi") is None
    or importlib.util.find_spec("httpx") is None
    or importlib.util.find_spec("plotly") is None,
    reason="Visualization API tests require web, test, and visualization extras.",
)


class VisualizationService:
    """Small service double exposing only operations used by visualization routes."""

    def __init__(self) -> None:
        self.registered = SimpleNamespace(model=object())

    def _get_registered(self, model_id: str):
        from malchan.app.services import ModelNotFoundError

        if model_id != "model-1":
            raise ModelNotFoundError(model_id)
        return self.registered

    def get_xai_importance(self, model_id, target, **kwargs):
        from malchan.app.schemas import XaiImportanceItem, XaiImportanceResponse

        self._get_registered(model_id)
        return XaiImportanceResponse(
            model_id=model_id,
            target=target,
            method=kwargs["method"],
            combined=kwargs["combined"],
            items=[XaiImportanceItem(feature="x", value=1.5)],
        )

    def get_xai_shap_values(self, model_id, target):
        from malchan.app.schemas import XaiShapValuesResponse

        self._get_registered(model_id)
        return XaiShapValuesResponse(
            model_id=model_id,
            target=target,
            features=["x"],
            output_names=["shap"],
            records=[{"x": 1.0}],
            shap_values={"shap": [[0.5]]},
        )


def _client():
    from fastapi.testclient import TestClient

    from malchan.app import create_app

    return TestClient(create_app(model_service=VisualizationService()))


def test_visualization_yy_endpoint_serializes_plotly_figure(monkeypatch) -> None:
    """The API should return the figure produced by malchan.visualization."""

    import plotly.graph_objects as go
    import malchan.visualization as visualization

    monkeypatch.setattr(
        visualization,
        "show_model_diagnostics",
        lambda model, target, **kwargs: go.Figure(
            data=[go.Scatter(x=[1.0], y=[2.0])],
            layout={"title": f"diagnostic:{target}"},
        ),
    )

    response = _client().get("/api/models/model-1/visualizations/y/yy")

    assert response.status_code == 200
    payload = response.json()
    assert payload["figure"]["data"][0]["type"] == "scatter"
    assert payload["figure"]["layout"]["title"]["text"] == "diagnostic:y"
    assert payload["metadata"]["target"] == "y"


def test_visualization_importance_endpoint_uses_xai_adapter(monkeypatch) -> None:
    """Importance responses should be passed to the visualization adapter."""

    import plotly.graph_objects as go
    import malchan.visualization as visualization

    captured = {}

    def show(response, n_bar):
        captured["response"] = response
        captured["n_bar"] = n_bar
        return go.Figure(data=[go.Bar(x=[1.5], y=["x"])])

    monkeypatch.setattr(visualization, "show_xai_importance", show)

    response = _client().get(
        "/api/models/model-1/visualizations/y/importance",
        params={"method": "shap", "top_n": 7},
    )

    assert response.status_code == 200
    assert captured["response"].target == "y"
    assert captured["n_bar"] == 7
    assert response.json()["figure"]["data"][0]["type"] == "bar"


def test_visualization_pdp_endpoint_uses_existing_plot(monkeypatch) -> None:
    """The one-dimensional PD endpoint should call show_model_pd_and_ice."""

    import plotly.graph_objects as go
    import malchan.visualization as visualization

    captured = {}
    monkeypatch.setattr(
        visualization,
        "visualization_outputs",
        lambda model, target: [target],
    )

    def show(model, target, feature_name, **kwargs):
        captured.update(
            {
                "model": model,
                "target": target,
                "feature_name": feature_name,
                **kwargs,
            }
        )
        return go.Figure(layout={"title": "existing-pd"})

    monkeypatch.setattr(visualization, "show_model_pd_and_ice", show)
    response = _client().get(
        "/api/models/model-1/visualizations/y/pdp",
        params={"feature": "x1", "include_ice": "true"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["figure"]["layout"]["title"]["text"] == "existing-pd"
    assert payload["metadata"]["visualization_function"] == "show_pd_and_ice"
    assert captured["target"] == "y"
    assert captured["feature_name"] == "x1"
    assert captured["ice"] is True


def test_visualization_pdp_2d_endpoint_uses_existing_plot(monkeypatch) -> None:
    """The two-dimensional PD endpoint should call show_model_pd_2d."""

    import plotly.graph_objects as go
    import malchan.visualization as visualization

    captured = {}
    monkeypatch.setattr(
        visualization,
        "visualization_outputs",
        lambda model, target: [target],
    )

    def show(model, target, feature_names, **kwargs):
        captured.update(
            {
                "model": model,
                "target": target,
                "feature_names": feature_names,
                **kwargs,
            }
        )
        return go.Figure(layout={"title": "existing-pd-2d"})

    monkeypatch.setattr(visualization, "show_model_pd_2d", show)
    response = _client().get(
        "/api/models/model-1/visualizations/y/pdp-2d",
        params={"feature_x": "x1", "feature_y": "x2"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["figure"]["layout"]["title"]["text"] == "existing-pd-2d"
    assert payload["metadata"]["visualization_function"] == "show_pd_2d"
    assert captured["feature_names"] == ["x1", "x2"]


def test_visualization_returns_404_for_unknown_model() -> None:
    """Unknown model identifiers should preserve the model API's 404 behavior."""

    response = _client().get("/api/models/missing/visualizations/y/yy")

    assert response.status_code == 404
    assert response.json()["detail"] == "Model not found."
