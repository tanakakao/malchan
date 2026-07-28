import importlib.util

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("plotly") is None,
    reason="Web visualization tests require the visualization extra.",
)


class FakeRegressionModel:
    """Minimal fitted single-output model for visualization tests."""

    target_col = "y"
    task = "regression"
    target_items = None

    def __init__(self) -> None:
        self.X = pd.DataFrame({"x1": [0.0, 1.0], "x2": [10.0, 20.0]})
        self.y = pd.DataFrame({"y": [1.0, 3.0]})
        self.cv_preds = {
            "train": pd.DataFrame({"y": [1.0, 3.0]}),
            "test": pd.DataFrame({"y": [1.2, 2.8]}),
        }

    def _get_X(self) -> pd.DataFrame:
        return self.X

    def _get_y(self) -> pd.DataFrame:
        return self.y

    def predict(self, X=None, proba=False, idx2item=False):
        data = self.X if X is None else X
        return pd.DataFrame({"y": data["x1"] * 2.0 + 1.0}, index=data.index)


class FakeClassificationModel(FakeRegressionModel):
    """Classification variant exposing selectable class outputs."""

    target_col = "label"
    task = "classification"
    target_items = np.asarray(["OK", "NG"])

    def __init__(self) -> None:
        super().__init__()
        self.y = pd.DataFrame({"label": [0, 1]})
        self.cv_preds = {
            "train": pd.DataFrame({"OK": [0.9, 0.2], "NG": [0.1, 0.8]}),
            "test": pd.DataFrame({"OK": [0.8, 0.3], "NG": [0.2, 0.7]}),
        }


class FakeModelWithoutCv(FakeRegressionModel):
    """Regression model that has not run cross validation."""

    def __init__(self) -> None:
        super().__init__()
        self.cv_preds = None


def test_show_model_diagnostics_supports_single_output_model() -> None:
    """The web adapter should reuse yy_plot_ml for a single-output pipeline."""

    from malchan.visualization import show_model_diagnostics

    figure = show_model_diagnostics(FakeRegressionModel(), "y")

    assert len(figure.data) == 2
    assert figure.data[0].type == "scatter"
    assert figure.layout.xaxis.title.text == "Actual Values"


def test_show_model_diagnostics_forwards_cv_residual_and_split(monkeypatch) -> None:
    """Web options should be forwarded directly to the existing yy_plot_ml function."""

    import plotly.graph_objects as go
    import malchan.visualization.web_api_plots as plots

    captured = {}

    def show(**kwargs):
        captured.update(kwargs)
        return go.Figure(layout={"title": "existing-yy"})

    monkeypatch.setattr(plots, "yy_plot_ml", show)
    figure = plots.show_model_diagnostics(
        FakeRegressionModel(),
        "y",
        cv=True,
        residual=True,
        train_test="train",
    )

    assert figure.layout.title.text == "existing-yy"
    assert captured["target"] == "y"
    assert captured["cv"] is True
    assert captured["residual"] is True
    assert captured["train_test"] == "train"


def test_show_model_diagnostics_rejects_cv_without_predictions() -> None:
    """CV plots should explain that cv_score must run before visualization."""

    from malchan.visualization import show_model_diagnostics

    with pytest.raises(ValueError, match="Run cv_score"):
        show_model_diagnostics(FakeModelWithoutCv(), "y", cv=True)


def test_show_model_diagnostics_rejects_classification_residual() -> None:
    """Residual mode is meaningful for regression targets only."""

    from malchan.visualization import show_model_diagnostics

    with pytest.raises(ValueError, match="regression targets only"):
        show_model_diagnostics(FakeClassificationModel(), "label", residual=True)


def test_show_model_shap_scatter_delegates_to_existing_visualization(monkeypatch) -> None:
    """The web adapter should pass feature, interaction, and class to show_shap_scatter."""

    import plotly.graph_objects as go
    import malchan.visualization.web_api_plots as plots

    captured = {}

    def show(**kwargs):
        captured.update(kwargs)
        return go.Figure(layout={"title": "existing-shap-scatter"})

    monkeypatch.setattr(plots, "show_shap_scatter", show)
    figure = plots.show_model_shap_scatter(
        FakeClassificationModel(),
        "label",
        "x1",
        interactive_col="x2",
        target_item="OK",
    )

    assert figure.layout.title.text == "existing-shap-scatter"
    assert captured["target"] == "label"
    assert captured["target_col"] == "x1"
    assert captured["interactive_col"] == "x2"
    assert captured["target_item"] == "OK"
    assert captured["model"].models["label"].target_col == "label"


def test_show_model_pd_and_ice_delegates_to_existing_visualization(monkeypatch) -> None:
    """The web adapter must not rebuild the one-dimensional PD chart."""

    import plotly.graph_objects as go
    import malchan.visualization.web_api_plots as plots

    captured = {}

    def show(**kwargs):
        captured.update(kwargs)
        return go.Figure(layout={"title": "existing-1d-pd"})

    monkeypatch.setattr(plots, "show_pd_and_ice", show)
    figure = plots.show_model_pd_and_ice(
        FakeRegressionModel(),
        "y",
        "x1",
        ice=False,
        output_index=0,
    )

    assert figure.layout.title.text == "existing-1d-pd"
    assert captured["target"] == "y"
    assert captured["target_col"] == "x1"
    assert captured["ice"] is False
    assert captured["col_idx"] == 0
    assert captured["model"].models["y"].target_col == "y"


def test_show_model_pd_2d_delegates_to_existing_visualization(monkeypatch) -> None:
    """The web adapter must not rebuild the two-dimensional PD chart."""

    import plotly.graph_objects as go
    import malchan.visualization.web_api_plots as plots

    captured = {}

    def show(**kwargs):
        captured.update(kwargs)
        return go.Figure(layout={"title": "existing-2d-pd"})

    monkeypatch.setattr(plots, "show_pd_2d", show)
    figure = plots.show_model_pd_2d(
        FakeRegressionModel(),
        "y",
        ["x1", "x2"],
        output_index=0,
    )

    assert figure.layout.title.text == "existing-2d-pd"
    assert captured["target"] == "y"
    assert captured["target_cols"] == ["x1", "x2"]
    assert captured["col_idx"] == 0
    assert captured["model"].models["y"].target_col == "y"


def test_visualization_outputs_returns_class_labels() -> None:
    """Classification output selectors should use fitted target labels."""

    from malchan.visualization import visualization_outputs

    assert visualization_outputs(FakeClassificationModel(), "label") == ["OK", "NG"]
