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

    def _get_X(self) -> pd.DataFrame:
        return self.X

    def _get_y(self) -> pd.DataFrame:
        return self.y

    def predict(self, X=None, proba=False, idx2item=False):
        data = self.X if X is None else X
        return pd.DataFrame({"y": data["x1"] * 2.0 + 1.0}, index=data.index)

    def get_pd_2d(self, target_cols):
        assert target_cols == ["x1", "x2"]
        values = np.asarray(
            [
                [1.0, 1.2],
                [2.0, 2.2],
                [3.0, 3.2],
                [4.0, 4.2],
            ]
        )
        x_grid = np.asarray([0.0, 1.0, 0.0, 1.0])
        y_grid = np.asarray([10.0, 10.0, 20.0, 20.0])
        return values, x_grid, y_grid


class FakeClassificationModel(FakeRegressionModel):
    """Classification variant exposing selectable class outputs."""

    target_col = "label"
    task = "classification"
    target_items = np.asarray(["OK", "NG"])


def test_show_model_diagnostics_supports_single_output_model() -> None:
    """The web adapter should reuse yy_plot_ml for a single-output pipeline."""

    from malchan.visualization import show_model_diagnostics

    figure = show_model_diagnostics(FakeRegressionModel(), "y")

    assert len(figure.data) == 2
    assert figure.data[0].type == "scatter"
    assert figure.layout.xaxis.title.text == "Actual Values"


def test_show_model_pd_2d_reshapes_flat_model_grid() -> None:
    """Flattened model grids should become a two-dimensional Plotly contour."""

    from malchan.visualization import show_model_pd_2d

    figure = show_model_pd_2d(
        FakeRegressionModel(),
        "y",
        ["x1", "x2"],
    )

    contour = figure.data[0]
    assert contour.type == "contour"
    assert np.asarray(contour.z).shape == (2, 2)
    assert list(contour.x) == [0.0, 1.0]
    assert list(contour.y) == [10.0, 20.0]
    assert figure.data[1].name == "Observed data"


def test_visualization_outputs_returns_class_labels() -> None:
    """Classification output selectors should use fitted target labels."""

    from malchan.visualization import visualization_outputs

    assert visualization_outputs(FakeClassificationModel(), "label") == ["OK", "NG"]
