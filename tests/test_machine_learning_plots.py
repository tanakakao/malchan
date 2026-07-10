import sys
import types

import numpy as np
import pandas as pd

from malchan.pipeline.single_output import PipelineSharedContext, SingleOutputMLModelPipeline


class FakeScatter:
    """Test double for plotly.graph_objects.Scatter."""

    def __init__(self, **kwargs):
        """Store scatter trace keyword arguments as attributes."""
        self.__dict__.update(kwargs)


class FakeBar:
    """Test double for plotly.graph_objects.Bar."""

    def __init__(self, **kwargs):
        """Store bar trace keyword arguments as attributes."""
        self.__dict__.update(kwargs)


class FakeHeatmap:
    """Test double for plotly.graph_objects.Heatmap."""

    def __init__(self, **kwargs):
        """Store heatmap trace keyword arguments as attributes."""
        self.__dict__.update(kwargs)


class FakeContour:
    """Test double for plotly.graph_objects.Contour."""

    def __init__(self, **kwargs):
        """Store contour trace keyword arguments as attributes."""
        self.__dict__.update(kwargs)


class FakeFigure:
    """Test double for plotly.graph_objects.Figure."""

    def __init__(self, data=None):
        """Create an empty figure and optionally add initial data."""
        self.data = []
        self.layout = {}
        if data is not None:
            self.data.append(data)

    def add_trace(self, trace):
        """Append a trace to the figure."""
        self.data.append(trace)

    def update_layout(self, **kwargs):
        """Store layout updates."""
        self.layout.update(kwargs)


fake_plotly = types.ModuleType("plotly")
fake_go = types.ModuleType("plotly.graph_objects")
fake_go.Figure = FakeFigure
fake_go.Scatter = FakeScatter
fake_go.Bar = FakeBar
fake_go.Heatmap = FakeHeatmap
fake_go.Contour = FakeContour
fake_plotly.graph_objects = fake_go
sys.modules.setdefault("plotly", fake_plotly)
sys.modules.setdefault("plotly.graph_objects", fake_go)

import importlib.util  # noqa: E402

module_path = "src/malchan/visualization/machine_learning_plots.py"
spec = importlib.util.spec_from_file_location("machine_learning_plots", module_path)
machine_learning_plots = importlib.util.module_from_spec(spec)
spec.loader.exec_module(machine_learning_plots)
show_importances = machine_learning_plots.show_importances
yy_plot_ml = machine_learning_plots.yy_plot_ml
show_pd_and_ice = machine_learning_plots.show_pd_and_ice
show_pd_2d = machine_learning_plots.show_pd_2d
show_shap_scatter = machine_learning_plots.show_shap_scatter
show_shap_beeswarm = machine_learning_plots.show_shap_beeswarm


class DummyMultiOutputModel:
    """Minimal multi-output model for visualization tests."""

    def __init__(self):
        """Create a model whose child target is stored in shared context."""
        context = PipelineSharedContext(
            X=pd.DataFrame({"x": [0, 1, 2]}),
            Y=pd.DataFrame({"property": [1.0, 2.0, 3.0]}),
            num_cols=["x"],
            cat_cols=[],
            smiles_cols=[],
            comp_cols=[],
            all_cols=["x"],
            unique_cols={},
        )
        child_model = SingleOutputMLModelPipeline()
        child_model.context = context
        child_model.target_col = "property"
        child_model.task = "regression"
        self.models = {"property": child_model}

    def predict(self):
        """Return predictions for the shared-context training rows."""
        return pd.DataFrame({"property": [1.1, 1.9, 3.2]})


class DummyNoImportanceChild:
    """Child model that simulates estimators without built-in importances."""

    def __init__(self):
        """Create a child model with feature names but no importance getter."""
        self.feature_names = ["x", "z"]


class DummyNoImportanceModel:
    """Multi-output model whose child has no importance API."""

    def __init__(self):
        """Create the child model mapping."""
        self.models = {"property": DummyNoImportanceChild()}


def test_show_importances_returns_empty_figure_when_getter_is_missing():
    """show_importances returns an empty figure for models without importances."""
    fig = show_importances(model=DummyNoImportanceModel(), target="property")

    assert fig.data == []


def test_show_importances_returns_empty_figure_when_importances_are_none():
    """show_importances returns an empty figure when importances are unavailable."""
    fig = show_importances(feature_names=["x", "z"], feature_importances=None)

    assert fig.data == []


def test_yy_plot_ml_uses_shared_context_y_when_child_y_is_none():
    """yy_plot_ml can plot multi-output models whose child y is stored in context."""
    fig = yy_plot_ml(model=DummyMultiOutputModel(), target="property", cv=False)

    assert list(fig.data[0].x) == [1.0, 2.0, 3.0]
    assert list(fig.data[0].y) == [1.1, 1.9, 3.2]


class DummyVisualizationChild(SingleOutputMLModelPipeline):
    """Child model with shared-context data and visualization payloads."""

    def __init__(self):
        """Create shared-context visualization fixtures."""
        super().__init__()
        self.context = PipelineSharedContext(
            X=pd.DataFrame({"x": [0.0, 1.0, 2.0], "z": [2.0, 1.0, 0.0]}),
            Y=pd.DataFrame({"property": [1.0, 2.0, 3.0]}),
            num_cols=["x", "z"],
            cat_cols=[],
            smiles_cols=[],
            comp_cols=[],
            all_cols=["x", "z"],
            unique_cols={},
        )
        self.target_col = "property"
        self.task = "regression"
        self.shap_values = np.array([[0.1, 0.2], [0.2, 0.1], [0.3, 0.0]])

    def get_pd_and_ice(self, target_col):
        """Return one-dimensional PD/ICE test data."""
        return np.array([[1.0, 1.1, 1.2], [2.0, 2.1, 2.2]]), np.array([0.0, 1.0])

    def get_pd_2d(self, target_cols):
        """Return two-dimensional PD/ICE test data."""
        return np.ones((2, 3, 2)), np.array([0.0, 1.0]), np.array([0.0, 1.0])

    def get_shap_scatter_data(self, target_col):
        """Return SHAP scatter test data."""
        return {
            "x": pd.DataFrame({"x": [0.0, 1.0, 2.0], "shap": [0.1, 0.2, 0.3]}),
            "z": pd.DataFrame({"z": [2.0, 1.0, 0.0], "shap": [0.2, 0.1, 0.0]}),
        }


class DummyVisualizationModel:
    """Minimal multi-output model for additional visualization tests."""

    def __init__(self):
        """Create the shared-context child model mapping."""
        self.models = {"property": DummyVisualizationChild()}


def test_pd_and_ice_uses_shared_context_X_and_y_when_child_attrs_are_none():
    """show_pd_and_ice can overlay actual data from shared context."""
    fig = show_pd_and_ice(model=DummyVisualizationModel(), target="property", target_col="x")

    actual_trace = fig.data[-1]
    assert list(actual_trace.x) == [0.0, 1.0, 2.0]
    assert list(actual_trace.y) == [1.0, 2.0, 3.0]


def test_pd_2d_uses_shared_context_X_and_y_when_child_attrs_are_none():
    """show_pd_2d can overlay actual data from shared context."""
    fig = show_pd_2d(model=DummyVisualizationModel(), target="property", target_cols=["x", "z"])

    actual_trace = fig.data[-1]
    assert list(actual_trace.x) == [0.0, 1.0, 2.0]
    assert list(actual_trace.y) == [2.0, 1.0, 0.0]
    assert list(actual_trace.marker["color"]) == [1.0, 2.0, 3.0]
    assert fig.layout["xaxis"]["range"] == [0.0, 1.0]
    assert fig.layout["yaxis"]["range"] == [0.0, 1.0]


def test_shap_scatter_uses_shared_context_rawX_when_child_X_is_none():
    """show_shap_scatter can plot SHAP data with raw X from shared context."""
    fig = show_shap_scatter(model=DummyVisualizationModel(), target="property", target_col="x")

    assert list(fig.data[0].x) == [0.0, 1.0, 2.0]
    assert list(fig.data[0].y) == [0.1, 0.2, 0.3]


def test_shap_scatter_accepts_dataframe_scatter_data():
    """show_shap_scatter accepts direct DataFrame output from get_shap_scatter_data."""
    X_shappd = pd.DataFrame({"x": [0.0, 1.0, 2.0], "shap": [0.1, 0.2, 0.3]})
    rawX = pd.DataFrame({"x": [0.0, 1.0, 2.0]})

    fig = show_shap_scatter(
        X_shappd=X_shappd,
        rawX=rawX,
        shap_values=np.array([[0.1], [0.2], [0.3]]),
        target_col="x",
    )

    assert list(fig.data[0].x) == [0.0, 1.0, 2.0]
    assert list(fig.data[0].y) == [0.1, 0.2, 0.3]


def test_shap_scatter_model_colors_by_interactive_col_from_rawX():
    """show_shap_scatter uses rawX for model interactive_col colors."""
    fig = show_shap_scatter(
        model=DummyVisualizationModel(),
        target="property",
        target_col="x",
        interactive_col="z",
    )

    assert list(fig.data[0].marker["color"]) == [2.0, 1.0, 0.0]


def test_shap_scatter_colors_by_numeric_interactive_col_from_rawX():
    """show_shap_scatter uses rawX for numeric interactive_col colors."""
    X_shappd = {
        "x": pd.DataFrame({"x": [0.0, 1.0, 2.0], "shap": [0.1, 0.2, 0.3]})
    }
    rawX = pd.DataFrame({"x": [0.0, 1.0, 2.0], "z": [2.0, 1.0, 0.0]})

    fig = show_shap_scatter(
        X_shappd=X_shappd,
        rawX=rawX,
        shap_values=np.array([[0.1], [0.2], [0.3]]),
        target_col="x",
        interactive_col="z",
    )

    assert list(fig.data[0].marker["color"]) == [2.0, 1.0, 0.0]


def test_shap_scatter_raises_when_interactive_col_is_missing_from_rawX():
    """show_shap_scatter reports a missing interactive_col clearly."""
    X_shappd = pd.DataFrame({"x": [0.0, 1.0, 2.0], "shap": [0.1, 0.2, 0.3]})
    rawX = pd.DataFrame({"x": [0.0, 1.0, 2.0]})

    try:
        show_shap_scatter(
            X_shappd=X_shappd,
            rawX=rawX,
            shap_values=np.array([[0.1], [0.2], [0.3]]),
            target_col="x",
            interactive_col="z",
        )
    except ValueError as exc:
        assert "'z' is not found in rawX." in str(exc)
    else:
        raise AssertionError("Expected ValueError for missing interactive_col")


def test_shap_scatter_returns_empty_figure_when_shap_values_are_none():
    """show_shap_scatter returns an empty figure when SHAP values are unavailable."""
    fig = show_shap_scatter(
        X_shappd=pd.DataFrame({"x": [0.0], "shap": [0.1]}),
        rawX=pd.DataFrame({"x": [0.0]}),
        shap_values=None,
        target_col="x",
    )

    assert fig.data == []


def test_shap_beeswarm_returns_empty_figure_when_shap_values_are_none():
    """show_shap_beeswarm returns an empty figure when SHAP values are unavailable."""
    fig = show_shap_beeswarm(X=pd.DataFrame({"x": [0.0]}), shap_values=None)

    assert fig.data == []


def test_shap_beeswarm_uses_shared_context_X_when_child_X_is_none():
    """show_shap_beeswarm can build feature rankings from shared-context X."""
    fig = show_shap_beeswarm(model=DummyVisualizationModel(), target="property", n_shap_top=2)

    assert len(fig.data[0].x) == 6


def test_shap_beeswarm_spreads_dense_shap_values_more_than_sparse_values():
    """show_shap_beeswarm expands rows by local SHAP-value density."""
    X = pd.DataFrame({"dense": np.arange(8), "sparse": np.arange(8)})
    shap_values = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [0.0, 2.0],
            [0.0, 3.0],
            [0.0, 4.0],
            [0.0, 5.0],
            [0.0, 6.0],
            [0.0, 7.0],
        ]
    )

    fig = show_shap_beeswarm(X=X, shap_values=shap_values, n_shap_top=2)

    y_values = np.asarray(fig.data[0].y)
    dense_row = y_values[:8]
    sparse_row = y_values[8:]
    assert np.ptp(dense_row) > np.ptp(sparse_row)
    assert np.ptp(dense_row) > 0.0

def test_shap_beeswarm_uses_roomier_vertical_layout():
    """show_shap_beeswarm reserves more vertical space for feature rows."""
    X = pd.DataFrame({"a": [0.0, 1.0], "b": [1.0, 0.0], "c": [0.5, 0.2]})
    shap_values = np.array([[0.1, 0.2, 0.3], [0.3, 0.2, 0.1]])

    fig = show_shap_beeswarm(X=X, shap_values=shap_values, n_shap_top=3)

    assert fig.layout["height"] == 460
    assert fig.layout["margin"] == {"t": 70, "b": 70}
    assert fig.layout["yaxis"]["range"] == [-0.8, 2.8]


class DummyMulticlassClassificationModel:
    """Minimal multiclass classifier with CV probability predictions."""

    def __init__(self):
        """Create multiclass classification fixtures."""
        child = SingleOutputMLModelPipeline()
        child.y = pd.Series(["a", "b", "c"], name="y_cat_str")
        child.task = "classification"
        child.target_items = ["a", "b", "c"]
        child.cv_preds = {
            "train": pd.DataFrame(
                {
                    "y_cat_str_a": [0.8, 0.1, 0.2],
                    "y_cat_str_b": [0.1, 0.7, 0.2],
                    "y_cat_str_c": [0.1, 0.2, 0.6],
                }
            )
        }
        self.models = {"y_cat_str": child}

    def predict(self):
        """Return label predictions for non-CV classification plotting."""
        return pd.DataFrame({"y_cat_str": ["a", "b", "c"]})


def test_yy_plot_ml_accepts_multiclass_cv_probability_matrix():
    """yy_plot_ml converts multiclass CV probability matrices before confusion_matrix."""
    fig = yy_plot_ml(model=DummyMulticlassClassificationModel(), target="y_cat_str", cv=True)

    assert fig.data[0].z.tolist() == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    assert list(fig.data[0].x) == ["a", "b", "c"]
    assert list(fig.data[0].y) == ["a", "b", "c"]
