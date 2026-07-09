import sys
import types

import pandas as pd

from malchan.pipeline.single_output import PipelineSharedContext, SingleOutputMLModelPipeline


class FakeScatter:
    """Test double for plotly.graph_objects.Scatter."""

    def __init__(self, **kwargs):
        """Store scatter trace keyword arguments as attributes."""
        self.__dict__.update(kwargs)


class FakeHeatmap:
    """Test double for plotly.graph_objects.Heatmap."""

    def __init__(self, **kwargs):
        """Store heatmap trace keyword arguments as attributes."""
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
fake_go.Heatmap = FakeHeatmap
fake_plotly.graph_objects = fake_go
sys.modules.setdefault("plotly", fake_plotly)
sys.modules.setdefault("plotly.graph_objects", fake_go)

import importlib.util  # noqa: E402

module_path = "src/malchan/visualization/machine_learning_plots.py"
spec = importlib.util.spec_from_file_location("machine_learning_plots", module_path)
machine_learning_plots = importlib.util.module_from_spec(spec)
spec.loader.exec_module(machine_learning_plots)
yy_plot_ml = machine_learning_plots.yy_plot_ml


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


def test_yy_plot_ml_uses_shared_context_y_when_child_y_is_none():
    """yy_plot_ml can plot multi-output models whose child y is stored in context."""
    fig = yy_plot_ml(model=DummyMultiOutputModel(), target="property", cv=False)

    assert list(fig.data[0].x) == [1.0, 2.0, 3.0]
    assert list(fig.data[0].y) == [1.1, 1.9, 3.2]
