import importlib.util
import sys
import types

import numpy as np
import pandas as pd


class FakeScatter:
    """Test double for plotly.graph_objects.Scatter."""

    def __init__(self, **kwargs):
        """Store scatter trace keyword arguments as attributes."""
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
fake_go.Bar = type("FakeBar", (), {})
fake_go.Heatmap = type("FakeHeatmap", (), {})
fake_go.Contour = type("FakeContour", (), {})
fake_plotly.graph_objects = fake_go
sys.modules.setdefault("plotly", fake_plotly)
sys.modules.setdefault("plotly.graph_objects", fake_go)

module_path = "src/malchan/visualization/machine_learning_plots.py"
spec = importlib.util.spec_from_file_location("machine_learning_plots_shap_sample", module_path)
machine_learning_plots = importlib.util.module_from_spec(spec)
spec.loader.exec_module(machine_learning_plots)
show_shap_beeswarm = machine_learning_plots.show_shap_beeswarm


class DummyChildModel:
    """Child model whose SHAP matrix uses polynomially transformed features."""

    def __init__(self):
        """Create raw and transformed feature matrices with different widths."""
        self.X = pd.DataFrame({"x": [0.0, 1.0, 2.0], "z": [2.0, 1.0, 0.0]})
        self.X_sample = pd.DataFrame(
            {
                "x": [0.0, 1.0, 2.0],
                "z": [2.0, 1.0, 0.0],
                "x^2": [0.0, 1.0, 4.0],
            }
        )
        self.shap_values = np.array(
            [
                [0.1, 0.2, 0.3],
                [0.2, 0.1, 0.4],
                [0.3, 0.0, 0.5],
            ]
        )

    def _get_X(self):
        """Return the original, untransformed training matrix."""
        return self.X

    def _shared_attr(self, name):
        """Return visualization metadata used by the plotting function."""
        if name == "cat_cols":
            return []
        raise KeyError(name)


class DummyModel:
    """Minimal multi-output model container for SHAP visualization."""

    def __init__(self):
        """Expose the transformed-feature child model under one target."""
        self.models = {"property": DummyChildModel()}


def test_shap_beeswarm_uses_shap_sample_after_polynomial_transform():
    """Model-based beeswarm uses X_sample whose columns match transformed SHAP values."""
    fig = show_shap_beeswarm(model=DummyModel(), target="property", n_shap_top=3)

    assert len(fig.data[0].x) == 9
    assert "x^2" in list(fig.layout["yaxis"]["ticktext"])
