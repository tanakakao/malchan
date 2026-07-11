import importlib.util
from types import SimpleNamespace

import pandas as pd
import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("optuna") is None,
    reason="Inverse-analysis compatibility tests require the inverse extra.",
)


class SharedMultiModel:
    """Multi-output test double storing training data in shared context."""

    def __init__(self) -> None:
        """Create shared input metadata matching MLModelPipeline storage."""

        self.X = None
        self.target_cols = ["strength", "cost"]
        self.num_cols = ["x"]
        self.cat_cols = []
        self.smiles_cols = []
        self.comp_cols = []
        self.context = SimpleNamespace(
            X=pd.DataFrame({"x": [1.0, 2.0]}),
            num_cols=["x"],
            cat_cols=[],
            smiles_cols=[],
            comp_cols=[],
        )
        self.received_obj_values = None

    def predict(self, X=None, proba=False, idx2item=False):
        """Return deterministic multi-output predictions."""

        return pd.DataFrame(
            {
                "strength": [float(value) * 2 for value in X["x"]],
                "cost": [float(value) + 1 for value in X["x"]],
            }
        )

    def predict_objective(self, X=None, obj_values=None):
        """Record multi-output objective values."""

        self.received_obj_values = obj_values
        return self.predict(X)


class SingleModel:
    """Single-output test double using the obj_value signature."""

    def __init__(self) -> None:
        """Create local single-output training metadata."""

        self.X = pd.DataFrame({"x": [1.0, 2.0]})
        self.target_col = "y"
        self.num_cols = ["x"]
        self.cat_cols = []
        self.smiles_cols = []
        self.comp_cols = []
        self.received_obj_value = None

    def predict(self, X=None, proba=False, idx2item=False):
        """Return deterministic single-output predictions."""

        return pd.DataFrame({"y": [float(value) * 2 for value in X["x"]]})

    def predict_objective(self, X=None, obj_value=None):
        """Record the single-output objective value."""

        self.received_obj_value = obj_value
        return self.predict(X)


def test_inverse_model_view_reads_shared_context() -> None:
    """The normalized view should expose shared MLModelPipeline training data."""

    from malchan.inverse_analysis.models import _InverseAnalysisModelView

    model = SharedMultiModel()
    view = _InverseAnalysisModelView(model)
    objective = view.predict_objective(
        pd.DataFrame({"x": [3.0]}),
        obj_values=[None, 100.0],
    )

    assert view.target_cols == ["strength", "cost"]
    assert view.X.equals(model.context.X)
    assert model.received_obj_values == [None, 100.0]
    assert objective.to_dict(orient="records") == [
        {"strength": 6.0, "cost": 4.0}
    ]


def test_inverse_model_view_maps_single_objective_signature() -> None:
    """The normalized view should map obj_values to single-model obj_value."""

    from malchan.inverse_analysis.models import _InverseAnalysisModelView

    model = SingleModel()
    view = _InverseAnalysisModelView(model)
    objective = view.predict_objective(
        pd.DataFrame({"x": [3.0]}),
        obj_values=[5.0],
    )

    assert view.target_cols == ["y"]
    assert model.received_obj_value == 5.0
    assert objective.to_dict(orient="records") == [{"y": 6.0}]


def test_integer_search_settings_are_cast_for_optuna() -> None:
    """Integral float values should be normalized before suggest_int calls."""

    from malchan.inverse_analysis.models import _normalize_integer_search_settings

    lower, upper, steps = _normalize_integer_search_settings(
        bounds_min=[0.0, 0.1],
        bounds_max=[10.0, 0.9],
        steps=[2.0, None],
        dtypes=["int", "float"],
    )

    assert lower == [0, 0.1]
    assert upper == [10, 0.9]
    assert steps == [2, None]
    assert isinstance(lower[0], int)
    assert isinstance(upper[0], int)
    assert isinstance(steps[0], int)
