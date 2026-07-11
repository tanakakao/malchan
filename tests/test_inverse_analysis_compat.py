import importlib.util
from types import SimpleNamespace

import pandas as pd
import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("optuna") is None,
    reason="Inverse-analysis compatibility tests require the inverse extra.",
)


class TargetModel:
    """Single-target child model used by a multi-output test double."""

    def __init__(self, target, offset=0.0, task="regression", target_items=None):
        """Initialize deterministic prediction behavior."""

        self.target_col = target
        self.offset = offset
        self.task = task
        self.target_items = target_items
        self.received_obj_value = None

    def predict(self, X=None, proba=False, idx2item=False):
        """Return regression values or deterministic class probabilities."""

        if self.task == "classification" and proba:
            return pd.DataFrame(
                {
                    f"{self.target_col}_0": [0.25] * len(X),
                    f"{self.target_col}_1": [0.75] * len(X),
                }
            )
        return pd.DataFrame(
            {
                self.target_col: [
                    float(value) + self.offset for value in X["x"]
                ]
            }
        )

    def predict_objective(self, X=None, obj_value=None):
        """Record the target value received from the normalized view."""

        self.received_obj_value = obj_value
        return self.predict(X)


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
        self.models = {
            "strength": TargetModel("strength", offset=3.0),
            "cost": TargetModel("cost", offset=1.0),
        }


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
        self.task = "regression"
        self.target_items = None
        self.received_obj_value = None

    def predict(self, X=None, proba=False, idx2item=False):
        """Return deterministic single-output predictions."""

        return pd.DataFrame({"y": [float(value) * 2 for value in X["x"]]})

    def predict_objective(self, X=None, obj_value=None):
        """Record the single-output objective value."""

        self.received_obj_value = obj_value
        return self.predict(X)


def test_inverse_model_view_reads_shared_context_and_child_models() -> None:
    """The normalized view should use shared data and target child pipelines."""

    from malchan.inverse_analysis.models import _InverseAnalysisModelView

    model = SharedMultiModel()
    view = _InverseAnalysisModelView(model)
    objective = view.predict_objective(
        pd.DataFrame({"x": [3.0]}),
        obj_values=[None, 100.0],
    )

    assert view.target_cols == ["strength", "cost"]
    assert view.X.equals(model.context.X)
    assert model.models["strength"].received_obj_value is None
    assert model.models["cost"].received_obj_value == 100.0
    assert objective.to_dict(orient="records") == [
        {"strength": 6.0, "cost": 4.0}
    ]


def test_inverse_model_view_supports_reordered_target_subset() -> None:
    """Selected targets should be evaluated in the requested order."""

    from malchan.inverse_analysis.models import _InverseAnalysisModelView

    model = SharedMultiModel()
    view = _InverseAnalysisModelView(model)
    view.set_active_targets(["cost"])
    objective = view.predict_objective(
        pd.DataFrame({"x": [3.0]}),
        obj_values=[5.0],
    )

    assert objective.columns.tolist() == ["cost"]
    assert objective.to_dict(orient="records") == [{"cost": 4.0}]
    assert model.models["cost"].received_obj_value == 5.0


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


def test_inverse_model_view_normalizes_numeric_class_label() -> None:
    """Numeric class labels should map to probability-column string suffixes."""

    from malchan.inverse_analysis.models import _InverseAnalysisModelView

    child = TargetModel(
        "quality",
        task="classification",
        target_items=[0, 1],
    )
    model = SimpleNamespace(
        X=pd.DataFrame({"x": [1.0, 2.0]}),
        target_col="quality",
        num_cols=["x"],
        cat_cols=[],
        smiles_cols=[],
        comp_cols=[],
        task="classification",
        target_items=[0, 1],
        predict=child.predict,
        predict_objective=child.predict_objective,
    )
    view = _InverseAnalysisModelView(model)
    view.validate_objectives(["quality"], [1])
    view.predict_objective(pd.DataFrame({"x": [3.0]}), obj_values=[1])

    assert child.received_obj_value == "1"


def test_public_inverse_analysis_infers_categories_and_unwraps_adapter(
    monkeypatch,
) -> None:
    """The public wrapper should infer observed categories from the raw model."""

    import malchan.inverse_analysis as inverse_package

    captured = {}

    def fake_inverse_analysis(model, *args, **kwargs):
        captured["model"] = model
        captured["args"] = args
        captured["kwargs"] = kwargs
        return pd.DataFrame(), SimpleNamespace()

    monkeypatch.setattr(inverse_package, "_inverse_analysis", fake_inverse_analysis)
    model = SimpleNamespace(
        X=pd.DataFrame(
            {
                "x": [1.0, 2.0],
                "grade": ["A", "B"],
            }
        ),
        num_cols=["x"],
        cat_cols=["grade"],
        smiles_cols=[],
        comp_cols=[],
    )
    adapter = SimpleNamespace(_model=model)

    inverse_package.inverse_analysis(
        adapter,
        obj_directions=["max"],
        target_cols=["y"],
    )

    assert captured["model"] is model
    assert captured["kwargs"]["cat_dict"] == {"grade": ["A", "B"]}


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
