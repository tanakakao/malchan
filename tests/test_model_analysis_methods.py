import sys
from types import ModuleType, SimpleNamespace

import pandas as pd


class FakeSinglePipeline:
    """Small pipeline double supporting comparison refits and CV scores."""

    score_by_model = {
        "model-a": {"RMSE": 2.0, "MAE": 1.5, "MAPE": 0.2, "R2": 0.5},
        "model-b": {"RMSE": 1.0, "MAE": 0.8, "MAPE": 0.1, "R2": 0.8},
    }

    def __init__(self) -> None:
        """Initialize an unfitted pipeline double."""

        self.X = None
        self.y = None
        self.context = None
        self.target_col = None
        self.task = None
        self.model_names = None
        self.fingerprints = []
        self.comp_method = None
        self.comp_feats = []
        self.num_impute_type = None
        self.num_scale_type = None
        self.cat_impute = False
        self.poly = False
        self.poly_degree = 1
        self.poly_interaction_only = True
        self.decomposition = False
        self.decomposition_method = "PCA"
        self.dec_n_components = 2
        self.sampling_method = None
        self.le = None

    @classmethod
    def fitted(cls) -> "FakeSinglePipeline":
        """Return a fitted seed model with two training rows."""

        model = cls()
        model.X = pd.DataFrame({"x": [1.0, 2.0]})
        model.y = pd.DataFrame({"y": [2.0, 4.0]})
        model.target_col = "y"
        model.task = "regression"
        model.model_names = ["model-a"]
        model.num_cols = ["x"]
        model.cat_cols = []
        model.smiles_cols = []
        model.comp_cols = []
        model.all_cols = ["x"]
        model.unique_cols = {}
        return model

    def _get_X(self):
        """Return local or shared training features."""

        return self.context.X if self.context is not None else self.X

    def _shared_attr(self, name):
        """Return local or shared metadata."""

        return getattr(self.context, name) if self.context is not None else getattr(self, name)

    def fit_from_context(self, context, target_col, **kwargs):
        """Record the candidate model and shared training context."""

        self.context = context
        self.target_col = target_col
        self.task = kwargs["task"]
        self.model_names = kwargs["model_names"]
        for name, value in kwargs.items():
            if name not in {"task", "model_names"}:
                setattr(self, name, value)

    def cv_score(self, method="kfold", n_splits=5):
        """Populate deterministic train and test CV score tables."""

        scores = self.score_by_model[self.model_names[0]]
        self.cv_scores = {
            "train": pd.DataFrame(
                [{key: value * 0.8 for key, value in scores.items()}]
            ),
            "test": pd.DataFrame([scores]),
        }


class FakeMultiPipeline:
    """Small multi-output double used to test inverse-analysis argument mapping."""

    def __init__(self) -> None:
        """Initialize fitted multi-output metadata."""

        self.target_cols = ["strength", "cost"]
        self.tasks = ["regression", "regression"]
        self.models = {}


def test_public_pipeline_classes_have_analysis_methods() -> None:
    """Public and direct pipeline imports should expose the same added methods."""

    from malchan.pipeline import MLModelPipeline, SingleOutputMLModelPipeline
    from malchan.pipeline.multi_output import MLModelPipeline as DirectMulti
    from malchan.pipeline.single_output import (
        SingleOutputMLModelPipeline as DirectSingle,
    )

    assert SingleOutputMLModelPipeline is DirectSingle
    assert MLModelPipeline is DirectMulti
    assert callable(SingleOutputMLModelPipeline.compare)
    assert callable(SingleOutputMLModelPipeline.inverse_analysis)
    assert callable(MLModelPipeline.compare)
    assert callable(MLModelPipeline.inverse_analysis)


def test_model_compare_ranks_candidates_and_retains_best_model(monkeypatch) -> None:
    """model.compare() should rank CV metrics and retain fitted candidates."""

    from malchan.models import compare as compare_module
    from malchan.pipeline.analysis_extensions import install_analysis_extensions

    monkeypatch.setattr(
        compare_module,
        "_available_model_names",
        lambda task: ["model-a", "model-b"],
    )
    install_analysis_extensions(FakeSinglePipeline, FakeMultiPipeline)

    model = FakeSinglePipeline.fitted()
    result = model.compare(
        model_names=["model-a", "model-b"],
        n_splits=2,
    )

    assert result.metric == "RMSE"
    assert result.best_model_name == "model-b"
    assert result.best_model.model_names == ["model-b"]
    assert result.ranking["model_name"].tolist() == ["model-b", "model-a"]
    assert result.ranking["rank"].tolist() == [1, 2]
    assert result.failures == {}
    assert model.comparison_result is result


def test_single_model_inverse_analysis_is_model_native(monkeypatch) -> None:
    """A fitted model should forward its own target to inverse analysis."""

    from malchan.pipeline.analysis_extensions import install_analysis_extensions

    captured = {}
    inverse_module = ModuleType("malchan.inverse_analysis")

    def fake_inverse_analysis(model, **kwargs):
        captured["model"] = model
        captured.update(kwargs)
        return pd.DataFrame([{"x": 2.0, "pred_y": 4.0}]), SimpleNamespace()

    inverse_module.inverse_analysis = fake_inverse_analysis
    monkeypatch.setitem(sys.modules, "malchan.inverse_analysis", inverse_module)
    install_analysis_extensions(FakeSinglePipeline, FakeMultiPipeline)

    model = FakeSinglePipeline.fitted()
    candidates, study = model.inverse_analysis("max", trials=10)

    assert captured["model"] is model
    assert captured["obj_directions"] == ["max"]
    assert captured["target_cols"] == ["y"]
    assert captured["trials"] == 10
    assert model.inverse_candidates.equals(candidates)
    assert model.inverse_study is study


def test_multi_model_inverse_analysis_accepts_objective_mapping(monkeypatch) -> None:
    """Multi-output models should derive aligned targets from a mapping."""

    from malchan.pipeline.analysis_extensions import install_analysis_extensions

    captured = {}
    inverse_module = ModuleType("malchan.inverse_analysis")

    def fake_inverse_analysis(model, **kwargs):
        captured["model"] = model
        captured.update(kwargs)
        return pd.DataFrame([{"x": 1.0}]), SimpleNamespace()

    inverse_module.inverse_analysis = fake_inverse_analysis
    monkeypatch.setitem(sys.modules, "malchan.inverse_analysis", inverse_module)
    install_analysis_extensions(FakeSinglePipeline, FakeMultiPipeline)

    model = FakeMultiPipeline()
    model.inverse_analysis(
        {"strength": "max", "cost": "min"},
        sampler_type="NSGAII",
    )

    assert captured["model"] is model
    assert captured["target_cols"] == ["strength", "cost"]
    assert captured["obj_directions"] == ["max", "min"]
    assert captured["sampler_type"] == "NSGAII"
