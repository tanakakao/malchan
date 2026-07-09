import sys
import types

import pandas as pd

from malchan.pipeline.single_output import SingleOutputMLModelPipeline


def test_cv_score_uses_probability_predictions_for_classification(monkeypatch):
    """分類CVの予測保存ではクラスラベルではなく確率を使用する。"""
    fake_training = types.ModuleType("malchan.models.training")
    fake_training.cv_fit = lambda *args, **kwargs: args[2]
    monkeypatch.setitem(sys.modules, "malchan.models.training", fake_training)

    pipeline = SingleOutputMLModelPipeline()
    pipeline.X = pd.DataFrame({"x": range(6)})
    pipeline.y = pd.Series(["a", "b", "c", "a", "b", "c"], name="y_cat_str")
    pipeline.task = "classification"
    pipeline.cat_index = []
    pipeline.cat_index_fit = []
    pipeline.model_names = ["dummy"]
    pipeline.sampling_method = None
    pipeline.ensemble = False
    proba_calls = []

    monkeypatch.setattr(pipeline, "_make_pipeline", lambda: object())
    monkeypatch.setattr(
        pipeline,
        "score",
        lambda X, y, model: pd.DataFrame({"ACCURACY": [1.0], "PRECISION": [1.0], "RECALL": [1.0], "F1": [1.0]}),
    )

    def predict(X=None, model=None, proba=False, idx2item=False):
        """Record predict mode and return probability-like numeric predictions."""
        proba_calls.append(proba)
        assert proba is True
        return pd.DataFrame({"y_cat_str_a": [0.6] * len(X), "y_cat_str_b": [0.3] * len(X), "y_cat_str_c": [0.1] * len(X)})

    monkeypatch.setattr(pipeline, "predict", predict)

    pipeline.cv_score(n_splits=3)

    assert proba_calls == [True] * 6
    assert list(pipeline.cv_preds["train"].columns) == ["y_cat_str_a", "y_cat_str_b", "y_cat_str_c"]
    assert list(pipeline.cv_preds["test"].columns) == ["y_cat_str_a", "y_cat_str_b", "y_cat_str_c"]
    assert pipeline.cv_preds["train"].dtypes.tolist() == ["float64", "float64", "float64"]


def test_cv_score_keeps_label_predictions_for_regression(monkeypatch):
    """回帰CVの予測保存では従来通り通常の予測値を使用する。"""
    fake_training = types.ModuleType("malchan.models.training")
    fake_training.cv_fit = lambda *args, **kwargs: args[2]
    monkeypatch.setitem(sys.modules, "malchan.models.training", fake_training)

    pipeline = SingleOutputMLModelPipeline()
    pipeline.X = pd.DataFrame({"x": range(6)})
    pipeline.y = pd.Series([1.0, 2.0, 3.0, 1.0, 2.0, 3.0], name="y")
    pipeline.task = "regression"
    pipeline.cat_index = []
    pipeline.cat_index_fit = []
    pipeline.model_names = ["dummy"]
    pipeline.sampling_method = None
    pipeline.ensemble = False
    proba_calls = []

    monkeypatch.setattr(pipeline, "_make_pipeline", lambda: object())
    monkeypatch.setattr(pipeline, "score", lambda X, y, model: pd.DataFrame({"RMSE": [0.0], "MAE": [0.0], "MAPE": [0.0], "R2": [1.0]}))

    def predict(X=None, model=None, proba=False, idx2item=False):
        """Record predict mode and return numeric regression predictions."""
        proba_calls.append(proba)
        return pd.DataFrame({"y": [1.0] * len(X)})

    monkeypatch.setattr(pipeline, "predict", predict)

    pipeline.cv_score(n_splits=3)

    assert proba_calls == [False] * 6
    assert list(pipeline.cv_preds["train"].columns) == ["y"]
