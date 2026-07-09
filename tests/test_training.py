import pytest


def test_tune_model_returns_none_base_params_for_single_model(monkeypatch):
    """単体モデルのチューニング結果ではbase_model_paramにNoneを返す。"""
    pytest.importorskip("optuna")
    from malchan.models import training

    class DummySearchCV:
        def __init__(self, estimator, params, **kwargs):
            self.estimator = estimator
            self.params = params
            self.best_estimator_ = estimator
            self.best_params_ = {"predictor__alpha": 0.1}

        def fit(self, X, y, **fit_params):
            return self

    monkeypatch.setattr(training, "OptunaSearchCV", DummySearchCV)
    monkeypatch.setattr(training, "get_params", lambda **kwargs: {"predictor__alpha": [0.1]})

    model, best_params, best_base_param = training.tune_model(
        X=[[0.0], [1.0]],
        y=[0.0, 1.0],
        model_pipeline=object(),
        model_names=["Ridge"],
        task="regression",
        n_trials=1,
    )

    assert model is not None
    assert best_params == {"alpha": 0.1}
    assert best_base_param is None
