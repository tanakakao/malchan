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


def test_catboost_regression_tuning_uses_iterations_parameter():
    """CatBoost回帰のチューニング対象にiterationsを使用する。

    CatBoostは``iterations``と``n_estimators``を同時に設定できないため、
    デフォルトパラメータで使っている``iterations``に統一されていることを確認する。
    """
    pytest.importorskip("optuna")
    from malchan.models.utils import get_param_grid_reg

    params = get_param_grid_reg("CatBoost")

    assert "predictor__iterations" in params
    assert "predictor__n_estimators" not in params


def test_catboost_classification_tuning_uses_iterations_parameter():
    """CatBoost分類のチューニング対象にiterationsを使用する。

    CatBoostは``iterations``と``n_estimators``を同時に設定できないため、
    デフォルトパラメータで使っている``iterations``に統一されていることを確認する。
    """
    pytest.importorskip("optuna")
    from malchan.models.utils import get_param_grid_cls

    params = get_param_grid_cls("CatBoost")

    assert "predictor__iterations" in params
    assert "predictor__n_estimators" not in params


def test_mlp_regression_hidden_layer_sizes_uses_optuna_distribution():
    """MLP回帰のhidden_layer_sizesはOptuna Distributionで定義する。"""
    optuna = pytest.importorskip("optuna")
    from malchan.models.utils import get_param_grid_reg

    params = get_param_grid_reg("多層パーセプトロン")

    distribution = params["predictor__hidden_layer_sizes"]
    assert isinstance(distribution, optuna.distributions.BaseDistribution)
    assert all(isinstance(choice, tuple) for choice in distribution.choices)


def test_mlp_classification_hidden_layer_sizes_uses_optuna_distribution():
    """MLP分類のhidden_layer_sizesはOptuna Distributionで定義する。"""
    optuna = pytest.importorskip("optuna")
    from malchan.models.utils import get_param_grid_cls

    params = get_param_grid_cls("多層パーセプトロン")

    distribution = params["predictor__hidden_layer_sizes"]
    assert isinstance(distribution, optuna.distributions.BaseDistribution)
    assert all(isinstance(choice, tuple) for choice in distribution.choices)
