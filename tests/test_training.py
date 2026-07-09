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


def test_bagging_tuning_searches_single_base_model_and_restores_ensemble(monkeypatch):
    """バギングのチューニングでは単体モデルを探索し、最終的にアンサンブルへ戻す。"""
    pytest.importorskip("optuna")
    from sklearn.ensemble import BaggingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from malchan.models import training

    captured = {}

    class DummySearchCV:
        def __init__(self, estimator, params, **kwargs):
            captured["predictor_type"] = type(estimator.named_steps["predictor"])
            captured["params"] = params
            self.best_estimator_ = estimator
            self.best_params_ = {"predictor__alpha": 0.25}

        def fit(self, X, y, **fit_params):
            return self

    monkeypatch.setattr(training, "OptunaSearchCV", DummySearchCV)
    monkeypatch.setattr(training, "get_params", lambda **kwargs: {"predictor__alpha": [0.25]})

    pipeline = Pipeline([("predictor", BaggingRegressor(estimator=Ridge()))])

    model, best_params, best_base_param = training.tune_model(
        X=[[0.0], [1.0], [2.0]],
        y=[0.0, 1.0, 2.0],
        model_pipeline=pipeline,
        model_names=["Ridge"],
        ens_type="バギング",
        task="regression",
        n_trials=1,
    )

    assert captured["predictor_type"] is Ridge
    assert captured["params"] == {"predictor__alpha": [0.25]}
    assert best_params is None
    assert best_base_param == {"alpha": 0.25}
    assert isinstance(model.named_steps["predictor"], BaggingRegressor)
    assert model.named_steps["predictor"].estimator.alpha == 0.25


def test_bagging_params_use_single_model_parameter_names(monkeypatch):
    """バギングの探索範囲はアンサンブル内推定器ではなく単体モデル用の名前で取得する。"""
    pytest.importorskip("optuna")
    from malchan.models import training

    monkeypatch.setattr(training, "get_param_grid_reg", lambda model_name: {"predictor__alpha": [0.1]})

    params = training.get_params(
        model_names=["Ridge"],
        task="regression",
        ens_type="バギング",
    )

    assert params == {"predictor__alpha": [0.1]}
