import sys
import types

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor


def _install_predictor_import_stubs():
    """テスト対象外の任意依存とutilsを最小限のダミーで補う。

    Returns:
        None: この関数は ``sys.modules`` を更新するだけで値を返さない。
    """
    optional_modules = {
        "xgboost": ["XGBRegressor", "XGBClassifier"],
        "lightgbm": ["LGBMRegressor", "LGBMClassifier"],
        "catboost": ["CatBoostRegressor", "CatBoostClassifier"],
    }
    for module_name, class_names in optional_modules.items():
        module = types.ModuleType(module_name)
        for class_name in class_names:
            setattr(module, class_name, type(class_name, (), {}))
        sys.modules.setdefault(module_name, module)

    utils = types.ModuleType("malchan.models.utils")
    utils.REG_MODEL_DICT = {
        "線形回帰": LinearRegression,
        "Ridge": Ridge,
        "多層パーセプトロン": MLPRegressor,
    }
    utils.CLS_MODEL_DICT = {}
    utils.AD_MODEL_DICT = {}
    utils.SAMPLING_DICT = {}
    utils.reg_default_params = {
        "線形回帰": {},
        "Ridge": {"alpha": 0.01},
        "多層パーセプトロン": {
            "hidden_layer_sizes": (10,),
            "early_stopping": True,
            "max_iter": 10,
            "random_state": 42,
        },
    }
    utils.cls_default_params = {}
    utils.ad_default_params = {}
    utils.get_param_grid_reg = lambda *args, **kwargs: {}
    utils.get_param_grid_cls = lambda *args, **kwargs: {}
    sys.modules.setdefault("malchan.models.utils", utils)


_install_predictor_import_stubs()

from malchan.models.pipelines.predictor_pipeline import make_predictor  # noqa: E402


def test_make_predictor_accepts_tuned_single_model_param_dict():
    """チューニング後の単体モデル用パラメータ辞書をそのまま利用する。"""
    predictor = make_predictor(
        model_names=["Ridge"],
        model_params={"alpha": 0.25},
        task="regression",
    )

    assert isinstance(predictor, Ridge)
    assert predictor.alpha == 0.25


def test_make_predictor_accepts_empty_tuned_linear_regression_param_dict():
    """線形回帰のチューニング後に空辞書が渡ってもKeyErrorにしない。"""
    predictor = make_predictor(
        model_names=["線形回帰"],
        model_params={},
        task="regression",
    )

    assert isinstance(predictor, LinearRegression)


def test_make_bagging_predictor_defaults_base_model_to_first_model():
    """バギングでbase_model未指定の場合は先頭モデルをベースモデルにする。"""
    predictor = make_predictor(
        model_names=["Ridge"],
        ensemble=True,
        ens_type="バギング",
        base_model=None,
        task="regression",
    )

    assert isinstance(predictor.estimator, Ridge)
    assert predictor.estimator.alpha == 0.01


def test_make_mlp_bagging_predictor_disables_internal_early_stopping():
    """MLPバギングではゼロ重み検証を避けるため内部早期終了を無効化する。"""
    base_model_params = {
        "hidden_layer_sizes": (5,),
        "early_stopping": True,
        "max_iter": 5,
        "random_state": 42,
    }

    predictor = make_predictor(
        model_names=["多層パーセプトロン"],
        ensemble=True,
        ens_type="バギング",
        base_model=None,
        base_model_params=base_model_params,
        task="regression",
    )

    assert isinstance(predictor.estimator, MLPRegressor)
    assert predictor.estimator.early_stopping is False
    assert base_model_params["early_stopping"] is True


def test_make_single_mlp_predictor_keeps_early_stopping_enabled():
    """単体MLPでは従来どおり内部早期終了を利用する。"""
    predictor = make_predictor(
        model_names=["多層パーセプトロン"],
        model_params={
            "hidden_layer_sizes": (5,),
            "early_stopping": True,
            "max_iter": 5,
            "random_state": 42,
        },
        task="regression",
    )

    assert isinstance(predictor, MLPRegressor)
    assert predictor.early_stopping is True


def test_make_boosting_predictor_defaults_base_model_to_first_model():
    """ブースティングでbase_model未指定の場合は先頭モデルをベースモデルにする。"""
    predictor = make_predictor(
        model_names=["Ridge"],
        ensemble=True,
        ens_type="ブースティング",
        base_model=None,
        task="regression",
    )

    assert isinstance(predictor.estimator, Ridge)
    assert predictor.estimator.alpha == 0.01
