from sklearn.base import RegressorMixin
from sklearn.linear_model import (
    LinearRegression, HuberRegressor, Ridge, Lasso, ElasticNet, TweedieRegressor,
    BayesianRidge, ARDRegression, PassiveAggressiveRegressor, LassoLars, OrthogonalMatchingPursuit
)
from sklearn.cross_decomposition import PLSRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor,
    VotingRegressor, StackingRegressor, BaggingRegressor, AdaBoostRegressor, HistGradientBoostingRegressor
)
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, DotProduct, WhiteKernel, RBF, Matern

from sklearn.base import ClassifierMixin
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, RidgeClassifier, PassiveAggressiveClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,ExtraTreesClassifier,GradientBoostingClassifier,
    VotingClassifier, StackingClassifier, BaggingClassifier, AdaBoostClassifier, HistGradientBoostingClassifier
)
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier

from typing import List, Optional, Union, Dict, Tuple, Callable, Any, Type
import warnings

from ..utils import (
    REG_MODEL_DICT, CLS_MODEL_DICT, FP_dcit, INORG_dict, AD_MODEL_DICT, SAMPLING_DICT,
    reg_default_params, cls_default_params, ad_default_params,
    get_param_grid_reg, get_param_grid_cls
)

warnings.simplefilter('ignore')

def make_model(
    model_name: str,
    task="regression",
    kwargs: Optional[Dict[str, Any]] = None
) -> Any:
    """
    指定されたモデル名に基づいて機械学習モデルを作成する関数。

    Args:
        model_name (str): 使用するモデルの名前。例: '線形回帰', 'Ridge', 'Lasso', 'XGBoost' など。
        kwargs (Optional[Dict[str, Any]]): モデルの初期化パラメータを指定する辞書。デフォルトは None。

    Returns:
        Any: 指定されたモデルのインスタンス。
    """
    # モデル名と対応するクラスを格納した辞書
    if task == "regression":
        model_dict: Dict[str, Type] = REG_MODEL_DICT
    elif task == "classification":
        model_dict: Dict[str, Type] = CLS_MODEL_DICT
    elif task == "AD":
        model_dict: Dict[str, Type] = AD_MODEL_DICT

    # kwargsがNoneの場合は空の辞書に置き換え
    if kwargs is None:
        kwargs = {}

    # 指定されたモデル名のクラスを辞書から取得し、インスタンスを作成して返す
    return model_dict[model_name](**kwargs)

def make_ens_predictor(
    ens_type: str,
    model_names: List[str],
    base_model: Optional[str] = None,
    model_params: Optional[List[Dict[str, Any]]] = None,
    base_model_params: Optional[Dict[str, Any]] = None,
    task="regression"
) -> Union[VotingRegressor, StackingRegressor, BaggingRegressor, AdaBoostRegressor]:
    """
    アンサンブル予測器を作成する関数。

    Args:
        ens_type (str): アンサンブルの種類。'アンサンブル', 'スタッキング', 'バギング', 'ブースティング' から選択。
        model_names (List[str]): 使用するモデルの名前のリスト。
        base_model (Optional[str]): ベースモデルの名前。スタッキング、バギング、ブースティングのときに使用。デフォルトは None。
        model_params (Optional[List[Dict[str, Any]]]): 各モデルのパラメータを指定するリスト。デフォルトは None。
        base_model_params (Optional[Dict[str, Any]]): ベースモデルのパラメータを指定する辞書。デフォルトは None。

    Returns:
        Union[VotingRegressor, StackingRegressor, BaggingRegressor, AdaBoostRegressor]: 作成されたアンサンブル予測器。
    """
    # モデルの数を取得
    n_model = len(model_names)

    # モデルのパラメータが指定されていない場合、デフォルトパラメータを使用
    if model_params is None:
        model_params = []
        for model_name in model_names:
            if task=="regression":
                _model_params = reg_default_params[model_name]
            else:
                _model_params = cls_default_params[model_name]
            model_params.append(_model_params)
    # else:
    #     # パラメータのリストがモデルの数と一致しない場合、例外を投げる
    #     if len(model_params) != n_model:
    #         raise ValueError("model_params の長さは model_names の長さと一致する必要があります。")

    if base_model:
        if base_model_params is None:
            if task=="regression":
                base_model_params = reg_default_params[base_model]
            else:
                base_model_params = cls_default_params[base_model]
       
    # 各モデルのインスタンスを作成
    predictors = [make_model(model_names[i], task, model_params[i]) for i in range(n_model)]

    # モデル名とインスタンスのリストを作成
    est_list = [('model' + str(i + 1), predictors[i]) for i in range(n_model)]

    # 指定されたアンサンブルの種類に応じてアンサンブル予測器を作成して返す
    if ens_type == 'アンサンブル':
        if task=="regression":
            return VotingRegressor(estimators=est_list)
        elif task=="classification":
            return VotingClassifier(estimators=est_list, voting='soft')
        else:
            raise ValueError("task は regression か classification で指定してください。")
    elif ens_type == 'スタッキング':
        final_estimator = make_model(base_model, task, base_model_params)
        if task=="regression":
            return StackingRegressor(
                estimators=est_list,
                final_estimator=final_estimator
            )
        elif task=="classification":
            return StackingClassifier(
                estimators=est_list,
                final_estimator=final_estimator
            )
        else:
            raise ValueError("task は regression か classification で指定してください。")
    elif ens_type == 'バギング':
        if task=="regression":
            return BaggingRegressor(
                estimator=make_model(base_model, task, base_model_params),
                n_estimators=50,
                max_samples=1.0,
                max_features=1.0,
                bootstrap=True,
                bootstrap_features=False,
                n_jobs=-1
            )
        elif task=="classification":
            return BaggingClassifier(
                estimator=make_model(base_model, task, base_model_params),
                n_estimators=50,
                max_samples=1.0,
                max_features=1.0,
                bootstrap=True,
                bootstrap_features=False,
                n_jobs=-1
            )
        else:
            raise ValueError("task は regression か classification で指定してください。")
    elif ens_type == 'ブースティング':
        if task=="regression":
            return AdaBoostRegressor(
                estimator=make_model(base_model, task, base_model_params),
                n_estimators=75,
                learning_rate=0.1
            )
        elif task=="classification":
            return AdaBoostClassifier(
                estimator=make_model(base_model, task, base_model_params),
                n_estimators=75,
                learning_rate=0.1
            )
        else:
            raise ValueError("task は regression か classification で指定してください。")

def make_predictor(
    model_names: List[str],
    ensemble: bool = False,
    ens_type: Optional[str] = None,
    base_model: Optional[str] = None,
    model_params: Optional[List[Dict[str, Any]]] = None,
    base_model_params: Optional[Dict[str, Any]] = None,
    task = "regression",
) -> RegressorMixin:
    """
    モデルまたはアンサンブル予測器を作成する関数。

    Args:
        model_names (List[str]): 使用するモデルの名前のリスト。アンサンブルの場合は複数、単体モデルの場合は1つ。
        ens_type (Optional[str]): アンサンブルの種類。'アンサンブル', 'スタッキング', 'バギング', 'ブースティング' から選択。デフォルトは None。
        base_model (Optional[str]): ベースモデルの名前。スタッキング、バギング、ブースティングのときに使用。デフォルトは None。
        model_params (Optional[List[Dict[str, Any]]): 各モデルのパラメータを指定するリスト。デフォルトは None。
        base_model_params (Optional[Dict[str, Any]]): ベースモデルのパラメータを指定する辞書。デフォルトは None。

    Returns:
        RegressorMixin: 作成されたモデルまたはアンサンブル予測器。
    """
    if task == "AD":
        model_name = model_names[0]
        model_params = ad_default_params[model_name]
        return make_model(
                    model_name,
                    task,
                    model_params
                )
    # アンサンブルタイプが指定されている場合はアンサンブル予測器を作成
    if ensemble:
        return make_ens_predictor(
            ens_type=ens_type,
            model_names=model_names,
            base_model=base_model,
            model_params=model_params,
            base_model_params=base_model_params,
            task=task
        )
    # アンサンブルタイプが指定されていない場合は単体モデルを作成
    else:
        # モデルのパラメータが指定されていない場合、デフォルトパラメータを使用
        model_name = model_names[0]
        if (model_params is None)|(model_params == [{}]):
            if task=="regression":
                model_params = reg_default_params[model_name]
            else:
                model_params = cls_default_params[model_name]
        else:
            model_params = model_params[0]
        return make_model(
                    model_name,
                    task,
                    model_params
                )
