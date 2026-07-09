import numpy as np
import pandas as pd
import optuna
from optuna.integration import OptunaSearchCV

from sklearn.base import RegressorMixin

import lightgbm as lgb

from imblearn.over_sampling import RandomOverSampler, SMOTE, SMOTENC
from imblearn.under_sampling import RandomUnderSampler
from sklearn.utils.class_weight import compute_sample_weight

from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope

from sklearn.pipeline import Pipeline

from collections import Counter
import copy
from typing import List, Optional, Union, Dict, Tuple, Callable, Any, Type
import warnings

from .utils import get_param_grid_reg, get_param_grid_cls

optuna.logging.disable_default_handler()
warnings.simplefilter('ignore')

def make_sampling_preprocess(
    y,
    method="smote",
    cat_idx=None,
    k_neighbors=5,
    random_state= 42
):
    if method == "ros":
        return RandomOverSampler(random_state=random_state)
    if method == "rus":
        return RandomUnderSampler(random_state=random_state)

    # --- ここから SMOTE 系 ---
    # y が与えられていれば fold内の最小クラス数で k を安全化
    safe_k = k_neighbors
    if y is not None:
        cnt = Counter(y)
        # 0を除き最小カウント
        min_count = min(c for c in cnt.values() if c > 0)
        if min_count < 2:
            # SMOTE不可能 → ROSへフォールバック
            return RandomOverSampler(random_state=random_state)
        safe_k = max(1, min(k_neighbors, min_count - 1))

    # SMOTENCを使うかの判定は「Noneかどうか」で
    use_smotenc = cat_idx is not None and len(cat_idx) > 0

    if use_smotenc:
        return SMOTENC(
            categorical_features=list(cat_idx),
            k_neighbors=safe_k,
            random_state=random_state
        )
    else:
        return SMOTE(k_neighbors=safe_k, random_state=random_state)
    return sampler

def fit_model(
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    model_pipeline: Pipeline,
    model_names: List[str],
    ensemble: bool,
    sampling_method=None,
    cat_index: Optional[List[int]] = None,
    cat_index_fit: Optional[List[int]] = None
) -> Pipeline:
    """
    モデルパイプラインにデータを適合させる関数。

    Args:
        X (Union[np.ndarray, pd.DataFrame]): 特徴量データ。
        y (Union[np.ndarray, pd.Series]): ターゲットデータ。
        model_pipeline (Pipeline): モデルパイプライン。
        model_names (List[str]): 使用するモデルの名前のリスト。
        ensemble (bool): アンサンブルモデルかどうかを示すフラグ。
        cat_index (Optional[List[int]]): カテゴリカル特徴量のインデックスのリスト。デフォルトは None。

    Returns:
        Pipeline: データに適合したモデルパイプライン。
    """
    fit_params: Dict[str, Any] = {}

    # LightGBMモデルの場合、カテゴリカル特徴量のインデックスを設定
    if model_names[0] == 'LightGBM' and not ensemble:
        fit_params['predictor__categorical_feature'] = cat_index_fit
        fit_params['predictor__callbacks'] = [lgb.log_evaluation(period=0)]
        
    # CatBoostモデルの場合、カテゴリカル特徴量のインデックスを設定
    elif model_names[0] == 'CatBoost' and not ensemble:
        fit_params['predictor__cat_features'] = cat_index_fit
        fit_params['predictor__verbose'] = False

    if sampling_method=="sample_weight" and model_names[0] not in ["MLPClassifier","GaussianProcessClassifier"] and not ensemble:
        fit_params['predictor__sample_weight'] = compute_sample_weight("balanced", y)

    if sampling_method in ["ros","rus","smote"]:
        sm = make_sampling_preprocess(y=y, method=sampling_method, cat_idx=cat_index, k_neighbors=5)
        X, y = sm.fit_resample(X, y)
    
    # モデルパイプラインをデータに適合させる
    if model_names[0] in ["OneClassSVM", "IsolationForest","EllipticEnvelope"]:
        model_pipeline.fit(X, **fit_params)
    else:
        model_pipeline.fit(X, y, **fit_params)
    return model_pipeline

def get_params(
    model_names: List[str],
    task="regression",
    base_model: Optional[str] = None,
    ens_type: Optional[str] = None
) -> Dict[str, Any]:
    """
    モデルまたはアンサンブル予測器のハイパーパラメータを取得する関数。

    Args:
        model_names (List[str]): 使用するモデルの名前のリスト。アンサンブルの場合は複数、単体モデルの場合は1つ。
        base_model (Optional[str]): ベースモデルの名前。スタッキング、バギング、ブースティングのときに使用。デフォルトは None。
        ens_type (Optional[str]): アンサンブルの種類。'アンサンブル', 'スタッキング', 'バギング', 'ブースティング' から選択。デフォルトは None。

    Returns:
        Dict[str, Any]: ハイパーパラメータの辞書。
    """
    if ens_type in ['アンサンブル', 'スタッキング']:
        params = {}
        # 各モデルのハイパーパラメータを取得し、名前を変更して辞書に追加
        for i, model_name in enumerate(model_names):
            if task=="regression":
                _params = get_param_grid_reg(model_name)
            elif task=="classification":
                _params = get_param_grid_cls(model_name)
            else:
                raise ValueError("task は regression か classification で指定してください。")
            _params = {k.replace('predictor', f'predictor__model{i + 1}'): v for k, v in _params.items()}
            params.update(_params)
        # スタッキングの場合、最終推定器のハイパーパラメータを取得して辞書に追加
        if ens_type == 'スタッキング':
            if task=="regression":
                _params = get_param_grid_reg(base_model)
            elif task=="classification":
                _params = get_param_grid_cls(base_model)
            else:
                raise ValueError("task は regression か classification で指定してください。")
            _params = {k.replace('predictor', 'predictor__final_estimator'): v for k, v in _params.items()}
            params.update(_params)
    elif ens_type in ['バギング', 'ブースティング']:
        # バギングまたはブースティングの場合、ベースモデルのハイパーパラメータを取得して辞書に追加
        if task=="regression":
            _params = get_param_grid_reg(base_model)
        elif task=="classification":
            _params = get_param_grid_cls(base_model)
        else:
            raise ValueError("task は regression か classification で指定してください。")
        params = {k.replace('predictor', 'predictor__estimator'): v for k, v in _params.items()}
    else:
        # 単体モデルの場合、そのモデルのハイパーパラメータを取得
        if task=="regression":
            params = get_param_grid_reg(model_names[0])
        elif task=="classification":
            params = get_param_grid_cls(model_names[0])
        else:
            raise ValueError("task は regression か classification で指定してください。")    
    return params

def tune_model(
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    model_pipeline: Pipeline,
    model_names: List[str],
    base_model: Optional[str] = None,
    ens_type: Optional[str] = None,
    sampling_method=None,
    cat_index: Optional[List[int]] = [],
    cat_index_fit: Optional[List[int]] =[],
    task="regression",
    n_trials=100,
    verbose=0
) -> Tuple[RegressorMixin, Optional[Union[Dict[str, Any], List[Dict[str, Any]]]], Optional[Dict[str, Any]]]:
    """モデルのハイパーパラメータをOptunaSearchCVでチューニングする。

    Args:
        X (Union[np.ndarray, pd.DataFrame]): 特徴量データ。
        y (Union[np.ndarray, pd.Series]): ターゲットデータ。
        model_pipeline (Pipeline): チューニング対象のモデルパイプライン。
        model_names (List[str]): 使用するモデル名のリスト。
        base_model (Optional[str]): アンサンブルのベースモデル名。
        ens_type (Optional[str]): アンサンブル種別。
        sampling_method: 分類タスクで使用するサンプリング方式。
        cat_index (Optional[List[int]]): 前処理前のカテゴリ特徴量インデックス。
        cat_index_fit (Optional[List[int]]): 学習時にモデルへ渡すカテゴリ特徴量インデックス。
        task: タスク種別。``regression`` または ``classification``。
        n_trials: Optunaの試行回数。
        verbose: OptunaSearchCVのログ出力レベル。

    Returns:
        Tuple[RegressorMixin, Optional[Union[Dict[str, Any], List[Dict[str, Any]]]], Optional[Dict[str, Any]]]:
            チューニング済みモデル、単体モデルまたは構成モデルの最良パラメータ、
            アンサンブルのベースモデル最良パラメータ。単体モデルでは第3要素はNone。
    """
    if ens_type:
        # アンサンブルモデルのパラメータを取得
        params = get_params(
            model_names=model_names,
            task=task,
            base_model=base_model,
            ens_type=ens_type
        )
    else:
        # アンサンブルモデルでない場合、そのままモデルをフィット
        params = get_params(
            model_names=model_names,
            task=task
        )

    fit_params: Dict[str, Any] = {}
    best_base_param = None

    # LightGBMモデルの場合、カテゴリカル特徴量のインデックスを設定
    if model_names[0] == 'LightGBM' and (ens_type is not None):
        fit_params['predictor__categorical_feature'] = cat_index_fit

    # CatBoostモデルの場合、カテゴリカル特徴量のインデックスを設定
    elif model_names[0] == 'CatBoost' and not (ens_type is not None):
        fit_params['predictor__cat_features'] = cat_index_fit if len(cat_index)>0 else None
        fit_params['predictor__verbose'] = False

    if sampling_method=="sample_weight" and model_names[0] not in ["MLPClassifier","GaussianProcessClassifier"] and (ens_type is not None) and task=="classification":
        fit_params['predictor__sample_weight'] = compute_sample_weight("balanced", y)
    
    if sampling_method in ["ros","rus","smote"]:
        sm = make_sampling_preprocess(y=y, method=sampling_method, cat_idx=cat_index, k_neighbors=5)
        X, y = sm.fit_resample(X, y)
    
    # OptunaSearchCVでモデルのハイパーパラメータをチューニング
    model_tune = OptunaSearchCV(
            model_pipeline,
            params,
            cv=5,
            n_jobs=-1,
            n_trials=n_trials,
            scoring='neg_root_mean_squared_error' if task=="regression" else "accuracy",
            verbose=verbose
    )
    model_tune.fit(X, y, **fit_params)
    model = model_tune.best_estimator_
    best_params_ = model_tune.best_params_
    
    if ens_type:
        if ens_type in ['スタッキング']:
            best_params = []
            for i in range(len(model_names)):
                _bp = {}
                for k in best_params_.keys():
                    if k.split("__")[1]==f"model{i+1}":
                        _bp[k.replace(f"predictor__model{i+1}__", "")] = best_params_[k]
                best_params.append(_bp)
        else:
            best_params = None
        if ens_type in ['スタッキング', 'バギング', 'ブースティング']:
            best_base_param = {}
            for k in best_params_.keys():
                if k.split("__")[1]=="final_estimator":
                    best_base_param[k.replace("predictor__final_estimator__", "")] = best_params_[k]
                elif k.split("__")[1]=="estimator":
                    best_base_param[k.replace("predictor__estimator__", "")] = best_params_[k]
        else:
            best_base_param = None
    else:
        best_params = {}
        for k in best_params_.keys():
            best_params[k.replace("predictor__", "")] = best_params_[k]        
    return model, best_params, best_base_param

def cv_fit(
    X_train,
    y_train,
    cv_model,
    cat_index,
    cat_index_fit,
    model_names,
    task,
    sampling_method=None,
    ensemble=False,
):
    # cv_model = copy.deepcopy(model)  # モデルの深いコピーを作成
    fit_params: Dict[str, Any] = {}
    
    # LightGBMモデルの場合、カテゴリカル特徴量のインデックスを設定
    if 'LightGBM' in model_names and not ensemble:
        fit_params['predictor__categorical_feature'] = cat_index_fit

    # CatBoostモデルの場合、カテゴリカル特徴量のインデックスを設定
    elif 'CatBoost' in model_names and not ensemble:
        fit_params['predictor__cat_features'] = cat_index_fit
        fit_params['predictor__verbose'] = False

    if sampling_method=="sample_weight" and model_names[0] not in ["MLPClassifier","GaussianProcessClassifier"] and not ensemble and task=="classification":
        fit_params['predictor__sample_weight'] = compute_sample_weight("balanced", y_train)
    
    if sampling_method in ["ros","rus","smote"]:
        sm = make_sampling_preprocess(y=y_train, method=sampling_method, cat_idx=cat_index, k_neighbors=5)
        X_train, y_train = sm.fit_resample(X_train, y_train)
    
    cv_model.fit(X_train, y_train, **fit_params)
    return cv_model