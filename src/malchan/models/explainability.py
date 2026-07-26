import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.linear_model import (
    LinearRegression, HuberRegressor, Ridge, Lasso, ElasticNet, TweedieRegressor,
    BayesianRidge, ARDRegression, PassiveAggressiveRegressor, LassoLars,
    OrthogonalMatchingPursuit, LogisticRegression, RidgeClassifier,
    PassiveAggressiveClassifier,
)
from sklearn.cross_decomposition import PLSRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor,
    RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier,
    IsolationForest,
)
from sklearn.neural_network import MLPRegressor, MLPClassifier
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.svm import SVR, SVC, OneClassSVM
from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.inspection import permutation_importance
from sklearn.covariance import EllipticEnvelope
import shap
from typing import List, Optional, Union, Dict, Tuple, Any
import warnings

warnings.simplefilter('ignore')


def get_pd_and_ice(
    X: pd.DataFrame,
    _model: RegressorMixin,
    target: str,
    unique_dict: Optional[dict] = {},
    bounds=None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    部分依存プロット（Partial Dependence Plot）および個別条件期待値（ICE）を取得する関数。

    Args:
        X (pd.DataFrame): 特徴量データ。
        model (RegressorMixin): モデルオブジェクト（回帰器）。
        target (str): 部分依存プロットを作成する特徴量の名前。

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - 部分依存プロットデータ（各サンプルのICEを含む）。
            - 特徴量の値（x軸）。
    """
    if target in unique_dict.keys():
        _col_names = [c for c in X.columns if target in c]
        target_idx = np.where(np.isin(X.columns.to_numpy(), _col_names))[0]
        xticks = np.array(unique_dict[target])
    else:
        target_idx = np.where(np.array(X.columns == target))[0][0]
        if bounds is not None:
            xticks = np.linspace(bounds[0], bounds[1], 200)
        else:
            xticks = np.linspace(X[target].min(), X[target].max(), 200)

    if X.shape[0] > 300:
        X_sample = X.sample(300)
    else:
        X_sample = X

    X_PD = []
    for i in range(len(X_sample)):
        Xi = X_sample.values[[i], :]
        Xi = np.repeat(Xi, len(xticks), axis=0)
        Xi[:, target_idx] = np.array(xticks).reshape(-1, 1) if target in unique_dict.keys() else np.array(xticks)
        Xi = pd.DataFrame(Xi, columns=X.columns)
        _x = _model.predict(Xi, proba=_model.task == "classification").values
        if _x.shape[-1] == 1:
            _x = _x.reshape(-1, 1)
        else:
            _x = _x.reshape(-1, 1, _x.shape[-1])
        X_PD.append(_x)

    X_PD = np.concatenate(X_PD, axis=1)
    return X_PD, xticks


def get_pd_and_ice_2d(
    X: pd.DataFrame,
    _model: RegressorMixin,
    targets: List[str],
    unique_dict: Optional[Dict[str, Any]] = None,
    bounds=None,
    n_grid: int = 75,
    max_samples: int = 300,
    random_state: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if unique_dict is None:
        unique_dict = {}
    if len(targets) != 2:
        raise ValueError(f"targets must be length 2, got {len(targets)}: {targets}")

    metas = []
    for i, t in enumerate(targets):
        if t in unique_dict:
            onehot_cols = [c for c in X.columns if str(c).startswith(f"{t}_")]

            if len(onehot_cols) >= 2:
                idx = np.where(np.isin(X.columns.to_numpy(), onehot_cols))[0]
                cats = list(unique_dict[t])
                metas.append({"name": t, "kind": "onehot", "idx": idx, "cols": onehot_cols, "ticks": cats})
            else:
                if t not in X.columns:
                    raise ValueError(f"unique_dict has key '{t}', but '{t}' not in X.columns.")
                idx = int(np.where(X.columns.to_numpy() == t)[0][0])
                cats = list(unique_dict[t])
                metas.append({"name": t, "kind": "categorical", "idx": idx, "ticks": cats})
        else:
            if t not in X.columns:
                raise ValueError(f"'{t}' not in X.columns.")
            idx = int(np.where(X.columns.to_numpy() == t)[0][0])
            if bounds is not None:
                lo, hi = bounds[i][0], bounds[i][1]
            else:
                lo, hi = float(X[t].min()), float(X[t].max())
            ticks = np.linspace(lo, hi, n_grid)
            metas.append({"name": t, "kind": "numeric", "idx": idx, "ticks": ticks})

    ticks0 = metas[0]["ticks"]
    ticks1 = metas[1]["ticks"]
    g0, g1 = np.meshgrid(ticks0, ticks1, indexing="xy")
    g0 = g0.ravel()
    g1 = g1.ravel()
    n_grid_total = g0.shape[0]

    if X.shape[0] > max_samples:
        X_sample = X.sample(max_samples, random_state=random_state)
    else:
        X_sample = X

    def _apply_onehot(Xi: np.ndarray, base_name: str, cols: List[str], cats: List[Any], chosen: np.ndarray) -> None:
        Xi[:, np.where(np.isin(X.columns.to_numpy(), cols))[0]] = 0.0

        col_set = set(cols)
        cat_to_col = {}
        for cat in cats:
            c = f"{base_name}_{cat}"
            if c in col_set:
                cat_to_col[cat] = int(np.where(X.columns.to_numpy() == c)[0][0])

        if len(cat_to_col) == 0 and len(cats) == len(cols):
            for cat, col in zip(cats, cols):
                cat_to_col[cat] = int(np.where(X.columns.to_numpy() == col)[0][0])

        if len(cat_to_col) == 0:
            raise ValueError(
                f"Cannot map categories to one-hot columns for '{base_name}'. "
                f"cats={cats}, onehot_cols={cols[:5]}..."
            )

        for cat, abs_col in cat_to_col.items():
            mask = chosen == cat
            if np.any(mask):
                Xi[mask, abs_col] = 1.0

    X_PD_list = []
    for i in range(len(X_sample)):
        Xi = np.repeat(X_sample.iloc[[i]].to_numpy(), n_grid_total, axis=0)

        m0 = metas[0]
        if m0["kind"] == "numeric":
            Xi[:, m0["idx"]] = g0.astype(float)
        elif m0["kind"] == "categorical":
            Xi[:, m0["idx"]] = g0
        else:
            _apply_onehot(Xi, m0["name"], m0["cols"], m0["ticks"], g0)

        m1 = metas[1]
        if m1["kind"] == "numeric":
            Xi[:, m1["idx"]] = g1.astype(float)
        elif m1["kind"] == "categorical":
            Xi[:, m1["idx"]] = g1
        else:
            _apply_onehot(Xi, m1["name"], m1["cols"], m1["ticks"], g1)

        Xi_df = pd.DataFrame(Xi, columns=X.columns)

        pred = _model.predict(Xi_df, proba=getattr(_model, "task", "") == "classification")
        pred = np.asarray(getattr(pred, "values", pred))

        if pred.ndim == 1:
            pred = pred.reshape(-1, 1)
        elif pred.ndim == 2 and pred.shape[1] >= 2:
            pred = pred.reshape(-1, 1, pred.shape[1])
        else:
            pred = pred.reshape(-1, 1)

        X_PD_list.append(pred)

    X_PD = np.concatenate(X_PD_list, axis=1)
    return X_PD, g0, g1


def get_shap_values(
    model: RegressorMixin,
    X: pd.DataFrame
) -> Tuple[Union[np.ndarray, None], Union[shap.Explainer, None]]:
    """
    SHAP (SHapley Additive exPlanations) 値を計算する関数。モデルの種類に応じて適切なエクスプレイナーを使用します。
    """
    if len(X) > 300:
        X_sample = shap.utils.sample(X, 300)
    else:
        X_sample = X
    if isinstance(model, (DecisionTreeRegressor, RandomForestRegressor, ExtraTreesRegressor,
                          GradientBoostingRegressor, XGBRegressor, LGBMRegressor)):
        explainer = shap.TreeExplainer(model, X_sample, feature_names=X.columns)
    elif isinstance(model, (DecisionTreeClassifier, RandomForestClassifier, ExtraTreesClassifier,
                            GradientBoostingClassifier, XGBClassifier, LGBMClassifier)):
        explainer = shap.Explainer(model.predict_proba, X_sample, feature_names=X.columns)
    elif isinstance(model, (CatBoostRegressor, CatBoostClassifier)):
        explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
    elif isinstance(model, (LinearRegression, Ridge, Lasso, ElasticNet, TweedieRegressor, HuberRegressor,
                            PLSRegression, BayesianRidge, ARDRegression, PassiveAggressiveRegressor, LassoLars, OrthogonalMatchingPursuit)):
        explainer = shap.LinearExplainer(model, X_sample, feature_names=X.columns)
    elif isinstance(model, (GaussianProcessRegressor, SVR, MLPRegressor, OneClassSVM, IsolationForest)):
        explainer = shap.Explainer(model.predict, X_sample)
    elif isinstance(model, (GaussianProcessClassifier, SVC, MLPClassifier,
                            LogisticRegression, RidgeClassifier, PassiveAggressiveClassifier, GaussianNB)):
        explainer = shap.Explainer(model.predict_proba, X_sample)
    elif isinstance(model, (OneClassSVM, EllipticEnvelope, IsolationForest)):
        explainer = shap.Explainer(model.decision_function, X_sample)
    else:
        return None, None, None, None

    num_features_for_explainer = X_sample.shape[1]
    required_max_evals_for_this_explainer = 0

    if isinstance(explainer, shap.explainers._permutation.PermutationExplainer):
        required_max_evals_for_this_explainer = 2 * num_features_for_explainer + 1
        required_max_evals_for_this_explainer = max(required_max_evals_for_this_explainer, 1000)
    elif isinstance(explainer, shap.explainers._exact.ExactExplainer):
        if num_features_for_explainer > 15:
            print(f"Warning: ExactExplainer with {num_features_for_explainer} features will be extremely slow. Consider switching explainer type or reducing features.")
        required_max_evals_for_this_explainer = 2 ** num_features_for_explainer
    elif isinstance(explainer, shap.explainers._kernel.KernelExplainer):
        required_max_evals_for_this_explainer = max(2 * num_features_for_explainer + 1, 1000)
    else:
        required_max_evals_for_this_explainer = max(2 * num_features_for_explainer + 1, 1000)

    try:
        shap_values = explainer(X_sample, check_additivity=False)
    except TypeError:
        shap_values = explainer(X_sample, max_evals=required_max_evals_for_this_explainer)

    return shap_values.values, shap_values.base_values, explainer, X_sample


def get_shap_scatter(
    X: pd.DataFrame,
    shap_values: np.ndarray,
    target_col: str,
    modelname: Optional[str] = None,
    unique_dict: Dict[str, list] = {},
    smiles_cols=None,
    comp_cols=None,
    le=None
) -> pd.DataFrame:
    """SHAP値を用いて、ターゲット列に対する散布図データを取得する。"""
    smiles_cols = [] or smiles_cols
    comp_cols = [] or comp_cols
    if target_col in unique_dict.keys():
        _col_names = [c for c in X.columns if target_col in c]
        _col_idx = np.where(np.isin(X.columns.to_numpy(), _col_names))[0]

        X_plot = X[_col_names].copy()
        comb_cols = [f'{target_col}_{c}' for c in unique_dict[target_col]]
        X_plot['cat'] = comb_cols[0]

        if modelname in ['LightGBM']:
            for i in range(1, len(comb_cols)):
                X_plot.loc[X_plot[target_col] == i, 'cat'] = comb_cols[i]
        elif (modelname in ['CatBoost']) | (target_col in smiles_cols) | (target_col in comp_cols):
            for i in range(1, len(comb_cols)):
                X_plot.loc[X_plot[target_col] == unique_dict[target_col][i], 'cat'] = comb_cols[i]
        else:
            for i in range(1, len(comb_cols)):
                X_plot.loc[X_plot[comb_cols[i]] == 1, 'cat'] = comb_cols[i]

        X_plot = X_plot[['cat']].rename(columns={'cat': target_col})
        if len(shap_values.shape) == 2:
            X_plot['shap'] = shap_values[:, _col_idx].sum(axis=1)
        else:
            X_plot[['shap_' + str(c) for c in le.inverse_transform(np.arange(shap_values.shape[-1]))]] = shap_values[:, _col_idx].sum(axis=1)
        X_plot = X_plot.sort_values(target_col)
    else:
        shap_col_idx = np.where(np.array(X.columns) == target_col)[0][0]
        X_plot = X[[target_col]].copy()
        if len(shap_values.shape) == 2:
            X_plot['shap'] = shap_values[:, shap_col_idx]
        else:
            X_plot[['shap_' + str(c) for c in le.inverse_transform(np.arange(shap_values.shape[-1]))]] = shap_values[:, shap_col_idx]

    return X_plot


def get_importances(model: RegressorMixin) -> np.ndarray:
    """与えられたモデルの特徴量重要度を取得する。"""
    if isinstance(model, (LinearRegression, Ridge, Lasso, ElasticNet, TweedieRegressor, HuberRegressor,
                          BayesianRidge, ARDRegression, PassiveAggressiveRegressor, LassoLars, OrthogonalMatchingPursuit,
                          RidgeClassifier, PassiveAggressiveClassifier)):
        feature_importances = model.coef_
        if type(feature_importances[0]) == list:
            feature_importances = feature_importances[0]
        elif len(feature_importances.shape) == 2:
            feature_importances = feature_importances[0]
    elif isinstance(model, (PLSRegression, LogisticRegression)):
        feature_importances = model.coef_[0]
    elif isinstance(model, GaussianNB):
        feature_importances = model.theta_.mean(axis=0)
    elif isinstance(model, (DecisionTreeRegressor, RandomForestRegressor, ExtraTreesRegressor,
                            GradientBoostingRegressor, XGBRegressor, LGBMRegressor, CatBoostRegressor,
                            DecisionTreeClassifier, RandomForestClassifier, ExtraTreesClassifier,
                            GradientBoostingClassifier, XGBClassifier, LGBMClassifier, CatBoostClassifier)):
        feature_importances = model.feature_importances_
    else:
        warnings.warn(f"Model type {type(model).__name__} is not supported for feature importance extraction.")
        return None

    return np.array(feature_importances)


def get_pfi_values(
    model: RegressorMixin,
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.DataFrame, np.ndarray]
) -> np.ndarray:
    """パーミュテーションインポータンスを計算する。"""
    if isinstance(X, pd.DataFrame):
        X = X.values
    if isinstance(y, pd.DataFrame):
        y = y.values.ravel()

    if X.shape[0] > 300:
        max_samples = 300 / X.shape[0]
    else:
        max_samples = 1.0

    pfi_result = permutation_importance(
        model,
        X,
        y,
        max_samples=max_samples,
        n_jobs=-1,
        random_state=0
    )
    return pfi_result.importances_mean


def get_feature_importances(
    model: RegressorMixin,
    importance_type: str = 'model',
    n_features: int = 15,
    shap_values: Optional[np.ndarray] = None,
    X: Optional[Union[pd.DataFrame, np.ndarray]] = None,
    y: Optional[Union[pd.DataFrame, np.ndarray]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """特徴量重要度と上位特徴量のインデックスを取得する。"""
    if importance_type == 'model':
        feature_importances = get_importances(model)
    elif importance_type == 'shap':
        if shap_values is None:
            raise ValueError("shap_values must be provided for SHAP importance.")
        feature_importances = np.sqrt((shap_values ** 2).sum(axis=0))
    elif importance_type == 'pfi':
        if X is None or y is None:
            raise ValueError("X and y must be provided for permutation importance.")
        feature_importances = get_pfi_values(model, X, y)
    else:
        raise ValueError(f"Unknown importance_type: {importance_type}")

    importance_idx = np.argsort(-np.abs(feature_importances))[:n_features]
    return feature_importances, importance_idx
