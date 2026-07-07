import numpy as np
import pandas as pd
import optuna
from optuna.distributions import IntDistribution, FloatDistribution, CategoricalDistribution
from optuna.integration import OptunaSearchCV

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as imbPipeline
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, MaxAbsScaler, PolynomialFeatures,
    FunctionTransformer, OrdinalEncoder, OneHotEncoder, LabelEncoder
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_union

from rdkit import Chem
from skfp.fingerprints import ECFPFingerprint, MACCSFingerprint, RDKit2DDescriptorsFingerprint, MordredFingerprint, PubChemFingerprint, AtomPairFingerprint
from types import SimpleNamespace
from pathlib import Path
import shutil
from xenonpy._conf import __cfg_root__
from xenonpy.datatools import preset
from xenonpy.descriptor import Compositions
from pymatgen.core import Composition as PMGComposition
from pymatgen.core import Composition as PMGComposition
from matminer.featurizers.composition import (
    ElementProperty, OxidationStates, ValenceOrbital, AtomicOrbitals, YangSolidSolution,
    IonProperty, TMetalFraction, Stoichiometry, ElementFraction, Meredig
)

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.impute import KNNImputer

from sklearn.decomposition import PCA

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

from sklearn.inspection import permutation_importance

import shap
from itertools import combinations

from imblearn.over_sampling import RandomOverSampler, SMOTE, SMOTENC
from imblearn.under_sampling import RandomUnderSampler
from sklearn.utils.class_weight import compute_sample_weight

from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope

from collections import Counter
import copy
from typing import List, Optional, Union, Dict, Tuple, Callable, Any, Type
import warnings

optuna.logging.disable_default_handler()
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
        # ターゲット列に関連する特徴量の列名とインデックスを取得
        _col_names = [c for c in X.columns if target in c]
        target_idx = np.where(np.isin(X.columns.to_numpy(), _col_names))[0]
        # comb_cols = [f'{target}_{c}' for c in unique_dict[target]]
        xticks = np.array(unique_dict[target])
    
    else:
        # 対象特徴量のインデックスを取得
        target_idx = np.where(np.array(X.columns == target))[0][0]
        
        # x 軸の範囲を設定
        if bounds is not None:
            xticks = np.linspace(bounds[0], bounds[1], 200)
        else:
            xticks = np.linspace(X[target].min(), X[target].max(), 200)

    if X.shape[0]>300:
        X_sample = X.sample(300)
    else:
        X_sample = X
    
    X_PD = []
    for i in range(len(X_sample)):
        # 各サンプルのデータを取得し、xticks の長さに繰り返す
        Xi = X_sample.values[[i], :]
        Xi = np.repeat(Xi, len(xticks), axis=0)
        # 対象特徴量の値を xticks に設定
        Xi[:, target_idx] = np.array(xticks).reshape(-1,1) if target in unique_dict.keys() else np.array(xticks)
        Xi = pd.DataFrame(Xi, columns=X.columns)
        # モデルの予測結果を保存
        _x = _model.predict(Xi, proba=_model.task=="classification").values
        if _x.shape[-1]==1:
            _x=_x.reshape(-1, 1)
        else:
            _x=_x.reshape(-1, 1, _x.shape[-1])
        X_PD.append(_x)
    
    # 部分依存プロットデータを結合
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
            # まずは one-hot (prefix) を探す
            onehot_cols = [c for c in X.columns if str(c).startswith(f"{t}_")]

            if len(onehot_cols) >= 2:
                idx = np.where(np.isin(X.columns.to_numpy(), onehot_cols))[0]
                cats = list(unique_dict[t])  # 例: ["a","b","c"]
                metas.append({"name": t, "kind": "onehot", "idx": idx, "cols": onehot_cols, "ticks": cats})
            else:
                # one-hot ではない（single categorical / ordinal）
                if t not in X.columns:
                    raise ValueError(f"unique_dict has key '{t}', but '{t}' not in X.columns.")
                idx = int(np.where(X.columns.to_numpy() == t)[0][0])
                cats = list(unique_dict[t])
                metas.append({"name": t, "kind": "categorical", "idx": idx, "ticks": cats})
        else:
            # numeric
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

    # sampling
    if X.shape[0] > max_samples:
        X_sample = X.sample(max_samples, random_state=random_state)
    else:
        X_sample = X

    def _apply_onehot(Xi: np.ndarray, base_name: str, cols: List[str], cats: List[Any], chosen: np.ndarray) -> None:
        # 0クリア
        Xi[:, np.where(np.isin(X.columns.to_numpy(), cols))[0]] = 0.0

        col_set = set(cols)
        cat_to_col = {}
        for cat in cats:
            c = f"{base_name}_{cat}"
            if c in col_set:
                cat_to_col[cat] = int(np.where(X.columns.to_numpy() == c)[0][0])

        # fallback: catsとcolsが同数なら順番対応
        if len(cat_to_col) == 0 and len(cats) == len(cols):
            for cat, col in zip(cats, cols):
                cat_to_col[cat] = int(np.where(X.columns.to_numpy() == col)[0][0])

        if len(cat_to_col) == 0:
            raise ValueError(
                f"Cannot map categories to one-hot columns for '{base_name}'. "
                f"cats={cats}, onehot_cols={cols[:5]}..."
            )

        for cat, abs_col in cat_to_col.items():
            mask = (chosen == cat)
            if np.any(mask):
                Xi[mask, abs_col] = 1.0

    X_PD_list = []
    for i in range(len(X_sample)):
        Xi = np.repeat(X_sample.iloc[[i]].to_numpy(), n_grid_total, axis=0)

        # axis0
        m0 = metas[0]
        if m0["kind"] == "numeric":
            Xi[:, m0["idx"]] = g0.astype(float)
        elif m0["kind"] == "categorical":
            Xi[:, m0["idx"]] = g0  # そのまま代入（文字列/カテゴリ/数値コード）
        else:  # onehot
            _apply_onehot(Xi, m0["name"], m0["cols"], m0["ticks"], g0)

        # axis1
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
    
# def get_pd_and_ice_2d(
#     X: pd.DataFrame,
#     _model: RegressorMixin,
#     targets: List[str],
#     unique_dict: Optional[dict] = {},
#     bounds=None
# ) -> Tuple[np.ndarray, np.ndarray]:
#     """
#     部分依存プロット（Partial Dependence Plot）および個別条件期待値（ICE）を取得する関数。

#     Args:
#         X (pd.DataFrame): 特徴量データ。
#         model (RegressorMixin): モデルオブジェクト（回帰器）。
#         target (str): 部分依存プロットを作成する特徴量の名前。

#     Returns:
#         Tuple[np.ndarray, np.ndarray]:
#             - 部分依存プロットデータ（各サンプルのICEを含む）。
#             - 特徴量の値（x軸）。
#     """
#     target_idx_list = []
#     xticks_list = []
#     for i, target in enumerate(targets):
#         if target in unique_dict.keys():
#             # ターゲット列に関連する特徴量の列名とインデックスを取得
#             _col_names = [c for c in X.columns if target in c]
#             target_idx = np.where(np.isin(X.columns.to_numpy(), _col_names))[0]
#             # comb_cols = [f'{target}_{c}' for c in unique_dict[target]]
#             xticks = unique_dict[target]
        
#         else:
#             # 対象特徴量のインデックスを取得
#             target_idx = np.where(np.array(X.columns == target))[0][0]
            
#             # x 軸の範囲を設定
#             if bounds is not None:
#                 xticks = np.linspace(bounds[i][0], bounds[i][1], 75)
#             else:
#                 xticks = np.linspace(X[target].min(), X[target].max(), 75)

#         target_idx_list.append(target_idx)
#         xticks_list.append(xticks)

#     xticks1, xticks2 = np.meshgrid(xticks_list[0], xticks_list[1])
#     xticks1 = xticks1.ravel()
#     xticks2 = xticks2.ravel()

#     if X.shape[0]>300:
#         X_sample = X.sample(300)
#     else:
#         X_sample = X
    
#     X_PD = []
#     for i in range(len(X_sample)):
#         # 各サンプルのデータを取得し、xticks の長さに繰り返す
#         Xi = X_sample.values[[i], :]
#         Xi = np.repeat(Xi, len(xticks1), axis=0)

#         # 対象特徴量の値を xticks に設定
#         Xi[:, target_idx_list[0]] = np.array(xticks1).reshape(-1,1) if target in unique_dict.keys() else np.array(xticks1)
#         Xi[:, target_idx_list[1]] = np.array(xticks2).reshape(-1,1) if target in unique_dict.keys() else np.array(xticks2)
#         Xi = pd.DataFrame(Xi, columns=X.columns)
#         # モデルの予測結果を保存
#         _x = _model.predict(Xi, proba=_model.task=="classification").values
#         if _x.shape[-1]==1:
#             _x=_x.reshape(-1, 1)
#         else:
#             _x=_x.reshape(-1, 1, _x.shape[-1])
#         X_PD.append(_x)
    
#     # 部分依存プロットデータを結合
#     X_PD = np.concatenate(X_PD, axis=1)
#     return X_PD, xticks1, xticks2

def get_shap_values(
    model: RegressorMixin,
    X: pd.DataFrame
) -> Tuple[Union[np.ndarray, None], Union[shap.Explainer, None]]:
    """
    SHAP (SHapley Additive exPlanations) 値を計算する関数。モデルの種類に応じて適切なエクスプレイナーを使用します。

    Args:
        model (RegressorMixin): モデルオブジェクト（回帰器）。
        X (pd.DataFrame): 特徴量データ。

    Returns:
        Tuple[Union[np.ndarray, None], Union[shap.Explainer, None]]:
            - SHAP 値の配列または None。
            - SHAP エクスプレイナーオブジェクトまたは None。
    """
    if len(X)>300:
        X_sample = shap.utils.sample(X, 300)
    else:
        X_sample = X
    # モデルの種類に応じた SHAP エクスプレイナーの選択
    if isinstance(model, (DecisionTreeRegressor, RandomForestRegressor, ExtraTreesRegressor,
                          GradientBoostingRegressor, XGBRegressor, LGBMRegressor)):
        explainer = shap.TreeExplainer(model, X_sample, feature_names=X.columns)
    elif isinstance(model, (DecisionTreeClassifier, RandomForestClassifier, ExtraTreesClassifier,
                          GradientBoostingClassifier, XGBClassifier, LGBMClassifier)):
        explainer = shap.Explainer(model.predict_proba, X_sample, feature_names=X.columns)
    elif isinstance(model, (CatBoostRegressor, CatBoostClassifier)):
        # explainer = shap.TreeExplainer(model, X_sample, feature_names=X.columns, feature_perturbation="interventional")
        explainer = shap.TreeExplainer(model,feature_perturbation="tree_path_dependent")
    elif isinstance(model, (LinearRegression, Ridge, Lasso, ElasticNet, TweedieRegressor, HuberRegressor,
                            PLSRegression, BayesianRidge, ARDRegression, PassiveAggressiveRegressor, LassoLars, OrthogonalMatchingPursuit,
                            )):
        explainer = shap.LinearExplainer(model, X_sample, feature_names=X.columns)
    elif isinstance(model, (GaussianProcessRegressor, SVR, MLPRegressor,OneClassSVM,IsolationForest)):
        explainer = shap.Explainer(model.predict, X_sample)
    elif isinstance(model, (GaussianProcessClassifier, SVC, MLPClassifier,
                            LogisticRegression, RidgeClassifier, PassiveAggressiveClassifier, GaussianNB)):
        explainer = shap.Explainer(model.predict_proba, X_sample)
    elif isinstance(model, (OneClassSVM, EllipticEnvelope, IsolationForest)):
        explainer = shap.Explainer(model.decision_function, X_sample)
    else:
        return None, None, None, None

    # explainerに渡されるX_sampleの特徴量数を取得
    num_features_for_explainer = X_sample.shape[1]

    # explainerのタイプに応じてrequired_max_evalsを計算
    required_max_evals_for_this_explainer = 0
    current_explainer_type = type(explainer)

    if isinstance(explainer, shap.explainers._permutation.PermutationExplainer):
        required_max_evals_for_this_explainer = 2 * num_features_for_explainer + 1
        # PermutationExplainerの場合、計算負荷が高くなりすぎないように、
        # ある程度の閾値を設けて、max_evalsを増やすことも検討できる
        required_max_evals_for_this_explainer = max(required_max_evals_for_this_explainer, 1000)

    elif isinstance(explainer, shap.explainers._exact.ExactExplainer):
        # ExactExplainerは特徴量が多いと計算コストが指数関数的に増大します。
        # num_features が10程度であれば 2**10 = 1024 で許容範囲ですが、
        # もし num_features が例えば20を超えると 2**20 = 1,048,576 となり現実的ではなくなります。
        # このExplainerが選択された場合のnum_featuresが常に少ないことを確認してください。
        if num_features_for_explainer > 15: # 2**15 = 32768。これ以上は通常非現実的
             print(f"Warning: ExactExplainer with {num_features_for_explainer} features will be extremely slow. Consider switching explainer type or reducing features.")
             # ここで強制的にPermutationExplainerに切り替えるなどのロジックも考えられますが、
             # 現在はexplainerがExactExplainerであることを前提としています。
        required_max_evals_for_this_explainer = 2**num_features_for_explainer

    elif isinstance(explainer, shap.explainers._kernel.KernelExplainer):
        # KernelExplainerのデフォルトのn_samplesは通常500。
        # より良い近似を得るには、 num_features * 2048 + 2 などが必要な場合もあります。
        # ここでは、もしエラーが出るならその時に対応する。
        # とりあえずPermutationExplainerのロジックと同等か、少し多めに設定
        required_max_evals_for_this_explainer = max(2 * num_features_for_explainer + 1, 1000)

    else:
        # TreeExplainer, LinearExplainerなど、他のExplainerの場合
        # これらは通常、max_evalsの要件が異なるか、デフォルトで十分な場合が多い。
        # 問題ない可能性が高いが、念のためPermutationExplainerのロジックで大きめに確保
        required_max_evals_for_this_explainer = max(2 * num_features_for_explainer + 1, 1000)


    # explainerの呼び出し部分
    try:
        # 'check_additivity' 引数を渡す (一部ExplainerではTypeErrorを発生させるためtry-exceptで囲む)
        shap_values = explainer(X_sample, check_additivity=False,
                                # max_evals=required_max_evals_for_this_explainer
                               )
    except TypeError:
        # 'check_additivity' がサポートされていないExplainerの場合
        shap_values = explainer(X_sample, max_evals=required_max_evals_for_this_explainer)

    return shap_values.values, shap_values.base_values, explainer, X_sample

    # # SHAP 値の計算
    # num_features = X_sample.shape[1]
    # required_max_evals = 2 * num_features + 1
    # try:
    #     shap_values = explainer(X_sample, check_additivity=False, max_evals=max(required_max_evals, 700))
    # except TypeError:
    #     shap_values = explainer(X_sample, max_evals=max(required_max_evals, 700))
    # return shap_values.values, shap_values.base_values, explainer, X_sample

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
    """
    SHAP値を用いて、ターゲット列に対する散布図データを取得する関数。

    Args:
        X (pd.DataFrame): 特徴量データ。
        shap_values (np.ndarray): SHAP値（サンプル数 x 特徴量数）。
        target_col (str): ターゲット列の名前。
        modelname (Optional[str]): モデルの名前（デフォルトは None）。
        unique_dict (Dict[str, list]): 特徴量のユニーク値に関する辞書（デフォルトは空辞書）。

    Returns:
        pd.DataFrame: 散布図データ（ターゲット列とSHAP値）。
    """
    smiles_cols = [] or smiles_cols
    comp_cols = [] or comp_cols
    if target_col in unique_dict.keys():
        # ターゲット列に関連する特徴量の列名とインデックスを取得
        _col_names = [c for c in X.columns if target_col in c]
        _col_idx = np.where(np.isin(X.columns.to_numpy(), _col_names))[0]

        # プロット用データを作成
        X_plot = X[_col_names].copy()
        comb_cols = [f'{target_col}_{c}' for c in unique_dict[target_col]]
        X_plot['cat'] = comb_cols[0]

        if modelname in ['LightGBM']:
            for i in range(1, len(comb_cols)):
                X_plot.loc[X_plot[target_col] == i, 'cat'] = comb_cols[i]
        elif (modelname in ['CatBoost'])|(target_col in smiles_cols)|(target_col in comp_cols):
            for i in range(1, len(comb_cols)):
                X_plot.loc[X_plot[target_col] == unique_dict[target_col][i], 'cat'] = comb_cols[i]
        else:
            for i in range(1, len(comb_cols)):
                X_plot.loc[X_plot[comb_cols[i]] == 1, 'cat'] = comb_cols[i]

        # 結果をターゲット列名に変更
        X_plot = X_plot[['cat']].rename(columns={'cat': target_col})
        # SHAP値を追加
        if len(shap_values.shape)==2:
            X_plot['shap'] = shap_values[:, _col_idx].sum(axis=1)
        else:
            X_plot[['shap'+"_"+str(c) for c in le.inverse_transform(np.arange(shap_values.shape[-1]))]] = shap_values[:, _col_idx].sum(axis=1)
        # X_plot['shap'] = shap_values[:, _col_idx].sum(axis=1)
        X_plot = X_plot.sort_values(target_col)
    else:
        # 単一のターゲット列の場合
        shap_col_idx = np.where(np.array(X.columns) == target_col)[0][0]
        X_plot = X[[target_col]].copy()
        if len(shap_values.shape)==2:
            X_plot['shap'] = shap_values[:, shap_col_idx]
        else:
            X_plot[['shap'+"_"+str(c) for c in le.inverse_transform(np.arange(shap_values.shape[-1]))]] = shap_values[:, shap_col_idx]

    return X_plot

def get_importances(
    model: RegressorMixin
) -> np.ndarray:
    """
    与えられた回帰モデルの特徴量重要度を取得する関数。モデルの種類に応じて適切な方法で重要度を取得します。

    Args:
        model (RegressorMixin): 特徴量の重要度を取得したい回帰モデルオブジェクト。

    Returns:
        np.ndarray: 特徴量の重要度を示す配列。対応していないモデルの場合は空の配列を返します。
    """
    
    # 線形モデルの特徴量重要度の取得
    if isinstance(model, (LinearRegression, Ridge, Lasso, ElasticNet, TweedieRegressor, HuberRegressor,
                          BayesianRidge, ARDRegression, PassiveAggressiveRegressor, LassoLars, OrthogonalMatchingPursuit,
                          RidgeClassifier, PassiveAggressiveClassifier)):
        feature_importances = model.coef_
        if type(feature_importances[0])==list:
            feature_importances = feature_importances[0]
        elif len(feature_importances.shape)==2:
            feature_importances = feature_importances[0]
            
            
    # PLS回帰モデルの特徴量重要度の取得（最初の成分）
    elif isinstance(model, (PLSRegression, LogisticRegression)):
        feature_importances = model.coef_[0]

    elif isinstance(model, GaussianNB):
        feature_importances = model.theta_.mean(axis=0)
        
    # 決定木ベースのモデルの特徴量重要度の取得
    elif isinstance(model, (DecisionTreeRegressor, RandomForestRegressor, ExtraTreesRegressor,
                            GradientBoostingRegressor, XGBRegressor, LGBMRegressor, CatBoostRegressor,
                            DecisionTreeClassifier, RandomForestClassifier, ExtraTreesClassifier,
                            GradientBoostingClassifier, XGBClassifier, LGBMClassifier, CatBoostClassifier)):
        feature_importances = model.feature_importances_
    
    # サポートしていないモデルに対する処理
    else:
        warnings.warn(f"Model type {type(model).__name__} is not supported for feature importance extraction.")
        return None  # Noneを返す
    
    # 重要度を配列として返す
    return np.array(feature_importances)

def get_pfi_values(
    model: RegressorMixin,
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.DataFrame, np.ndarray]
) -> np.ndarray:
    """
    パーミュテーションインポータンスを用いて特徴量の重要度を計算する関数。
    パーミュテーションインポータンスは、モデルの予測精度に対する各特徴量の影響を評価します。

    Args:
        model (RegressorMixin): 回帰モデルオブジェクト。
        X (Union[pd.DataFrame, np.ndarray]): 特徴量データ。
        y (Union[pd.DataFrame, np.ndarray]): 目的変数データ。

    Returns:
        np.ndarray: 各特徴量に対応するパーミュテーションインポータンスの平均値配列。
    """
    # X と y が pd.DataFrame なら np.array に変換
    if isinstance(X, pd.DataFrame):
        X = X.values
    if isinstance(y, pd.DataFrame):
        y = y.values.ravel()  # 目的変数が 2D の場合も対応するために flatten

    if X.shape[0]>300:
        max_samples = 300 / X.shape[0]
    else:
        max_samples = 1.0
    
    # パーミュテーションインポータンスの計算
    pfi_result = permutation_importance(
        model,    # 使用するモデル
        X, # 特徴量
        y, # 目的変数
        max_samples=max_samples,
        n_jobs=-1,        # 並列処理を使用して計算を高速化
        random_state=0    # 再現性のためランダムシードを設定
    )
    
    # パーミュテーションインポータンスの平均値を取得
    pfi_values = pfi_result.importances_mean
    
    return pfi_values

def get_feature_importances(
    model: RegressorMixin,
    importance_type: str = 'model',
    n_features: int = 15,
    shap_values: Optional[np.ndarray] = None,
    X: Optional[Union[pd.DataFrame, np.ndarray]] = None,
    y: Optional[Union[pd.DataFrame, np.ndarray]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    特徴量の重要度を計算し、上位特徴量のインデックスを取得する関数。

    Args:
        model (RegressorMixin): モデルオブジェクト（回帰器）。
        importance_type (str): 重要度の計算方法 ('model', 'shap', 'pfi')。
        n_features (int): 取得する特徴量の数。
        shap_values (Optional[np.ndarray]): SHAP 値の配列（`importance_type` が 'shap' の場合に使用）。
        X (Optional[Union[pd.DataFrame, np.ndarray]]): 特徴量データ（`importance_type` が 'pfi' の場合に使用）。
        y (Optional[Union[pd.DataFrame, np.ndarray]]): 目的変数データ（`importance_type` が 'pfi' の場合に使用）。

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - 特徴量重要度を示す配列。
            - 上位特徴量のインデックスを示す配列。
    """
    if importance_type == 'model':
        # モデルから特徴量重要度を取得
        feature_importances = get_importances(model)
    elif importance_type == 'shap':
        # SHAP 値から特徴量重要度を計算
        if shap_values is None:
            raise ValueError("shap_values must be provided for SHAP importance.")
        feature_importances = np.sqrt((shap_values**2).sum(axis=0))
    elif importance_type == 'pfi':
        # パーミュテーションインポータンスから特徴量重要度を計算
        if X is None or y is None:
            raise ValueError("X and y must be provided for permutation importance.")
        feature_importances = get_pfi_values(model, X, y)
    else:
        raise ValueError(f"Unknown importance_type: {importance_type}")

    # 特徴量重要度の上位 n_features を取得
    importance_idx = np.argsort(-np.abs(feature_importances))[:n_features]
    
    return feature_importances, importance_idx