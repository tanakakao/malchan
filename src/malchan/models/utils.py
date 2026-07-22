import pandas as pd
import numpy as np
import optuna
from optuna.distributions import IntDistribution, FloatDistribution, CategoricalDistribution

from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, LabelEncoder

from matminer.featurizers.composition import (
    ElementProperty, OxidationStates, ValenceOrbital, AtomicOrbitals, YangSolidSolution,
    IonProperty, TMetalFraction, Stoichiometry, ElementFraction, Meredig
)

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

from rdkit import Chem
from skfp.fingerprints import ECFPFingerprint, MACCSFingerprint, RDKit2DDescriptorsFingerprint, MordredFingerprint, PubChemFingerprint, AtomPairFingerprint, AvalonFingerprint, PhysiochemicalPropertiesFingerprint
from skfp.fingerprints import AutocorrFingerprint, E3FPFingerprint, MORSEFingerprint, RDFFingerprint
from matminer.featurizers.composition import (
    ElementProperty, OxidationStates, ValenceOrbital, AtomicOrbitals, YangSolidSolution,
    IonProperty, TMetalFraction, Stoichiometry, ElementFraction,
    Meredig, BandCenter, Miedema, CohesiveEnergy, WenAlloys, Stoichiometry, ElectronAffinity
)

from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope

from typing import List, Optional, Union, Dict, Tuple, Callable, Any, Type
import warnings

optuna.logging.disable_default_handler()
warnings.simplefilter('ignore')


def get_mlp_hidden_layer_size_candidates() -> list[tuple[int, ...]]:
    """MLPのhidden_layer_sizes探索候補を返す。

    OptunaSearchCVは各パラメータ値にOptunaのDistributionを要求するため、
    hidden_layer_sizesの各層を個別のIntDistributionのタプルとして指定できない。
    そのため、MLPでよく使う1〜3層の構成をCategoricalDistributionの候補として返す。

    Returns:
        list[tuple[int, ...]]: hidden_layer_sizesに渡せる層構成の候補。
    """
    return (
        [(units,) for units in range(5, 105, 5)]
        + [(units, units) for units in range(10, 105, 10)]
        + [(units, units, units) for units in range(10, 105, 10)]
    )

REG_MODEL_DICT = {
    '線形回帰': LinearRegression,
    'Ridge': Ridge,
    'Lasso': Lasso,
    'ElasticNet': ElasticNet,
    'PLS回帰': PLSRegression,
    'ロバスト回帰': HuberRegressor,
    '一般化線形モデル': TweedieRegressor,
    '回帰木': DecisionTreeRegressor,
    'ランダムフォレスト回帰': RandomForestRegressor,
    'Extra-Trees': ExtraTreesRegressor,
    'Gradient Boosting': GradientBoostingRegressor,
    'HistGradientBoosting': HistGradientBoostingRegressor,
    'XGBoost': XGBRegressor,
    'LightGBM': LGBMRegressor,
    'CatBoost': CatBoostRegressor,
    'サポートベクター回帰': SVR,
    'K近傍法': KNeighborsRegressor,
    'ガウス過程回帰': GaussianProcessRegressor,
    'ベイズ線形回帰': BayesianRidge,
    'ARDベイズ線形回帰': ARDRegression,
    'オンライン学習': PassiveAggressiveRegressor,
    'LassoLars':LassoLars,
    '直交マッチング追跡':OrthogonalMatchingPursuit,
    '多層パーセプトロン':MLPRegressor
}

reg_default_params = {
    '線形回帰':{
    },
    'Ridge':{
        'alpha': 1.0
    },
    'Lasso':{
        'alpha': 1e-2
    },
    'ElasticNet':{
        'alpha': 1e-2,
        'l1_ratio': 0.5
    },
    'PLS回帰':
    {
        'n_components': 2
    },
    'ロバスト回帰':{
        'epsilon': 1.35
    },
    '一般化線形モデル':{
        'alpha': 1e-2,
        'power': 0,
        'link': 'auto'
    },
    '回帰木':{
        'max_depth': 5,
        'min_samples_split': 2,
        'min_samples_leaf': 2,
        'ccp_alpha': 0
    },
    'ランダムフォレスト回帰':{
        'n_estimators': 150,
        'max_features': 0.8,
        'max_depth': 6,
        'min_samples_split': 2,
        'min_samples_leaf': 2,
        'ccp_alpha': 0
    },
    'Extra-Trees':{
        'n_estimators': 150,
        'max_features': 0.8,
        'max_depth': 5,
        'min_samples_split': 2,
        'min_samples_leaf': 2,
        'ccp_alpha': 0
    },
    'Gradient Boosting':{
        'learning_rate': 0.1,
        'n_estimators': 150,
        'max_features': 0.8,
        'max_depth': 5,
        'min_samples_split': 2,
        'min_samples_leaf': 2,
        'ccp_alpha': 0
    },
    'HistGradientBoosting':{
        'learning_rate': 0.1,
        'max_features': 0.8,
        'max_depth': 6,
        'min_samples_leaf': 2,
    },
    'XGBoost':{
        'n_estimators': 300,
        'max_depth': 6,
        'learning_rate': 0.05,
        'reg_alpha': 0.0,
        'reg_lambda': 1.0,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 1,
        'n_jobs': -1,
        'random_state': 42,
        'verbosity': 0
    },
    'LightGBM': {
        'n_estimators': 300,
        'max_depth': -1,
        'num_leaves': 31,
        'learning_rate': 0.05,
        'reg_alpha': 0.0,
        'reg_lambda': 0.0,
        'colsample_bytree': 1.0,
        'subsample': 1.0,
        'min_child_samples': 20,
        'n_jobs': -1,
        'random_state': 42,
        'verbose': -1
    },
    'CatBoost':{
        'learning_rate': 0.05,
        'iterations': 500,
        'depth': 6,
        'l2_leaf_reg': 3.0,
        'random_seed': 42,
        'verbose': False,
    },
    'サポートベクター回帰':{
        'kernel': 'rbf',
        'C': 1.0,
        'epsilon': 0.1,
        'gamma': 'scale'
    },
    'K近傍法':{
        'n_neighbors': 5,
        'weights': 'distance'
    },
    'ガウス過程回帰':{
        'kernel': ConstantKernel() * DotProduct() + WhiteKernel()
    },
    'ベイズ線形回帰':{
        'alpha_1': 1e-6,
        'alpha_2': 1e-6,
        'lambda_1': 1e-6,
        'lambda_2': 1e-6
    },
    'ARDベイズ線形回帰':{
        'alpha_1': 1e-6,
        'alpha_2': 1e-6,
        'lambda_1': 1e-6,
        'lambda_2': 1e-6
    },
    'オンライン学習':{
        'C': 0.1,
    },
    'LassoLars':{
        'alpha': 1e-2
    },
    '直交マッチング追跡':{
    },
    '多層パーセプトロン':{
        'hidden_layer_sizes': (100,),
        'activation': "relu",
        'solver': 'adam',
        'alpha': 1e-4,
        'learning_rate_init': 1e-3,
        'learning_rate': "constant",
        'early_stopping': True,
        'max_iter': 500,
        'random_state': 42
    }
}

CLS_MODEL_DICT = {
    'ロジスティック回帰': LogisticRegression,
    # 'Ridge': RidgeClassifier,
    '決定木': DecisionTreeClassifier,
    'ランダムフォレスト': RandomForestClassifier,
    'Extra-Trees': ExtraTreesClassifier,
    'Gradient Boosting': GradientBoostingClassifier,
    'HistGradientBoosting': HistGradientBoostingClassifier,
    'XGBoost': XGBClassifier,
    'LightGBM': LGBMClassifier,
    'CatBoost': CatBoostClassifier,
    'サポートベクターマシン': SVC,
    'K近傍法': KNeighborsClassifier,
    'ガウス過程分類': GaussianProcessClassifier,
    'ナイーブベイズ': GaussianNB,
    # 'オンライン学習': PassiveAggressiveClassifier,
    '多層パーセプトロン':MLPClassifier
}

cls_default_params = {
    'ロジスティック回帰':{
        'penalty': 'l2',
        'C': 1.0,
        'solver': 'lbfgs',
        'max_iter': 1000
    },
    # 'Ridge':{
    #     'alpha': 1e-2
    # },
    '決定木':{
        'max_depth': 5,
        'min_samples_split': 2,
        'min_samples_leaf': 2,
        'ccp_alpha': 0
    },
    'ランダムフォレスト':{
        'n_estimators': 150,
        'max_features': 0.8,
        'max_depth': 5,
        'min_samples_split': 2,
        'min_samples_leaf': 2,
        'ccp_alpha': 0
    },
    'Extra-Trees':{
        'n_estimators': 150,
        'max_features': 0.8,
        'max_depth': 5,
        'min_samples_split': 2,
        'min_samples_leaf': 2,
        'ccp_alpha': 0
    },
    'Gradient Boosting':{
        'learning_rate': 0.1,
        'n_estimators': 150,
        'max_features': 0.8,
        'max_depth': 5,
        'min_samples_split': 2,
        'min_samples_leaf': 2,
        'ccp_alpha': 0
    },
    'HistGradientBoosting':{
        'learning_rate': 0.1,
        'max_features': 0.8,
        'max_depth': 5,
        'min_samples_leaf': 2,
    },
    'XGBoost':{
        'n_estimators': 300,
        'max_depth': 6,
        'learning_rate': 0.05,
        'reg_alpha': 0.0,
        'reg_lambda': 1.0,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 1,
        'n_jobs': -1,
        'random_state': 42,
        'verbosity': 0
    },
    'LightGBM': {
        'n_estimators': 300,
        'max_depth': -1,
        'num_leaves': 31,
        'learning_rate': 0.05,
        'reg_alpha': 0.0,
        'reg_lambda': 0.0,
        'colsample_bytree': 1.0,
        'subsample': 1.0,
        'min_child_samples': 20,
        'n_jobs': -1,
        'random_state': 42,
        'verbose': -1
    },
    'CatBoost':{
        'learning_rate': 0.05,
        'iterations': 500,
        'depth': 6,
        'l2_leaf_reg': 3.0,
        'random_seed': 42,
        'verbose': False,
    },
    'サポートベクターマシン':{
        'kernel': 'rbf',
        'C': 1.0,
        'gamma': 'scale',
        'probability': True
    },
    'K近傍法':{
        'n_neighbors': 5,
        'weights': 'distance'
    },
    'ガウス過程分類':{
        'kernel': ConstantKernel() * DotProduct() + WhiteKernel()
    },
    'ナイーブベイズ':{
    },
    # 'オンライン学習':{
    #     'C': 0.1,
    # },
    '多層パーセプトロン':{
        'hidden_layer_sizes': (100,),
        'activation': "relu",
        'solver': 'adam',
        'alpha': 1e-4,
        'learning_rate_init': 1e-3,
        'learning_rate': "constant",
        'early_stopping': True,
        'max_iter': 500,
        'random_state': 42
    }
}

FP_dcit = {
    "ECFP": ECFPFingerprint(count=True),
    "MACCS": MACCSFingerprint(),
    "RDKit": RDKit2DDescriptorsFingerprint(),
    # "Mordred": MordredFingerprint(),
    "PubChem": PubChemFingerprint(),
    "AtomPair": AtomPairFingerprint(),
    "Avalon": AvalonFingerprint(),
    "PhysChem": PhysiochemicalPropertiesFingerprint(),
    "Autocorr": AutocorrFingerprint(use_3D=True),
    "E3FP": E3FPFingerprint(),
    "MORSE": MORSEFingerprint(),
    "RDF": RDFFingerprint()
}


INORG_dict = {
    # "OxidationStates": OxidationStates(),
    "ElementProperty":ElementProperty.from_preset("magpie", impute_nan=True),
    "ValenceOrbital":ValenceOrbital(impute_nan=True),
    "IonProperty":IonProperty(impute_nan=True),
    # "AtomicOrbitals": AtomicOrbitals(),
    "YangSolidSolution": YangSolidSolution(),
    "TMetalFraction": TMetalFraction(),
    "Stoichiometry": Stoichiometry(),
    "Meredig":Meredig(),
    "BandCenter":BandCenter(),
    "Miedema":Miedema(),
    # "CohesiveEnergy":CohesiveEnergy(),
    # "WenAlloys":WenAlloys(),
    # "ElectronAffinity":ElectronAffinity(),
    "ElementFraction": ElementFraction(),  # 全元素比（列が多くなるので必要に応じて）
}

MENDELEEV_FEATURES = [
    "atomic_number",
    "atomic_weight",
    "period",
    "group_id",
    "block",
    "covalent_radius_cordero",
    "covalent_radius_pyykko",
    "atomic_radius_rahm",
    "atomic_radius",
    "vdw_radius",
    "en_pauling",
    "en_allen",
    "electron_affinity",
    "dipole_polarizability",
    "fusion_heat",
    "evaporation_heat",
    "specific_heat_capacity",
    "thermal_conductivity",
    "density",
    "mendeleev_number",
    "pettifor_number",
    "en_miedema",
    "miedema_electron_density",
    "miedema_molar_volume",
    "metallic_radius"
]

AD_MODEL_DICT = {
    "OneClassSVM": OneClassSVM,
    "IsolationForest": IsolationForest,
    "EllipticEnvelope":EllipticEnvelope
}

SAMPLING_DICT = {
    "OverSampling":"ros",
    "UnderSampling":"rus",
    "SMOTE":"smote",
    "重みづけ":"sample_weight"
}

ad_default_params = {
    'OneClassSVM':{
        'nu' : 0.2,
        'kernel': "rbf",
        "gamma": "auto"
    },
    'IsolationForest':{
        'n_estimators': 100,
        'contamination' : 'auto',
    },
    'EllipticEnvelope':{
        'contamination':0.01
    }
}

def get_param_grid_reg(model_name: str) -> dict:
    """
    指定されたモデル名に基づいて、ハイパーパラメータグリッドを返す関数。

    Args:
        model_name (str): モデルの名前。

    Returns:
        dict: モデルに対応するハイパーパラメータのグリッド。
    """
    kernels = [
        ConstantKernel() * DotProduct() + WhiteKernel(),
        ConstantKernel() * RBF() + WhiteKernel(),
        ConstantKernel() * RBF() + WhiteKernel() + ConstantKernel() * DotProduct(),
        ConstantKernel() * Matern(nu=1.5) + WhiteKernel(),
        ConstantKernel() * Matern(nu=1.5) + WhiteKernel() + ConstantKernel() * DotProduct(),
        ConstantKernel() * Matern(nu=0.5) + WhiteKernel(),
        ConstantKernel() * Matern(nu=0.5) + WhiteKernel() + ConstantKernel() * DotProduct(),
        ConstantKernel() * Matern(nu=2.5) + WhiteKernel(),
        ConstantKernel() * Matern(nu=2.5) + WhiteKernel() + ConstantKernel() * DotProduct()
    ]
    
    param_grids = {
        '線形回帰': {
            'predictor__fit_intercept': CategoricalDistribution([True])
        },
        'Ridge': {
            'predictor__alpha': FloatDistribution(1e-3, 1e+3, log=True)
        },
        'Lasso': {
            'predictor__alpha': FloatDistribution(1e-3, 1e+3, log=True)
        },
        'ElasticNet': {
            'predictor__alpha': FloatDistribution(1e-3, 1e+3, log=True),
            'predictor__l1_ratio': FloatDistribution(0.0, 1.0, step=0.1)
        },
        'PLS回帰': {
            'predictor__n_components': IntDistribution(1, 10)
        },
        'ロバスト回帰': {
            'predictor__epsilon': FloatDistribution(1, 10, log=False)
        },
        '一般化線形モデル': {
            'predictor__alpha': FloatDistribution(1e-3, 1e+1, log=True),
            'predictor__power': CategoricalDistribution([0]),
            'predictor__link': CategoricalDistribution(['identity'])
        },
        '回帰木': {
            'predictor__max_depth': IntDistribution(3, 20),
            'predictor__min_samples_split': IntDistribution(2, 10),
            'predictor__min_samples_leaf': IntDistribution(1, 4),
            'predictor__ccp_alpha': FloatDistribution(1e-3, 1, log=True)
        },
        'ランダムフォレスト回帰': {
            'predictor__n_estimators': IntDistribution(50, 300, step=50),
            'predictor__max_depth': IntDistribution(3, 20),
            'predictor__max_features': FloatDistribution(0.1, 0.9, step=0.1),
            'predictor__min_samples_split': IntDistribution(2, 10),
            'predictor__min_samples_leaf': IntDistribution(1, 4)
        },
        'Extra-Trees': {
            'predictor__n_estimators': IntDistribution(50, 300, step=50),
            'predictor__max_depth': IntDistribution(3, 20),
            'predictor__max_features': CategoricalDistribution(['sqrt', None]),
            'predictor__min_samples_split': IntDistribution(2, 10),
            'predictor__min_samples_leaf': IntDistribution(1, 4)
        },
        'Gradient Boosting': {
            'predictor__learning_rate': FloatDistribution(1e-2, 3e-1, log=True),
            'predictor__n_estimators': IntDistribution(100, 1000, step=50),
            'predictor__max_depth': IntDistribution(3, 30),
            'predictor__max_features': CategoricalDistribution(['sqrt', None]),
            'predictor__min_samples_split': IntDistribution(2, 10),
            'predictor__min_samples_leaf': IntDistribution(1, 4)
        },
        'HistGradientBoosting': {
            'predictor__learning_rate': FloatDistribution(1e-3, 1., log=True),
            'predictor__max_depth': IntDistribution(3, 30),
            'predictor__max_features': FloatDistribution(0.1, 1.0, step=0.1),
            'predictor__min_samples_leaf': IntDistribution(1, 4)
        },
        'XGBoost': {
            'predictor__n_estimators': IntDistribution(100, 800, step=50),
            'predictor__max_depth': IntDistribution(2, 12),
            'predictor__learning_rate': FloatDistribution(1e-2, 3e-1, log=True),
            'predictor__gamma': FloatDistribution(1e-8, 10, log=True),
            'predictor__reg_alpha': FloatDistribution(1e-3, 1e+1, log=True),
            'predictor__reg_lambda': FloatDistribution(1e-3, 1e+1, log=True),
            'predictor__min_child_weight': IntDistribution(1, 10),
            # 'predictor__subsample': FloatDistribution(0.1, 1.0, step=0.1),
            'predictor__colsample_bytree': FloatDistribution(0.1, 1.0, step=0.1),
            'predictor__colsample_bylevel': FloatDistribution(0.1, 1.0, step=0.1),
            'predictor__colsample_bynode': FloatDistribution(0.1, 1.0, step=0.1)
        },
        'LightGBM': {
            'predictor__learning_rate': FloatDistribution(1e-2, 3e-1, log=True),
            'predictor__n_estimators': IntDistribution(100, 1000, step=50),
            'predictor__num_leaves': IntDistribution(8, 128),
            'predictor__max_depth': CategoricalDistribution([-1, 3, 4, 5, 6, 8, 10, 12]),
            'predictor__min_child_samples': IntDistribution(5, 100),
            'predictor__reg_alpha': FloatDistribution(1e-3, 1e+1, log=True),
            'predictor__reg_lambda': FloatDistribution(1e-3, 1e+1, log=True),
            'predictor__colsample_bytree': FloatDistribution(0.1, 1.0, step=0.1),
            # 'predictor__subsample': FloatDistribution(0.5, 1.0, step=0.1)
        },
        'CatBoost': {
            'predictor__learning_rate': FloatDistribution(1e-2, 3e-1, log=True),
            'predictor__iterations': IntDistribution(200, 1000, step=50),
            'predictor__depth': IntDistribution(4, 10),
            'predictor__random_strength': FloatDistribution(1e-3, 10, log=True),
            'predictor__l2_leaf_reg': FloatDistribution(1e-1, 1e2, log=True),
            # 'predictor__subsample': FloatDistribution(0.2, 0.8, step=0.1),
        },
        'サポートベクター回帰': {
            'predictor__kernel': CategoricalDistribution(['linear', 'poly', 'rbf']),
            'predictor__C': FloatDistribution(1e-2, 1e+3, log=True),
            'predictor__epsilon': FloatDistribution(1e-2, 1e+1, log=True)
        },
        'K近傍法': {
            'predictor__n_neighbors': IntDistribution(1, 15),
            'predictor__weights': CategoricalDistribution(['uniform', 'distance'])
        },
        'ガウス過程回帰': {
            'predictor__kernel': CategoricalDistribution(kernels)
        },
        'ベイズ線形回帰': {
            'predictor__alpha_1': FloatDistribution(1e-8, 1e-2, log=True),
            'predictor__alpha_2': FloatDistribution(1e-8, 1e-2, log=True),
            'predictor__lambda_1': FloatDistribution(1e-8, 1e-2, log=True),
            'predictor__lambda_2': FloatDistribution(1e-8, 1e-2, log=True)
        },
        'ARDベイズ線形回帰': {
            'predictor__alpha_1': FloatDistribution(1e-8, 1e-2, log=True),
            'predictor__alpha_2': FloatDistribution(1e-8, 1e-2, log=True),
            'predictor__lambda_1': FloatDistribution(1e-8, 1e-2, log=True),
            'predictor__lambda_2': FloatDistribution(1e-8, 1e-2, log=True)
        },
        'オンライン学習': {
            'predictor__C': FloatDistribution(1e-3, 1e+2, log=True),
        },
        'LassoLars': {
            'predictor__alpha': FloatDistribution(1e-3, 1e+3, log=True)
        },
        '直交マッチング追跡': {
            'predictor__fit_intercept': CategoricalDistribution([True])
        },
        '多層パーセプトロン': {
            'predictor__hidden_layer_sizes': CategoricalDistribution(get_mlp_hidden_layer_size_candidates()),
            'predictor__activation': CategoricalDistribution(["relu", "tanh", "logistic"]),
            'predictor__solver': CategoricalDistribution(["lbfgs", "adam"]),
            'predictor__alpha': FloatDistribution(1e-6, 1e-1, log=True),
            'predictor__learning_rate_init': FloatDistribution(1e-4, 1e-2, log=True),
            'predictor__learning_rate': CategoricalDistribution(["constant"]),
        }
    }
    
    # 存在しないモデル名の場合はNoneを返す
    return param_grids.get(model_name, None)

def get_param_grid_cls(model_name: str) -> dict:
    """
    指定されたモデル名に基づいて、ハイパーパラメータグリッドを返す関数。

    Args:
        model_name (str): モデルの名前。

    Returns:
        dict: モデルに対応するハイパーパラメータのグリッド。
    """
    kernels = [
        ConstantKernel() * DotProduct() + WhiteKernel(),
        ConstantKernel() * RBF() + WhiteKernel(),
        ConstantKernel() * RBF() + WhiteKernel() + ConstantKernel() * DotProduct(),
        ConstantKernel() * Matern(nu=1.5) + WhiteKernel(),
        ConstantKernel() * Matern(nu=1.5) + WhiteKernel() + ConstantKernel() * DotProduct(),
        ConstantKernel() * Matern(nu=0.5) + WhiteKernel(),
        ConstantKernel() * Matern(nu=0.5) + WhiteKernel() + ConstantKernel() * DotProduct(),
        ConstantKernel() * Matern(nu=2.5) + WhiteKernel(),
        ConstantKernel() * Matern(nu=2.5) + WhiteKernel() + ConstantKernel() * DotProduct()
    ]
    
    param_grids = {
        'ロジスティック回帰': {
            'predictor__penalty': CategoricalDistribution(['l1', 'l2']),
            'predictor__solver': CategoricalDistribution(['liblinear']),
            'predictor__C': FloatDistribution(1e-4, 1e2, log=True)
        },
        # 'Ridge': {
        #     'predictor__alpha': FloatDistribution(1e-3, 1e+3, log=True)
        # },
        '決定木': {
            'predictor__max_depth': IntDistribution(3, 20),
            'predictor__min_samples_split': IntDistribution(2, 10),
            'predictor__min_samples_leaf': IntDistribution(1, 4),
            'predictor__ccp_alpha': FloatDistribution(1e-3, 1, log=True)
        },
        'ランダムフォレスト': {
            'predictor__n_estimators': IntDistribution(50, 300, step=50),
            'predictor__max_depth': IntDistribution(3, 20),
            'predictor__max_features': FloatDistribution(0.1, 0.9, step=0.1),
            'predictor__min_samples_split': IntDistribution(2, 10),
            'predictor__min_samples_leaf': IntDistribution(1, 4)
        },
        'Extra-Trees': {
            'predictor__n_estimators': IntDistribution(50, 300, step=50),
            'predictor__max_depth': IntDistribution(3, 20),
            'predictor__max_features': CategoricalDistribution(['sqrt', None]),
            'predictor__min_samples_split': IntDistribution(2, 10),
            'predictor__min_samples_leaf': IntDistribution(1, 4)
        },
        'Gradient Boosting': {
            'predictor__learning_rate': FloatDistribution(1e-2, 3e-1, log=True),
            'predictor__n_estimators': IntDistribution(100, 1000, step=50),
            'predictor__max_depth': IntDistribution(3, 30),
            'predictor__max_features': CategoricalDistribution(['sqrt', None]),
            'predictor__min_samples_split': IntDistribution(2, 10),
            'predictor__min_samples_leaf': IntDistribution(1, 4)
        },
        'HistGradientBoosting': {
            'predictor__learning_rate': FloatDistribution(1e-3, 1., log=True),
            'predictor__max_depth': IntDistribution(3, 30),
            'predictor__max_features': FloatDistribution(0.1, 1.0, step=0.1),
            'predictor__min_samples_leaf': IntDistribution(1, 4)
        },
        'XGBoost': {
            'predictor__n_estimators': IntDistribution(100, 800, step=50),
            'predictor__max_depth': IntDistribution(2, 12),
            'predictor__learning_rate': FloatDistribution(1e-2, 3e-1, log=True),
            'predictor__gamma': FloatDistribution(1e-8, 10, log=True),
            'predictor__reg_alpha': FloatDistribution(1e-3, 1e+1, log=True),
            'predictor__reg_lambda': FloatDistribution(1e-3, 1e+1, log=True),
            'predictor__min_child_weight': IntDistribution(1, 10),
            # 'predictor__subsample': FloatDistribution(0.1, 1.0, step=0.1),
            'predictor__colsample_bytree': FloatDistribution(0.1, 1.0, step=0.1),
            'predictor__colsample_bylevel': FloatDistribution(0.1, 1.0, step=0.1),
            'predictor__colsample_bynode': FloatDistribution(0.1, 1.0, step=0.1)
        },
        'LightGBM': {
            'predictor__learning_rate': FloatDistribution(1e-2, 3e-1, log=True),
            'predictor__n_estimators': IntDistribution(100, 1000, step=50),
            'predictor__num_leaves': IntDistribution(8, 128),
            'predictor__max_depth': CategoricalDistribution([-1, 3, 4, 5, 6, 8, 10, 12]),
            'predictor__min_child_samples': IntDistribution(5, 100),
            'predictor__reg_alpha': FloatDistribution(1e-3, 1e+1, log=True),
            'predictor__reg_lambda': FloatDistribution(1e-3, 1e+1, log=True),
            'predictor__colsample_bytree': FloatDistribution(0.1, 1.0, step=0.1),
            # 'predictor__subsample': FloatDistribution(0.5, 1.0, step=0.1)
        },
        'CatBoost': {
            'predictor__learning_rate': FloatDistribution(1e-2, 3e-1, log=True),
            'predictor__iterations': IntDistribution(200, 1000, step=50),
            'predictor__depth': IntDistribution(4, 10),
            'predictor__random_strength': FloatDistribution(1e-3, 10, log=True),
            'predictor__l2_leaf_reg': FloatDistribution(1e-1, 1e2, log=True),
            # 'predictor__subsample': FloatDistribution(0.2, 0.8, step=0.1),
        },
        'サポートベクターマシン': {
            'predictor__kernel': CategoricalDistribution(['linear', 'poly', 'rbf']),
            'predictor__C': FloatDistribution(1e-2, 1e+3, log=True),
            'predictor__probability': CategoricalDistribution([True])
        },
        'K近傍法': {
            'predictor__n_neighbors': IntDistribution(1, 15),
            'predictor__weights': CategoricalDistribution(['uniform', 'distance'])
        },
        'ガウス過程分類': {
            'predictor__kernel': CategoricalDistribution(kernels)
        },
        'ナイーブベイズ': {
            'predictor__priors': CategoricalDistribution([None])
        },
        # 'オンライン学習': {
        #     'predictor__C': FloatDistribution(1e-3, 1e+2, log=True),
        # },
        '多層パーセプトロン': {
            'predictor__hidden_layer_sizes': CategoricalDistribution(get_mlp_hidden_layer_size_candidates()),
            'predictor__activation': CategoricalDistribution(["relu", "tanh", "logistic"]),
            'predictor__solver': CategoricalDistribution(["lbfgs", "adam"]),
            'predictor__alpha': FloatDistribution(1e-6, 1e-1, log=True),
            'predictor__learning_rate_init': FloatDistribution(1e-4, 1e-2, log=True),
            'predictor__learning_rate': CategoricalDistribution(["constant"]),
        }
    }
    
    # 存在しないモデル名の場合はNoneを返す
    return param_grids.get(model_name, None)

def get_cat_unique_values(df, cat_cols):
    unique_cols = {}
    for c in cat_cols:
        enc = OrdinalEncoder()
        enc.fit_transform(df[[c]])
        unique_cols[c] = enc.categories_[0].tolist()

    return unique_cols

def label_encode(y):
    le = LabelEncoder()
    _y = le.fit_transform(y)
    y_le = pd.DataFrame(_y, columns=y.columns)[y.columns[0]]
    return y_le, le

def feature_names_from_pipeline(model):
    feature_names = model['preprocess'].get_feature_names_out()
    feature_names = [c.replace('num_cat__num__','').replace('num_cat__cat__','').replace('comp__comp__','comp__') for c in feature_names]
    return feature_names