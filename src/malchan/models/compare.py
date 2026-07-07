import pandas as pd

from .models import SingleOutputMLModelPipeline
from .utils import CLS_MODEL_DICT, REG_MODEL_DICT

from typing import List, Optional, Union, Dict, Tuple, Callable, Any
import warnings
warnings.simplefilter('ignore')

def train_and_evaluate_models(
    df,
    target_cols,
    tasks,
    num_cols,
    cat_cols,
    smiles_col,
    comp_col
):
    """指定された特徴量・モデルを用いて学習・評価を行い、結果を比較する。

    各モデルに対して、交差検証スコアを計算し、訓練データとテストデータの
    指標 (RMSE, MAE, MAPE, R2) をまとめる。

    Args:
        X (pd.DataFrame): 入力データ（特徴量と目的変数を含む）。
        target_cols (list[str]): 目的変数のカラム名のリスト。
        num_cols (list[str]): 数値特徴量のカラム名のリスト。
        cat_cols (list[str]): カテゴリ特徴量のカラム名のリスト。
        num_impute_type (str, optional): 数値特徴量の欠損値補完方法（デフォルト: None）。
        num_scale_type (str, optional): 数値特徴量のスケーリング方法（デフォルト: None）。
        cat_impute (bool, optional): カテゴリ特徴量の欠損値補完を行うか（デフォルト: False）。
        poly (bool, optional): 多項式特徴量を作成するか（デフォルト: False）。
        poly_degree (int, optional): 多項式特徴量の次数（デフォルト: 1）。
        poly_interaction_only (bool, optional): 交互作用項のみを作成するか（デフォルト: True）。
        pca (bool, optional): PCA による次元削減を行うか（デフォルト: False）。
        pca_n_components (int, optional): PCA の主成分数（デフォルト: 2）。

    Returns:
        dict[str, pd.DataFrame]: 各目的変数ごとの評価結果を格納した辞書。
            - キー: 目的変数名
            - 値: 訓練・テストデータの指標（RMSE, MAE, MAPE, R2）を含む DataFrame
    """
    
    compare_result_dict = {}

    # 各モデルに対して学習と評価を実施
    for i, target in enumerate(target_cols):
        if tasks[i] == "regression":
            MODEL_DICT = REG_MODEL_DICT
        else:
            MODEL_DICT = CLS_MODEL_DICT

        scores = {}
        train_result = []
        test_result = []
        for modelname in MODEL_DICT.keys():
            model = SingleOutputMLModelPipeline()
            model.fit(
                df,
                target_col=target,
                task=tasks[i],
                num_cols=num_cols,
                cat_cols=cat_cols,
                smiles_cols=[smiles_col],
                fingerprints=["RDKit"],
                comp_cols=[comp_col],
                comp_method="xenonpy",
                comp_feats=None,
                model_names=[modelname],
                tuning=False,
                ensemble=False,
                ens_type=None,
                base_model=None,
                model_params=None,
                base_model_param = None,    
                num_impute_type=None,
                num_scale_type=None,
                cat_impute=False,
                poly=False,
                poly_degree=2,
                poly_interaction_only=False,
                decomposition=False,
                decomposition_method="PCA",
                dec_n_components=2
            )  # モデルの学習
            model.cv_score()  # 交差検証スコアの計算
            scores[modelname] = model.cv_scores

            # 訓練データのスコアを取得
            tmp_train = scores[modelname]["train"]
            tmp_train.index = [modelname]
            train_result.append(tmp_train)

            # テストデータのスコアを取得
            tmp_test = scores[modelname]["test"]
            tmp_test.index = [modelname]
            test_result.append(tmp_test)

        if tasks[i] == "regression":
            # 訓練データの評価指標をマルチインデックスの DataFrame に変換
            train_result = pd.concat(train_result)
            train_result.columns = pd.MultiIndex.from_tuples([
                ("train", "RMSE"), ("train", "MAE"), ("train", "MAPE"), ("train", "R2")
            ])
            
            # テストデータの評価指標をマルチインデックスの DataFrame に変換
            test_result = pd.concat(test_result)
            test_result.columns = pd.MultiIndex.from_tuples([
                ("test", "RMSE"), ("test", "MAE"), ("test", "MAPE"), ("test", "R2")
            ])
        else:
            # 訓練データの評価指標をマルチインデックスの DataFrame に変換
            train_result = pd.concat(train_result)
            train_result.columns = pd.MultiIndex.from_tuples([
                ("train", "ACCURACY"), ("train", "PRECISION"), ("train", "RECALL"), ("train", "F1")
            ])
            
            # テストデータの評価指標をマルチインデックスの DataFrame に変換
            test_result = pd.concat(test_result)
            test_result.columns = pd.MultiIndex.from_tuples([
                ("test", "ACCURACY"), ("test", "PRECISION"), ("test", "RECALL"), ("test", "F1")
            ])          
        
        # 訓練データとテストデータの評価結果を統合
        compare_result_dict[target] = pd.concat([train_result, test_result], axis=1)

    return compare_result_dict
