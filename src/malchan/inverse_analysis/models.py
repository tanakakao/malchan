import numpy as np
import pandas as pd
import plotly.graph_objects as go
import optuna

from typing import List, Optional, Union, Dict, Tuple, Callable, Any

from .utils import search_setting, get_sampler, objective, get_result, default_settings


def inverse_analysis(
    model,
    sampler_type: str = 'TPE',
    cat_dict: Dict = {},
    constraint_cols: List[str] = [],
    constraint_value: float = 1.0,
    obj_directions: List[str] = [],
    bounds_min: List[float] = [],
    bounds_max: List[float] = [],
    dtypes: List[str] = [],
    steps: List[float] = [],
    fix_values: List[float] = [],
    target_cols=[],
    trials: int = 250,
    n_candidate=10,
) -> Tuple[pd.DataFrame, pd.DataFrame, optuna.Study]:
    """
    モデルの逆解析を行い、最適化結果を返します。

    Args:
        model: 予測モデル。
        sampler_type (str): サンプラーのタイプ（例: 'TPE', 'CmaEs'など）。
        cat_dict (Dict): カテゴリ変数の辞書。
        constraint_cols (List[str]): 制約列のリスト。
        constraint_value (float): 制約値。
        obj_directions (List[str]): 目的方向（'min', 'max'など）のリスト。
        bounds_min (List[float]): 最小境界値のリスト。
        bounds_max (List[float]): 最大境界値のリスト。
        dtypes (List[str]): 各列のデータ型。
        steps (List[float]): 各列のステップ値。
        trials (int): 試行回数。

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, optuna.Study]: 最適化結果、試行データ、Optunaのstudyオブジェクト。
    """

    bounds_min, bounds_max, steps, dtypes, fix_values = default_settings(
        model,
        # obj_directions,
        bounds_min,
        bounds_max,
        steps,
        dtypes,
        fix_values
        )

    # 検索設定を取得
    df_range, y_cols, x_cols, obj_cols, obj_values, directions = search_setting(
        model,
        obj_directions,
        bounds_min,
        bounds_max,
        dtypes,
        steps,
        fix_values,
        target_cols
    )

    # サンプラーを取得
    sampler = get_sampler(sampler_type)

    # Optunaのスタディを作成
    study = optuna.create_study(
        directions=directions,
        sampler=sampler
    )

    # 最適化の実行
    study.optimize(
        lambda trial: objective(
            trial,
            model=model,
            cat_dict=cat_dict,
            y_cols=y_cols,
            obj_values=obj_values,
            df_range=df_range,
            x_cols=x_cols,
            constraint_cols=constraint_cols,
            constraint_value=constraint_value
        ),
        n_trials=trials
    )
    
    # 結果の取得
    _, df_trials = get_result(
        model,
        study,
        df_range,
        constraint_cols,
        constraint_value,
        x_cols,
        y_cols,
        obj_cols,
        fix_values
        )
    
    df_copy=df_trials[target_cols]
    for i, c in enumerate(y_cols):
        df_copy[c] = df_copy[c]*(-1)**(directions[i]=="maximize")
        df_copy[c] = (df_copy[c] - df_copy[c].min()) / (df_copy[c].max() - df_copy[c].min())
    df_trials["obj_score"] = df_copy[y_cols].sum(axis=1).values
    df_trials = df_trials.sort_values("obj_score", ascending=True).reset_index(drop=True).iloc[:n_candidate]
    df_trials = df_trials.drop(y_cols+["obj_score"], axis=1)

    return df_trials, study
