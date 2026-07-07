import numpy as np
import pandas as pd
import optuna

from decimal import Decimal
from optuna.samplers import (
    CmaEsSampler,
    GPSampler,
    TPESampler,
    # MOTPESampler,
    NSGAIISampler,
    NSGAIIISampler,
    QMCSampler
)
# import optunahub
from typing import List, Optional, Union, Dict, Tuple, Callable, Any


def dtype2str(x: Union[str, pd.Series]) -> str:
    """
    データ型を文字列に変換します。

    Args:
        x (Union[str, pd.Series]): データ型を表す文字列またはpandasのSeries

    Returns:
        str: 変換後のデータ型を表す文字列
    """
    if x == 'object':
        return 'object'
    elif x == 'int64':
        return 'int'
    elif x == 'float64':
        return 'float'
    else:
        raise ValueError(f"Unsupported dtype: {x}")

def set_step(x: pd.Series) -> Optional[float]:
    """
    データ型に応じて刻み幅を設定します。

    Args:
        x (pd.Series): 計算対象のpandasのSeries

    Returns:
        Optional[float]: 設定された刻み幅。オブジェクト型の場合はNoneを返す。
    """
    if x.dtypes == 'object':
        return None  # オブジェクト型の場合は刻み幅を設定しない
    elif x.dtypes == 'int64':
        return 1  # 整数型の場合は刻み幅を1に設定
    else:
        # 数値型の場合は、最小値と最大値から刻み幅を計算
        result = min(
            10 ** (Decimal(str(x.min())).as_tuple().exponent - 1), 
            10 ** (Decimal(str(x.max())).as_tuple().exponent - 1)
        )
        return max(result, 1.0e-5)  # 1.0e-5 より小さい場合は1.0e-5を返す

def set_obj_value(x: str) -> Optional[float]:
    """
    'min' または 'max' なら None、それ以外なら元の値を返します。
    
    Args:
        x (str): 判定対象の文字列。
    
    Returns:
        Optional[float]: None または元の値。
    """
    return None if x in ['min', 'max'] else x


def set_direction(x: str) -> str:
    """
    'min' なら 'minimize'、'max' なら 'maximize' を返します。
    
    Args:
        x (str): 判定対象の文字列。
    
    Returns:
        str: 'minimize' または 'maximize'。
    """
    return 'maximize' if x == 'max' else 'minimize'


def set_dtype(x: str) -> str:
    """
    データ型の文字列を 'int' に変換します。
    
    Args:
        x (str): データ型の文字列。
    
    Returns:
        str: 'int' なら 'int'、それ以外は元の値。
    """
    return 'int' if x == 'int64' else x


def set_step(x: pd.Series) -> Optional[float]:
    """
    Series の最小・最大値からステップサイズを計算します。
    カテゴリカルなデータの場合は None を返します。
    
    Args:
        x (pd.Series): 処理対象のデータ列。
    
    Returns:
        Optional[float]: ステップサイズ、または None。
    """
    if x.dtype == 'object':
        return None
    elif x.dtype == 'int64':
        return 1

    # 連続値の場合、最小・最大値の桁数に基づき、適切なステップを計算
    min_exp = Decimal(str(x.min())).as_tuple().exponent
    max_exp = Decimal(str(x.max())).as_tuple().exponent
    result = min(10 ** (min_exp - 1), 10 ** (max_exp - 1))

    # 最小ステップは 1.0e-5 とする
    return max(result, 1.0e-5)

def default_settings(
    model,
    # obj_directions,
    bounds_min,
    bounds_max,
    steps,
    dtypes,
    fix_values,
    # target_cols=None
):
    # # 目的変数の列名を取得
    # if target_cols is None:
    #     y_cols = model.target_cols
    # else:
    #     y_cols = [c for c in target_cols if c in model.target_cols]
    #     if len(y_cols)==0:
    #         raise ValueError("指定されたtargetがmodelにありません。")
    
    # 数値変数の列名を取得
    x_cols = model.num_cols
    # if len(obj_directions)==len(y_cols):
    #     directions = [set_direction(d) for d in obj_directions]
    # else:
    #     directions = ["maximize"]*len(y_cols)
    
    # # 目的関数の目標値を設定 (最小化・最大化以外の場合に使う)
    # if len(obj_directions)==len(y_cols):
    #     obj_values = [set_obj_value(d) for d in obj_directions]
    # else:
    #     obj_values = [None]*len(y_cols)
    
    # # 目的変数の列名を、obj_valuesがNoneでないものに限定して取得
    # obj_cols = [y_cols[i] for i, v in enumerate(obj_values) if v is not None]

    if len(bounds_min)!=len(x_cols):
        bounds_min=list(model.X[model.num_cols].min().values)
    if len(bounds_max)!=len(x_cols):
        bounds_max=list(model.X[model.num_cols].max().values)

    if len(steps)!=len(x_cols):
        steps = list(model.X[model.num_cols].apply(set_step, axis=0))

    # データ型を取得
    if len(dtypes)!=len(x_cols):
        dtypes = list(model.X[model.num_cols].dtypes.apply(dtype2str))

    if len(fix_values)!=len(x_cols + model.cat_cols+ model.smiles_cols+ model.comp_cols):
        fix_values = [None]*len(x_cols + model.cat_cols+ model.smiles_cols+ model.comp_cols)
    return bounds_min, bounds_max, steps, dtypes, fix_values

def adjust_rows_to_sum_with_constraints(
    df: pd.DataFrame,
    x_cols: List[str],
    fix_values: List[Optional[float]],
    constraint_cols: List[str],
    constraint_value: float,
    df_range: pd.DataFrame
) -> pd.DataFrame:
    """
    固定値とmin/max制約付きで、指定列の合計が constraint_value になるよう調整。

    Args:
        df (pd.DataFrame): 元のデータ。
        x_cols (List[str]): 全体の調整対象列。
        fix_values (List[Optional[float]]): x_cols に対応する固定値リスト。Noneなら可変。
        constraint_cols (List[str]): 合計制約をかける列のリスト。
        constraint_value (float): 制約する合計値。
        df_range (pd.DataFrame): 各列に対する min/max 制約。

    Returns:
        pd.DataFrame: 調整済みのデータ。
    """
    adjusted_df = df.copy()

    for i, row in adjusted_df.iterrows():
        # 固定値をセット
        for col, val in zip(x_cols, fix_values):
            if val is not None:
                adjusted_df.at[i, col] = val

        # 制約対象列の合計を算出
        current_row = adjusted_df.loc[i, constraint_cols]
        total = current_row.sum()
        diff = constraint_value - total

        if diff == 0:
            continue

        # 可変かつ制約対象である列のみを抽出
        adjustable_cols = [
            col for col, val in zip(x_cols, fix_values)
            if val is None and col in constraint_cols
        ]

        # 大きい順に調整優先順位を決める
        sort_cols = current_row[adjustable_cols].sort_values(ascending=False).index.tolist()

        for col in sort_cols:
            current = adjusted_df.at[i, col]
            min_val = df_range.loc[col, 'min']
            max_val = df_range.loc[col, 'max']

            if diff > 0:
                gap = max_val - current
                adjust = min(gap, diff)
            else:
                gap = current - min_val
                adjust = max(-gap, diff)

            adjusted_df.at[i, col] = current + adjust
            diff -= adjust

            if diff == 0:
                break
    return adjusted_df.drop_duplicates().reset_index(drop=True)

def search_setting(
    model,
    obj_directions: List[str],
    bounds_min: List[float],
    bounds_max: List[float],
    dtypes: List[str],
    steps: List[Optional[float]],
    fix_values: List[float],
    target_cols: List[str]=None
) -> Tuple[pd.DataFrame, List[str], List[str], List[str], List[Optional[float]], List[int]]:
    """
    検索の設定を行う関数。
    
    Args:
        model: モデルオブジェクト。target_cols, num_cols, cat_cols などの属性を持つ。
        obj_directions (List[str]): 目的関数の方向（'min' または 'max'）。
        bounds_min (List[float]): 各数値変数の最小値リスト。
        bounds_max (List[float]): 各数値変数の最大値リスト。
        dtypes (List[str]): 各数値変数のデータ型リスト。
        steps (List[Optional[float]]): 各数値変数のステップ（離散化の場合）。

    Returns:
        Tuple:
            - df_range: 各変数の範囲、データ型、ステップサイズを含むDataFrame。
            - y_cols: 目的変数の列名リスト。
            - x_cols: 全ての特徴量（数値・カテゴリ）の列名リスト。
            - obj_cols: 目的変数の列名リスト（最小化または最大化の対象となるもの）。
            - obj_values: 目的関数の目標値リスト（Noneの場合もあり）。
            - directions: 目的関数の方向を示すリスト（1: 最大化, -1: 最小化）。
    """
    
    # 目的変数の列名を取得
    if target_cols is None or len(target_cols)==0:
        y_cols = model.target_cols
    else:
        y_cols = target_cols
        if len(y_cols)==0:
            raise ValueError("指定されたtargetがmodelにありません。")
    
    if len(obj_directions)!=len(y_cols):
        raise ValueError(f"targetとobj_directionsの長さが一致しません。 {len(y_cols)}-{len(obj_directions)}")

    # 数値変数の列名を取得
    x_cols = model.num_cols
    
    # 目的方向 (最大化: 1, 最小化: -1) を設定
    if len(obj_directions)==len(y_cols):
        directions = [set_direction(d) for d in obj_directions]
    else:
        directions = ["maximize"]*len(y_cols)
    
    # 目的関数の目標値を設定 (最小化・最大化以外の場合に使う)
    if len(obj_directions)==len(y_cols):
        obj_values = [set_obj_value(d) for d in obj_directions]
    else:
        obj_values = [None]*len(y_cols)
    
    # 目的変数の列名を、obj_valuesがNoneでないものに限定して取得
    obj_cols = [y_cols[i] for i, v in enumerate(obj_values) if v is not None]

    if len(bounds_min)!=len(x_cols):
        bounds_min=list(model.X[model.num_cols].min().values)
    if len(bounds_max)!=len(x_cols):
        bounds_max=list(model.X[model.num_cols].max().values)

    if len(steps)!=len(x_cols):
        steps = list(model.X[model.num_cols].apply(set_step, axis=0))

    # データ型を取得
    if len(dtypes)!=len(x_cols):
        dtypes = list(model.X[model.num_cols].dtypes.apply(dtype2str))

    if len(fix_values)!=len(x_cols + model.cat_cols+model.smiles_cols+model.comp_cols):
        fix_values = [None]*len(x_cols + model.cat_cols+model.smiles_cols+model.comp_cols)

    # 数値・カテゴリ変数の範囲、データ型、ステップをDataFrameに格納
    df_range = pd.DataFrame(
        {
            'min': bounds_min + [None] * len(model.cat_cols+model.smiles_cols+model.comp_cols),
            'max': bounds_max + [None] * len(model.cat_cols+model.smiles_cols+model.comp_cols),
            'dtype': dtypes + ['object'] * len(model.cat_cols+model.smiles_cols+model.comp_cols),
            'step': steps + [None] * len(model.cat_cols+model.smiles_cols+model.comp_cols),
            'fix':fix_values
        },
        index=x_cols + model.cat_cols+model.smiles_cols+model.comp_cols
    )
    
    return df_range, y_cols, x_cols+model.cat_cols+model.smiles_cols+model.comp_cols, obj_cols, obj_values, directions

def get_sampler(
    sampler_type: str
) -> Optional[object]:
    """
    指定されたサンプラータイプに応じてOptunaのサンプラーを返す関数。
    
    Args:
        sampler_type (str): 使用するサンプラーのタイプ。以下の値を取る:
                            'CmaEs', 'GP', 'TPE', 'MOTPE', 'NSGAII', 'NSGAIII'
    
    Returns:
        Optional[object]: 選択されたサンプラーオブジェクト。無効なサンプラータイプの場合はNone。
    """
    
    # サンプラータイプと対応するサンプラークラスを辞書で定義
    sampler_dict = {
        'CmaEs': CmaEsSampler,
        'GP': GPSampler,
        'QMS': QMCSampler,
        'TPE': TPESampler,
        'MOTPE': TPESampler,
        'NSGAII': NSGAIISampler,
        'NSGAIII': NSGAIIISampler,
        # 'Auto': optunahub.load_module(package="samplers/auto_sampler").AutoSampler
    }
    
    # サンプラーを取得
    sampler = sampler_dict.get(sampler_type, None)
    
    if sampler is not None:
        # 必要に応じてサンプラーの設定を変更可能
        sampler = sampler(
            # constraints_func=lambda trial: trial.user_attrs.get("constraints", None)
        )
    else:
        # 不正なサンプラータイプが指定された場合の警告やエラーメッセージを追加
        print(f"Warning: Invalid sampler type '{sampler_type}' specified.")
    
    return sampler

def suggest_parameter(
    trial: optuna.trial.Trial,
    param_name: str,
    dtype: str,
    param_min: float,
    param_max: float,
    step: Optional[float] = None,
    categories: Optional[List] = None
) -> float:
    """
    データ型に基づいてOptunaのサンプル関数を実行し、パラメータをサンプリングします。

    Args:
        trial (optuna.trial.Trial): Optunaのトライアルオブジェクト。
        param_name (str): サンプリング対象のパラメータ名。
        dtype (str): データ型 ('float', 'int', 'object')。
        param_min (float): サンプリングの最小値。
        param_max (float): サンプリングの最大値。
        step (Optional[float]): サンプリングのステップサイズ。float/int用。
        categories (Optional[List]): カテゴリカル変数の候補リスト。

    Returns:
        float: サンプリングされたパラメータ値。
    """
    if dtype == 'float':
        return trial.suggest_float(param_name, param_min, param_max, step=step)
    elif dtype == 'int':
        return trial.suggest_int(param_name, param_min, param_max, step=step)
    elif dtype == 'object' and categories:
        return trial.suggest_categorical(param_name, categories)
    else:
        raise ValueError(f"Unsupported data type: {dtype}")

def objective(
    trial: optuna.trial.Trial,
    model,
    cat_dict: Dict[str, List],
    y_cols: List[str],
    obj_values: List[Optional[float]],
    df_range: pd.DataFrame,
    x_cols: List[str],
    constraint_cols: List[str],
    constraint_value: float
) -> Tuple:
    """
    Optunaの目的関数。探索範囲と制約条件に基づいてパラメータをサンプリングし、モデルの予測結果を返す。

    Args:
        trial (optuna.trial.Trial): Optunaのトライアルオブジェクト。
        model: 使用する予測モデル。
        cat_dict (Dict[str, List]): カテゴリカル変数の辞書。
        obj_values (List[Optional[float]]): 目的関数の値。
        df_range (pd.DataFrame): 探索範囲とデータ型、ステップサイズが格納されたDataFrame。
        x_cols (List[str]): サンプリング対象の特徴量カラム。
        constraint_cols (List[str]): 制約を持つ特徴量カラム。
        constraint_value (float): 制約値の合計。

    Returns:
        Tuple: モデルによる予測値を返す。
    """
    
    x = pd.DataFrame([[None] * len(x_cols)], columns=x_cols)

    fix_const_value = df_range.loc[(df_range.index.isin(constraint_cols))&(~df_range['fix'].isna())]['fix'].sum()
    not_fix_const_cols = df_range.loc[(df_range.index.isin(constraint_cols))&(df_range['fix'].isna())].index.tolist()

    # 制約変数のサンプリング
    for t in x_cols:
        dtype = df_range.loc[t]['dtype']
        min_val, max_val, step = df_range.loc[t]['min'], df_range.loc[t]['max'], df_range.loc[t]['step']
        fix_value = df_range.loc[t]['fix']
        is_fix_value = df_range.loc[t][['fix']].isna()[0]

        if is_fix_value:
            if dtype == 'object':
                # カテゴリ変数のサンプリング
                x[t] = suggest_parameter(
                    trial,
                    t,
                    dtype,
                    None,
                    None,
                    categories=cat_dict[t]#[0]
                    )
            else:
                # 通常のサンプリング
                x[t] = suggest_parameter(
                    trial,
                    t,
                    dtype,
                    min_val,
                    max_val,
                    step
                    )
        else:
            x[t] = fix_value

    x = adjust_rows_to_sum_with_constraints(
        x,
        x_cols,
        df_range['fix'].values,
        constraint_cols,
        constraint_value,
        df_range
        )
    # constraints
    trial.set_user_attr(
        "constraints",
        [
            x[constraint_cols].sum().sum() - constraint_value,
            -x[constraint_cols].sum().sum() + constraint_value
            ]
            )
    # trial.set_user_attr("constraints", [(x[constraint_cols].sum().sum() - constraint_value)**2])

    # モデルによる予測を実行
    pred = model.predict_objective(x, obj_values=obj_values)
    pred = pred[y_cols]
    return tuple(pred.values.ravel())

def get_result(
    model,
    study,
    df_range,
    constraint_cols,
    constraint_value,
    x_cols: List[str],
    y_cols: List[str],
    obj_cols: List[str],
    fix_values: List,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    モデルの最良試行結果と、全試行データの予測値および目的スコアを返します。

    Args:
        model: 予測モデル。
        study: Optunaのstudyオブジェクト。
        x_cols (List[str]): 入力特徴量のカラム名リスト。
        y_cols (List[str]): 予測ターゲットのカラム名リスト。
        obj_cols (List[str]): 目的列（スコア計算対象）リスト。

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: 最良試行結果と全試行データのデータフレーム。
    """
    
    # 最良試行データの取得と結果の予測
    result = pd.DataFrame(
         [sum([
             [trial.params[x] if fix_values[i] is None else fix_values[i]] for i, x in enumerate(x_cols)
             ]+[trial.values],[]) for trial in study.best_trials],
        columns=x_cols+y_cols
    )

    result = adjust_rows_to_sum_with_constraints(
        result,
        x_cols,
        fix_values,
        constraint_cols,
        constraint_value,
        df_range
        )

    # 予測値の計算
    result[[f'pred_{c}' for c in y_cols]] = model.predict(result[x_cols])[y_cols]
    
    # 全試行データの取得と予測
    df_trials = study.trials_dataframe()[
        ['params_' + c for i, c in enumerate(x_cols) if fix_values[i] is None] + [f'values_{i}' for i in range(len(y_cols))]
        ] if len(y_cols) > 1 else study.trials_dataframe()[
            ['params_' + c for i, c in enumerate(x_cols) if fix_values[i] is None] + ['value']
            ]

    for i, c in enumerate(x_cols):
        if fix_values[i] is not None:
            df_trials['params_' + c] = fix_values[i]
    df_trials = df_trials[
        ['params_' + c for c in x_cols] + [f'values_{i}' for i in range(len(y_cols))]
        ] if len(y_cols) > 1 else df_trials[
            ['params_' + c for c in x_cols] + ['value']
            ]
    df_trials.columns = x_cols + y_cols

    df_trials = adjust_rows_to_sum_with_constraints(
        df_trials,
        x_cols,
        fix_values,
        constraint_cols,
        constraint_value,
        df_range
        )

    # 全試行データの予測値と目的スコアの計算
    pred = model.predict(df_trials[x_cols], proba=True)
    df_trials[[f'pred_{c}' for c in pred.columns]] = pred
    return result[x_cols + [f'pred_{c}' for c in y_cols]], df_trials