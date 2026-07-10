import numpy as np
import pandas as pd
import plotly.graph_objects as go
from typing import Any, Dict, List, Optional, Tuple

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import confusion_matrix

from malchan.models import MLModelPipeline
# from models.xai import get_shap_scatter

import warnings
warnings.simplefilter('ignore')

def _resolve_model_target(model: Optional[MLModelPipeline], target: Optional[str], object_col: Optional[str]) -> Tuple[Optional[Any], Optional[str]]:
    """Resolve a child model from a model pipeline.

    Args:
        model (Optional[MLModelPipeline]): Model pipeline that has a ``models`` mapping.
        target (Optional[str]): Output column name used as a key of ``model.models``.
        object_col (Optional[str]): Alias for ``target`` kept for visualization APIs.

    Returns:
        Tuple[Optional[Any], Optional[str]]: The selected child model and resolved key.

    Raises:
        ValueError: If only one of ``model`` and output key is provided.
        KeyError: If the resolved output key does not exist in ``model.models``.
    """
    resolved_target = target if target is not None else object_col
    if model is None:
        if resolved_target is not None:
            raise ValueError("modelを指定しない場合、target/object_colは指定できません。")
        return None, None
    if resolved_target is None:
        raise ValueError("modelを指定する場合、targetまたはobject_colを指定してください。")
    if resolved_target not in model.models:
        raise KeyError(f"{resolved_target!r} is not found in model.models.")
    return model.models[resolved_target], resolved_target

def show_importances(
    feature_names: Optional[List[str]] = None,
    feature_importances: Optional[np.ndarray] = None,
    n_bar: int = 15,
    model: Optional[MLModelPipeline] = None,
    target: Optional[str] = None,
    object_col: Optional[str] = None,
    importance_type: str = "model",
) -> go.Figure:
    """特徴量の重要度を棒グラフで表示します。

    Args:
        feature_names (Optional[List[str]]): 特徴量の名前のリスト。
        feature_importances (Optional[np.ndarray]): 特徴量の重要度。
        n_bar (int): 表示する上位特徴量数。
        model (Optional[MLModelPipeline]): 学習済みモデルパイプライン。
        target (Optional[str]): 可視化対象の目的変数名。
        object_col (Optional[str]): ``target`` の別名。
        importance_type (str): ``model`` 指定時に取得する重要度の種類。

    Returns:
        go.Figure: 重要度の棒グラフ。

    Raises:
        ValueError: 必要な入力が不足している場合。
    """
    child_model, _ = _resolve_model_target(model, target, object_col)
    if child_model is not None:
        feature_names = child_model.feature_names
        importance_getter = getattr(child_model, f"{importance_type}_importance")
        feature_importances = importance_getter()
    if feature_names is None or feature_importances is None:
        raise ValueError("feature_names/feature_importances、またはmodelとtarget/object_colを指定してください。")
    # 特徴量重要度でソートするインデックスを取得
    sort_idx = np.argsort(-np.abs(feature_importances))

    # Plotly 図オブジェクトを作成
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=feature_importances[sort_idx][:n_bar],
            y=np.array(feature_names)[sort_idx][:n_bar],
            orientation='h'
        )
    )

    # グラフのレイアウトを設定
    fig.update_layout(
        width=600,
        height=150 + 50 * n_bar,
        title='Feature Importances',
        xaxis_title='Importance',
        yaxis_title='Features',
        xaxis_tickangle=-45,
        yaxis=dict(autorange='reversed')
    )

    return fig

def _get_training_X(child_model: Any) -> pd.DataFrame:
    """Return training feature values for visualization.

    Args:
        child_model (Any): Single-output model object. It may store features
            directly in ``X`` or indirectly through ``_get_X()`` when it is
            owned by a multi-output pipeline.

    Returns:
        pd.DataFrame: Training feature values.

    Raises:
        ValueError: If training feature values cannot be found.
    """
    X = child_model._get_X() if hasattr(child_model, "_get_X") else getattr(child_model, "X", None)
    if X is None:
        raise ValueError("学習データの説明変数が見つかりません。モデルをfitした後に実行してください。")
    return X

def _get_training_y(child_model: Any) -> pd.DataFrame:
    """Return training target values for visualization.

    Args:
        child_model (Any): Single-output model object. It may store targets
            directly in ``y`` or indirectly through ``_get_y()`` when it is
            owned by a multi-output pipeline.

    Returns:
        pd.DataFrame: Training target values.

    Raises:
        ValueError: If training target values cannot be found.
    """
    y = child_model._get_y() if hasattr(child_model, "_get_y") else getattr(child_model, "y", None)
    if y is None:
        raise ValueError("学習データの目的変数が見つかりません。モデルをfitした後に実行してください。")
    return y

def _yy_plot(
    y,
    pred_train,
    pred_test=None,
    residual=False
):
    """Create an actual-vs-predicted scatter plot for regression models.

    Args:
        y: Actual target values. Accepts pandas objects or array-like values.
        pred_train: Training predictions. Accepts pandas objects or array-like values.
        pred_test: Test or validation predictions. Defaults to None.
        residual (bool): Whether to plot residuals instead of predictions.

    Returns:
        go.Figure: Plotly figure containing the regression diagnostic plot.
    """
    y_values = np.asarray(y).ravel()
    pred_train_values = np.asarray(pred_train).ravel()

    if residual:
        y_train_value = pred_train_values - y_values
        if pred_test is not None:
            y_test_value = np.asarray(pred_test).ravel() - y_values
    else:
        y_train_value = pred_train_values
        if pred_test is not None:
            y_test_value = np.asarray(pred_test).ravel()
    
    # Plotly 図オブジェクトを作成
    fig = go.Figure()

    # 実際の値と予測値の散布図を追加
    fig.add_trace(
        go.Scatter(
            x=y_values,
            y=y_train_value,
            mode='markers',
            marker_color='blue',
            name='train'
        )
    )

    if pred_test is not None:
        fig.add_trace(
            go.Scatter(
                x=y_values,
                y=y_test_value,
                mode='markers',
                marker_color='green',
                name='test'
            )
        )

    # 対角線（実際の値 = 予測値）を追加
    if residual:
        x_min_val = min(y_values)
        x_max_val = max(y_values)
        y_min_val = 0
        y_max_val = 0
    else:
        x_min_val = min([min(y_values), min(y_train_value)])
        x_max_val = max([max(y_values), max(y_train_value)])
        y_min_val = x_min_val
        y_max_val = x_max_val

    fig.add_trace(
        go.Scatter(
            x=[x_min_val, x_max_val],
            y=[y_min_val, y_max_val],
            mode='lines',
            line=dict(color='orange', width=2),
            name='Ideal Fit'
        )
    )

    # グラフのレイアウトを設定
    fig.update_layout(
        title='Actual vs Predicted',
        xaxis_title='Actual Values',
        yaxis_title='Predicted Values',
        showlegend=True,
        width=650,
        height=600
    )

    return fig

def yy_plot_ml(
    model: MLModelPipeline,
    target: Optional[str] = None,
    cv: bool = False,
    residual: bool = False,
    train_test="train",
    object_col: Optional[str] = None,
) -> go.Figure:
    """実測値と予測値、または分類の混同行列を表示します。

    Args:
        model (MLModelPipeline): 学習済みモデルパイプライン。
        target (Optional[str]): 可視化対象の目的変数名。
        cv (bool): Cross validation の予測値を使うかどうか。
        residual (bool): 回帰で残差を表示するかどうか。
        train_test (str): 分類CVで表示する予測値の区分。
        object_col (Optional[str]): ``target`` の別名。

    Returns:
        go.Figure: Plotlyの図オブジェクト。
    """
    child_model, target = _resolve_model_target(model, target, object_col)
    y = _get_training_y(child_model)
    # 予測値を取得
    if not cv:
        if child_model.task=="classification":
            cm = confusion_matrix(y, model.predict()[target])
            labels = child_model.target_items
            fig = go.Figure(
                data=go.Heatmap(
                    z=cm,
                    x=labels,
                    y=labels,
                    colorscale="Blues",
                    text=cm,
                    texttemplate="%{text}"
                )
            )
    
            fig.update_layout(
                xaxis_title="predict",
                yaxis_title="actual",
                yaxis=dict(autorange="reversed")
            )
        else:
            fig = _yy_plot(
                y,
                model.predict()[target],
                None,
                residual
            )
    else:
        if child_model.task=="classification":
            cm = confusion_matrix(y.astype(int), child_model.cv_preds[train_test].astype(int))
            labels = child_model.target_items
            fig = go.Figure(
                data=go.Heatmap(
                    z=cm,
                    x=labels,
                    y=labels,
                    colorscale="Blues",
                    text=cm,
                    texttemplate="%{text}"
                )
            )
            fig.update_layout(
                xaxis_title="predict",
                yaxis_title="actual",
                yaxis=dict(autorange="reversed")
            )
        else:
            fig = _yy_plot(
                    y,
                    child_model.cv_preds["train"],
                    child_model.cv_preds["test"],
                    residual
            )         
    return fig

def show_pd_and_ice(
    X_PD: Optional[np.ndarray] = None,
    target_col: Optional[str] = None,
    xticks: Optional[np.ndarray] = None,
    X=None,
    y=None,
    ice: bool = True,
    col_idx=-1,
    model: Optional[MLModelPipeline] = None,
    target: Optional[str] = None,
    object_col: Optional[str] = None,
) -> go.Figure:
    """部分依存プロット（PD）とICEを表示します。

    Args:
        X_PD (Optional[np.ndarray]): 事前計算済みのPD/ICEデータ。
        target_col (Optional[str]): x軸に表示する特徴量名。
        xticks (Optional[np.ndarray]): ``target_col`` のグリッド値。
        X (Optional[pd.DataFrame]): 実データの特徴量。指定時は散布図を重ねます。
        y (Optional[pd.DataFrame]): 実データの目的変数。指定時は散布図を重ねます。
        ice (bool): ICE線を表示するかどうか。
        col_idx (int): 分類モデルの出力次元インデックス。
        model (Optional[MLModelPipeline]): 学習済みモデルパイプライン。
        target (Optional[str]): 可視化対象の目的変数名。
        object_col (Optional[str]): ``target`` の別名。

    Returns:
        go.Figure: PD/ICEのPlotly図オブジェクト。

    Raises:
        ValueError: 必要な入力が不足している場合。
    """
    child_model, _ = _resolve_model_target(model, target, object_col)
    if child_model is not None:
        if target_col is None:
            raise ValueError("target_colを指定してください。")
        X_PD, xticks = child_model.get_pd_and_ice(target_col)
        X = _get_training_X(child_model)
        y = _get_training_y(child_model)
    if X_PD is None or target_col is None or xticks is None:
        raise ValueError("X_PD/target_col/xticks、またはmodelとtarget/object_colとtarget_colを指定してください。")

    if len(X_PD.shape)==3:
        X_PD = X_PD[:,:,col_idx]
        if y is not None:
            y = (np.asarray(y).ravel()==col_idx).astype(int)
    elif y is not None:
        y = np.asarray(y).ravel()

    fig = go.Figure()

    if ice:
        for i in range(X_PD.shape[1]):
            fig.add_trace(
                go.Scatter(
                    x=xticks,
                    y=X_PD[:, i],
                    mode='lines',
                    line=dict(color='rgba(0,100,200,0.2)'),
                    name=None,
                    showlegend=False
                )
            )

    fig.add_trace(
        go.Scatter(
            x=xticks,
            y=X_PD.mean(axis=1),
            mode='lines',
            line=dict(color='orange', width=2),
            name='Partial Dependence('+target_col+')'
        )
    )

    if (X is not None) and (y is not None):
        fig.add_trace(
            go.Scatter(
                x=X[target_col].values.ravel(),
                y=y,
                mode='markers',
                marker_color="red",
                name='Actual Data('+target_col+')',
                marker=dict(size=10,)
            )
        )

    fig.update_layout(
        title='Partial Dependence Plot with ICE',
        xaxis_title=target_col,
        yaxis_title='Predicted Value',
        width=650,
        height=600
    )

    return fig


def show_pd_2d(
    X_PD: Optional[np.ndarray] = None,
    target_cols: Optional[List[str]] = None,
    xticks1: Optional[np.ndarray] = None,
    xticks2: Optional[np.ndarray] = None,
    X=None,
    y=None,
    col_idx=-1,
    model: Optional[MLModelPipeline] = None,
    target: Optional[str] = None,
    object_col: Optional[str] = None,
) -> go.Figure:
    """2次元の部分依存プロットを表示します。

    Args:
        X_PD (Optional[np.ndarray]): 事前計算済みの2次元PDデータ。
        target_cols (Optional[List[str]]): x軸とy軸に表示する2つの特徴量名。
        xticks1 (Optional[np.ndarray]): 1つ目の特徴量のグリッド値。
        xticks2 (Optional[np.ndarray]): 2つ目の特徴量のグリッド値。
        X (Optional[pd.DataFrame]): 実データの特徴量。指定時は散布図を重ねます。
        y (Optional[pd.DataFrame]): 実データの目的変数。散布図の色に使います。
        col_idx (int): 分類モデルの出力次元インデックス。
        model (Optional[MLModelPipeline]): 学習済みモデルパイプライン。
        target (Optional[str]): 可視化対象の目的変数名。
        object_col (Optional[str]): ``target`` の別名。

    Returns:
        go.Figure: 2次元PDのPlotly図オブジェクト。

    Raises:
        ValueError: 必要な入力が不足している場合。
    """
    child_model, _ = _resolve_model_target(model, target, object_col)
    if child_model is not None:
        if target_cols is None:
            raise ValueError("target_colsを指定してください。")
        X_PD, xticks1, xticks2 = child_model.get_pd_2d(target_cols=target_cols)
        X = _get_training_X(child_model)
        y = _get_training_y(child_model)
    if X_PD is None or target_cols is None or xticks1 is None or xticks2 is None:
        raise ValueError("X_PD/target_cols/xticks1/xticks2、またはmodelとtarget/object_colとtarget_colsを指定してください。")

    if len(X_PD.shape)==3:
        X_PD = X_PD[:,:,col_idx]

    fig = go.Figure()
    fig.add_trace(
        go.Contour(
            z=X_PD.mean(axis=1),
            x=xticks1.ravel(),
            y=xticks2.ravel(),
            ncontours=25,
            showscale=True,
            colorbar=dict(title="予測値", lenmode="pixels", len=200),
            contours_coloring='heatmap',
            colorscale='RdBu_r',
            hoverinfo='none'
        )
    )

    if X is not None:
        marker_color = np.asarray(y).ravel() if y is not None else 'red'
        fig.add_trace(
            go.Scatter(
                x=X[target_cols[0]].values,
                y=X[target_cols[1]].values,
                mode='markers',
                name='Actual Data',
                marker=dict(
                    color=marker_color,
                    colorscale='RdBu_r',
                    size=10,
                    line=dict(color='black', width=1)
                )
            )
        )

    fig.update_layout(
        title='Partial Dependence Plot with ICE',
        xaxis_title=target_cols[0],
        yaxis_title=target_cols[1],
        xaxis=dict(range=[float(np.nanmin(xticks1)), float(np.nanmax(xticks1))]),
        yaxis=dict(range=[float(np.nanmin(xticks2)), float(np.nanmax(xticks2))]),
        showlegend=False,
        width=650,
        height=600
    )

    return fig

def show_shap_scatter(
    X_shappd=None,
    rawX: Optional[pd.DataFrame] = None,
    shap_values: Optional[np.ndarray] = None,
    target_col: Optional[str] = None,
    interactive_col: Optional[str] = None,
    unique_dict: Optional[Dict[str, list]] = None,
    target_item=None,
    target_items: Optional[Dict[str, list]] = None,
    model: Optional[MLModelPipeline] = None,
    target: Optional[str] = None,
    object_col: Optional[str] = None,
) -> go.Figure:
    """SHAP値の散布図を表示します。

    Args:
        X_shappd: 事前計算済みのSHAP散布図用データ。
        rawX (Optional[pd.DataFrame]): 元の特徴量データ。
        shap_values (Optional[np.ndarray]): SHAP値。
        target_col (Optional[str]): x軸に表示する特徴量名。
        interactive_col (Optional[str]): 色分けに使う特徴量名。
        unique_dict (Optional[Dict[str, list]]): カテゴリ特徴量のユニーク値辞書。
        target_item: 分類モデルのクラス名。
        target_items (Optional[Dict[str, list]]): 分類モデルのクラス名一覧。
        model (Optional[MLModelPipeline]): 学習済みモデルパイプライン。
        target (Optional[str]): 可視化対象の目的変数名。
        object_col (Optional[str]): ``target`` の別名。

    Returns:
        go.Figure: SHAP散布図のPlotly図オブジェクト。

    Raises:
        ValueError: 必要な入力が不足している場合。
    """
    child_model, _ = _resolve_model_target(model, target, object_col)
    if child_model is not None:
        if target_col is None:
            raise ValueError("target_colを指定してください。")
        X_shappd = child_model.get_shap_scatter_data(target_col)
        rawX = _get_training_X(child_model)
        shap_values = child_model.shap_values
        unique_dict = child_model._shared_attr("unique_cols")
        target_items = getattr(child_model, "target_items", None)
    if shap_values is None:
        raise ValueError("SHAP値がNoneです。正しいSHAP値を提供してください。")
    if X_shappd is None or rawX is None or target_col is None:
        raise ValueError("X_shappd/rawX/target_col、またはmodelとtarget/object_colとtarget_colを指定してください。")
    unique_dict = unique_dict or {}

    # SHAP値とターゲット列に対するデータを取得
    if isinstance(X_shappd, dict):
        X_plot = X_shappd[target_col]
    else:
        X_plot = X_shappd
    if isinstance(X_plot, pd.Series):
        X_plot = X_plot.to_frame(name=target_col)
    if target_col not in X_plot.columns:
        raise ValueError(f"{target_col!r} is not found in SHAP scatter data.")

    if target_item is not None:
        shap_col = 'shap_'+str(target_item)
    elif target_items is not None:
        shap_suffixes = list(target_items)
        shap_col = 'shap_'+str(shap_suffixes[X_plot.shape[-1]-2])
    elif X_plot.shape[-1]>2:
        shap_col = X_plot.columns[-1]
    else:
        shap_col = 'shap'
    if shap_col not in X_plot.columns:
        raise ValueError(f"{shap_col!r} is not found in SHAP scatter data.")
    
    # Plotly 図オブジェクトを作成
    fig = go.Figure()
    
    interactive_values = None
    if interactive_col is not None:
        if interactive_col not in rawX.columns:
            raise ValueError(f"{interactive_col!r} is not found in rawX.")
        interactive_values = rawX[interactive_col].to_numpy()

    if interactive_col is not None and interactive_col in unique_dict.keys():
        # インタラクティブにカテゴリで色分けする特徴量がある場合
        for c in sorted(pd.Series(interactive_values).dropna().unique()):
            mask = interactive_values == c
            fig.add_trace(
                go.Scatter(
                    x=X_plot.loc[mask, target_col],
                    y=X_plot.loc[mask, shap_col],
                    name=c,
                    mode='markers',
                    marker=dict(size=10,)
                )
            )
        fig.update_layout(
            showlegend=True,
            width=600,
            height=600
        )
    else:
        # インタラクティブな色分けがない場合、または数値特徴量で色分けする場合
        marker_color = interactive_values if interactive_col is not None else 'blue'
        fig.add_trace(
            go.Scatter(
                x=X_plot[target_col],
                y=X_plot[shap_col],
                marker=dict(
                    color=marker_color,
                    colorscale='bluered',
                    colorbar=dict(
                        thickness=10
                    ),
                    size=10,
                ),
                mode='markers'
            )
        )
        fig.update_layout(
            showlegend=False,
            xaxis_title=target_col,
            yaxis_title='Shap Value',
            width=600,
            height=600
        )
    
    return fig


def _make_density_beeswarm_offsets(values: np.ndarray, row_center: float, max_spread: float = 0.35) -> np.ndarray:
    """SHAP値の局所密度に応じたビーズウォーム用y座標を作成します。

    Args:
        values (np.ndarray): 1つの特徴量に対応するSHAP値。
        row_center (float): 特徴量行の中心y座標。
        max_spread (float): 行中心から上下に広げる最大幅。

    Returns:
        np.ndarray: 入力順に対応するビーズウォーム用y座標。
    """
    values = np.asarray(values, dtype=float).ravel()
    offsets = np.zeros(values.shape[0], dtype=float)
    if values.size <= 1:
        return row_center + offsets

    finite_mask = np.isfinite(values)
    if not finite_mask.any():
        return row_center + offsets

    finite_values = values[finite_mask]
    value_range = np.nanmax(finite_values) - np.nanmin(finite_values)
    if value_range == 0:
        bin_indices = np.zeros(values.shape[0], dtype=int)
    else:
        bin_count = min(100, max(10, int(np.ceil(np.sqrt(finite_values.size) * 2))))
        bin_edges = np.linspace(np.nanmin(finite_values), np.nanmax(finite_values), bin_count + 1)
        bin_indices = np.digitize(values, bin_edges[1:-1], right=False)

    for bin_index in np.unique(bin_indices[finite_mask]):
        point_indices = np.flatnonzero(finite_mask & (bin_indices == bin_index))
        if point_indices.size <= 1:
            continue
        order = point_indices[np.argsort(values[point_indices], kind="mergesort")]
        ranks = np.arange(order.size, dtype=float)
        side = np.where((ranks % 2) == 0, 1.0, -1.0)
        layer = np.floor((ranks + 1.0) / 2.0)
        max_layer = max(layer.max(), 1.0)
        offsets[order] = side * layer / max_layer * max_spread

    return row_center + offsets

def show_shap_beeswarm(
    X: Optional[pd.DataFrame] = None,
    shap_values: Optional[np.ndarray] = None,
    n_shap_top: int = 5,
    cat_cols: Optional[List[str]] = None,
    col_idx=0,
    model: Optional[MLModelPipeline] = None,
    target: Optional[str] = None,
    object_col: Optional[str] = None,
) -> go.Figure:
    """SHAP値のビーズウォームプロットを表示します。

    Args:
        X (Optional[pd.DataFrame]): 特徴量データ。
        shap_values (Optional[np.ndarray]): SHAP値。
        n_shap_top (int): 表示する上位特徴量数。
        cat_cols (Optional[List[str]]): カテゴリカル特徴量の列名リスト。
        col_idx (int): 分類モデルの出力次元インデックス。
        model (Optional[MLModelPipeline]): 学習済みモデルパイプライン。
        target (Optional[str]): 可視化対象の目的変数名。
        object_col (Optional[str]): ``target`` の別名。

    Returns:
        go.Figure: SHAPビーズウォームプロットのPlotly図オブジェクト。

    Raises:
        ValueError: 必要な入力が不足している場合。
    """
    child_model, _ = _resolve_model_target(model, target, object_col)
    if child_model is not None:
        X = _get_training_X(child_model)
        shap_values = child_model.shap_values
        cat_cols = child_model._shared_attr("cat_cols")
    if X is None or shap_values is None:
        raise ValueError("X/shap_values、またはmodelとtarget/object_colを指定してください。")
    cat_cols = cat_cols or []

    if len(shap_values.shape)==3:
        shap_values = shap_values[:,:,col_idx]

    # SHAP値の絶対値の合計で上位特徴量を選択
    shap_top = X.columns[np.argsort(-np.abs(shap_values).sum(axis=0))][:n_shap_top][::-1]
    
    # 特徴量のスケーリング
    minmax_scaler = MinMaxScaler()
    X_scaled = X.copy()
    
    try:
        X_scaled = pd.DataFrame(
            minmax_scaler.fit_transform(X_scaled),
            columns=X_scaled.columns
        )
    except ValueError:
        # カテゴリカルデータの処理
        enc = OrdinalEncoder()
        X_scaled[cat_cols] = enc.fit_transform(X_scaled[cat_cols])
        X_scaled = pd.DataFrame(
            minmax_scaler.fit_transform(X_scaled),
            columns=X_scaled.columns
        )
    
    shap_plot = pd.DataFrame(shap_values, columns=X.columns)[shap_top]
    scaled_plot = X_scaled[shap_top]
    x_values = []
    y_values = []
    color_values = []
    for row_index, feature_name in enumerate(shap_top):
        feature_shap = shap_plot[feature_name].to_numpy()
        x_values.extend(feature_shap)
        y_values.extend(_make_density_beeswarm_offsets(feature_shap, row_center=row_index))
        color_values.extend(-scaled_plot[feature_name].to_numpy())
    
    # ビーズウォームプロットの作成
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=y_values,
            mode='markers',
            marker=dict(
                color=color_values,
                colorscale='RdBu',
                size=10,
            ),
            opacity=0.7
        )
    )
    
    # レイアウトの設定
    fig.update_layout(
        width=600,
        height=220 + 80 * n_shap_top,
        margin=dict(t=70, b=70),
        yaxis=dict(
            tickmode='array',
            tickvals=np.arange(n_shap_top),
            ticktext=shap_top,
            range=[-0.8, n_shap_top - 0.2],
            automargin=True
        )
    )
    return fig