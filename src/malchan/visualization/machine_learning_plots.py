import numpy as np
import pandas as pd
import plotly.graph_objects as go
from typing import List, Optional, Union, Dict, Tuple, Callable, Any

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import confusion_matrix

from machine_learning.models import MLModelPipeline
# from models.xai import get_shap_scatter

import warnings
warnings.simplefilter('ignore')

def show_importances(
    feature_names: List[str],
    feature_importances: np.ndarray,
    n_bar: int = 15
) -> go.Figure:
    """
    特徴量の重要度を棒グラフで表示する関数。

    Args:
        feature_names (List[str]): 特徴量の名前のリスト。
        feature_importances (np.ndarray): 特徴量の重要度を示す配列。
        n_bar (int): 表示するバーの数（上位の特徴量数）。

    Returns:
        go.Figure: 棒グラフを含む Plotly 図オブジェクト。
    """
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

def _yy_plot(
    y,
    pred_train,
    pred_test=None,
    residual=False
):
    if residual:
        y_train_value = pred_train.values.ravel() - y.values.ravel()
        if pred_test is not None:
            y_test_value = pred_test.values.ravel() - y.values.ravel()
    else:
        y_train_value = pred_train.values.ravel()
        if pred_test is not None:
            y_test_value = pred_test.values.ravel()
    
    # Plotly 図オブジェクトを作成
    fig = go.Figure()

    # 実際の値と予測値の散布図を追加
    fig.add_trace(
        go.Scatter(
            x=y.values.ravel(),
            y=y_train_value,
            mode='markers',
            marker_color='blue',
            name='train'
        )
    )

    if pred_test is not None:
        fig.add_trace(
            go.Scatter(
                x=y.values.ravel(),
                y=y_test_value,
                mode='markers',
                marker_color='green',
                name='test'
            )
        )

    # 対角線（実際の値 = 予測値）を追加
    if residual:
        x_min_val = min(y.values.ravel())
        x_max_val = max(y.values.ravel())
        y_min_val = 0
        y_max_val = 0
    else:
        x_min_val = min([min(y.values.ravel()), min(y_train_value)])
        x_max_val = max([max(y.values.ravel()), max(y_train_value)])
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
    target: str,
    cv: bool = False,
    residual: bool = False,
    train_test="train",
) -> go.Figure:
    """
    実際の値と予測値をプロットする散布図と対角線を作成する関数。

    Args:
        _model (MLModelPipeline): モデルパイプラインオブジェクト。

    Returns:
        go.Figure: 散布図と対角線を含む Plotly 図オブジェクト。
    """
    # 予測値を取得
    if not cv:
        if model.models[target].task=="classification":
            cm = confusion_matrix(model.models[target].y, model.predict()[target])
            labels = model.models[target].target_items
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
                model.models[target].y,
                model.predict()[target],
                None,
                residual
            )
    else:
        if model.models[target].task=="classification":
            cm = confusion_matrix(model.models[target].y.astype(int), model.models[target].cv_preds[train_test].astype(int))
            labels = model.models[target].target_items
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
                    model.models[target].y,
                    model.models[target].cv_preds["train"],
                    model.models[target].cv_preds["test"],
                    residual
            )         
    return fig

def show_pd_and_ice(
    X_PD: np.ndarray,
    target_col: str,
    xticks: np.ndarray,
    X=None,
    y=None,
    ice: bool = True,
    col_idx=-1
) -> go.Figure:
    """
    部分依存プロット（Partial Dependence Plot）と個別条件期待値（ICE）を表示する関数。

    Args:
        X_PD (np.ndarray): 部分依存プロットデータ（ICEを含む）。
        xticks (np.ndarray): 特徴量の値（x軸）。
        ice (bool): ICEを表示するかどうかのフラグ（デフォルトは True）。

    Returns:
        go.Figure: 部分依存プロットとICEを含む Plotly 図オブジェクト。
    """
    if len(X_PD.shape)==3:
        X_PD = X_PD[:,:,col_idx]
        y = (y.values.ravel()==col_idx).astype(int)
    else:
        y=y.values.ravel()
    
    # Plotly 図オブジェクトを作成
    fig = go.Figure()
    
    # ICEプロットを追加するかどうかを判断
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
        
    # 部分依存プロットの平均値を追加
    fig.add_trace(
        go.Scatter(
            x=xticks,
            y=X_PD.mean(axis=1),
            mode='lines',
            line=dict(color='orange', width=2),
            name='Partial Dependence('+target_col+')'
        )
    )

    if (X is not None)&(y is not None):
        fig.add_trace(
            go.Scatter(
                x=X[target_col].values.ravel(),
                y=y,
                mode='markers',
                marker_color="red",
                name='Autual Data('+target_col+')',
                marker=dict(size=10,)
            )
        )
    

    # グラフのレイアウトを設定
    fig.update_layout(
        title='Partial Dependence Plot with ICE',
        xaxis_title=target_col,
        yaxis_title='Predicted Value',
        # showlegend=False,
        width=650,
        height=600
    )

    return fig

def show_pd_2d(
    X_PD: np.ndarray,
    target_cols: List[str],
    xticks1: np.ndarray,
    xticks2: np.ndarray,
    X=None,
    y=None,
    col_idx=-1
) -> go.Figure:
    """
    部分依存プロット（Partial Dependence Plot）と個別条件期待値（ICE）を表示する関数。

    Args:
        X_PD (np.ndarray): 部分依存プロットデータ（ICEを含む）。
        xticks (np.ndarray): 特徴量の値（x軸）。
        ice (bool): ICEを表示するかどうかのフラグ（デフォルトは True）。

    Returns:
        go.Figure: 部分依存プロットとICEを含む Plotly 図オブジェクト。
    """

    if len(X_PD.shape)==3:
        X_PD = X_PD[:,:,col_idx]
    
    # Plotly 図オブジェクトを作成
    fig = go.Figure() 
    
    # 部分依存プロットの平均値を追加
    fig.add_trace(
        go.Contour(
            z=X_PD.mean(axis=1),  # 取得関数の値
            x=xticks1.ravel(),  # グリッドのx軸
            y=xticks2.ravel(),  # グリッドのy軸
            ncontours=25,  # 等高線の数
            showscale=True,  # カラーバーを表示
            colorbar=dict(
                title="予測値",
                lenmode="pixels",
                len=200
            ),  # カラーバー設定
            contours_coloring='heatmap',  # 等高線のカラーリング方法
            colorscale='RdBu_r',  # カラースケール
            hoverinfo='none'  # ホバー情報を非表示
        )
    )

    if X is not None:
        fig.add_trace(
            go.Scatter(
                x=X[target_cols[0]].values,
                y=X[target_cols[1]].values,
                mode='markers',
                name='Actual Data',
                marker=dict(
                    color=y.values.ravel(),
                    colorscale='RdBu_r',
                    size=10,
                    line=dict(
                        color='black',  # エッジの色
                        width=1         # エッジの太さ
                    )
            )
        )
        )

    # グラフのレイアウトを設定
    fig.update_layout(
        title='Partial Dependence Plot with ICE',
        xaxis_title=target_cols[0],
        yaxis_title=target_cols[1],
        showlegend=False,
        width=650,
        height=600
    )

    return fig

def show_shap_scatter(
    X_shappd,
    rawX: pd.DataFrame,
    shap_values: np.ndarray,
    target_col: str,
    interactive_col: Optional[str] = None,
    unique_dict: Dict[str, list] = {},
    target_item=None,
    target_items: Dict[str, list] = None,
) -> go.Figure:
    """
    SHAP値を用いた散布図を表示する関数。

    Args:
        X (pd.DataFrame): 特徴量データ。
        shap_values (np.ndarray): SHAP値（サンプル数 x 特徴量数）。
        target_col (str): ターゲット列の名前。
        modelname (Optional[str]): モデルの名前（デフォルトは None）。
        interactive_col (Optional[str]): インタラクティブに色分けする特徴量（デフォルトは None）。
        unique_dict (Dict[str, list]): 特徴量のユニーク値に関する辞書（デフォルトは空辞書）。

    Returns:
        go.Figure: SHAP値の散布図を含む Plotly 図オブジェクト。
    """
    # SHAP値がNoneの場合のエラーハンドリング
    if shap_values is None:
        raise ValueError("SHAP値がNoneです。正しいSHAP値を提供してください。")
    
    # SHAP値とターゲット列に対するデータを取得
    X_plot = X_shappd[target_col]
    if target_item is not None:
        shap_col = 'shap_'+str(target_item)
    elif target_items is not None:
        shap_col = 'shap_'+target_items[X_plot.shape[-1]-2]
    elif X_plot.shape[-1]>2:
        shap_col = X_plot.columns[-1]
    else:
        shap_col = 'shap'
    
    # Plotly 図オブジェクトを作成
    fig = go.Figure()
    
    if interactive_col is not None and interactive_col in unique_dict.keys():
        # インタラクティブに色分けする特徴量がある場合
        for c in sorted(rawX[interactive_col].unique()):
            fig.add_trace(
                go.Scatter(
                    x=X_plot[rawX[interactive_col] == c][target_col],
                    y=X_plot[rawX[interactive_col] == c][shap_col],
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
        # インタラクティブな色分けがない場合
        fig.add_trace(
            go.Scatter(
                x=X_plot[target_col],
                y=X_plot[shap_col],
                marker=dict(
                    color=X_shappd[interactive_col][interactive_col] if interactive_col is not None else 'blue',
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

def show_shap_beeswarm(
    X: pd.DataFrame,
    shap_values: np.ndarray,
    n_shap_top: int = 5,
    cat_cols: List[str] = [],
    col_idx=0
) -> go.Figure:
    """
    SHAP値のビーズウォームプロットを作成する関数。

    Args:
        X (pd.DataFrame): 特徴量データ。
        shap_values (np.ndarray): SHAP値（サンプル数 x 特徴量数）。
        n_shap_top (int): プロットに表示する上位の特徴量の数。
        cat_cols (List[str]): カテゴリカル特徴量の列名リスト。

    Returns:
        go.Figure: SHAP値のビーズウォームプロットを含む Plotly 図オブジェクト。
    """
    # SHAP値がNoneの場合のエラーハンドリング
    if shap_values is None:
        raise ValueError("SHAP値がNoneです。正しいSHAP値を提供してください。")

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
    
    # SHAP値とスケーリングされたデータの結合
    df_melt = pd.concat(
        [
            pd.DataFrame(
                shap_values,
                columns=X.columns
            )[shap_top].melt().rename(columns={'variable': 'target', 'value': 'shap'}),
            X_scaled[shap_top].melt()['value']
        ],
        axis=1
    )
    
    # ビーズウォームプロットの作成
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df_melt['shap'],
            y=np.repeat(np.arange(n_shap_top), len(X)) + np.random.rand(len(X) * n_shap_top) * 0.2,
            mode='markers',
            marker=dict(
                color=-df_melt['value'],
                colorscale='RdBu',
                size=10,
            ),
            opacity=0.7
        )
    )
    
    # レイアウトの設定
    fig.update_layout(
        width=600,
        height=150 + 50 * n_shap_top,
        yaxis=dict(
            tickmode='array',
            tickvals=np.arange(n_shap_top),
            ticktext=shap_top,
            automargin=True
        )
    )
    return fig