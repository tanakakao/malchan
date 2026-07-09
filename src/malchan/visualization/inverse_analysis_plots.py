import numpy as np
import plotly.graph_objects as go
import optuna
from typing import List, Optional, Union, Dict, Tuple, Callable, Any

from malchan.models.training import fit_model, tune_model, cv_fit
from malchan.models.utils import get_cat_unique_values, feature_names_from_pipeline, label_encode
from malchan.models.explainability import get_shap_values, get_importances, get_pfi_values, get_shap_scatter, get_pd_and_ice, get_pd_and_ice_2d
from malchan.models.pipelines import make_pipeline

from malchan.visualization.machine_learning_plots import (
    _get_training_X,
    _get_training_y,
    show_importances,
    yy_plot_ml,
    show_pd_and_ice,
    show_pd_2d,
    show_shap_scatter,
    show_shap_beeswarm,
)

def show_ia_result_with_pd(
    df_trials,
    model,
    target_col,
    feature_col,
    target_item
):
    """Show inverse-analysis candidates over a one-dimensional PD plot.

    Args:
        df_trials: Trial candidate dataframe containing feature and prediction columns.
        model: Fitted multi-output model pipeline.
        target_col: Target column to visualize.
        feature_col: Feature column used as the PD x-axis.
        target_item: Classification label to visualize when the target is classification.

    Returns:
        go.Figure: Plotly figure containing PD/ICE and candidate points.
    """
    child_model = model.models[target_col]
    X = _get_training_X(child_model)
    y = _get_training_y(child_model)
    xpd, xticks = get_pd_and_ice(
        df_trials,
        child_model,
        target=feature_col,
        unique_dict= {},
        bounds=[
            X[feature_col].min(),
            X[feature_col].max()
        ]
    )
    
    if child_model.task=="classification":
        col_idx = child_model.item2idx[target_item]
        xpd = xpd[:,:,[col_idx]]    
        y = (y==col_idx).astype(int)
    
    fig = show_pd_and_ice(
        X_PD=xpd,
        target_col=feature_col,
        X=X,
        y=y,
        xticks=xticks,
        ice=True,
    )
    
    
    if child_model.task=="classification":
        y_trials = df_trials["pred_"+target_col+"_"+target_item]
    else:
        y_trials = df_trials["pred_"+target_col]
    
    fig.add_trace(
        go.Scatter(
            x=df_trials[feature_col],  # 候補点の予測値の平均
            y=y_trials,  # 同じ値をプロットする（等価線のため）
            mode="markers",
            showlegend=True,
            name='候補点',
            marker=dict(
                color='green',  # 候補点の色を指定
                size=10         # 候補点のサイズを指定
            ),
        )
    )
    return fig

def show_ia_result_with_pd_2d(
    df_trials,
    model,
    target_col,
    feature_cols,
    target_item
):
    """Show inverse-analysis candidates over a two-dimensional PD plot.

    Args:
        df_trials: Trial candidate dataframe containing feature and prediction columns.
        model: Fitted multi-output model pipeline.
        target_col: Target column to visualize.
        feature_cols: Two feature columns used as PD axes.
        target_item: Classification label to visualize when the target is classification.

    Returns:
        go.Figure: Plotly figure containing 2D PD and candidate points.
    """
    child_model = model.models[target_col]
    X = _get_training_X(child_model)
    y = _get_training_y(child_model)
    X_PD, xticks1, xticks2 = get_pd_and_ice_2d(
        df_trials,
        child_model,
        targets=feature_cols,
        unique_dict= {},
        bounds=[
            [X[feature_cols[0]].min(), X[feature_cols[0]].max()],
            [X[feature_cols[1]].min(), X[feature_cols[1]].max()]
        ]
    )
    
    if child_model.task=="classification":
        col_idx = child_model.item2idx[target_item]
        X_PD = X_PD[:,:,[col_idx]]    
        y = (y==col_idx).astype(int)
    
    
    fig = show_pd_2d(
        X_PD,
        target_cols=feature_cols,
        xticks1=xticks1,
        xticks2=xticks2,
        X=X,
        y=y,
        col_idx=-1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df_trials[feature_cols[0]],  # 候補点の予測値の平均
            y=df_trials[feature_cols[1]],  # 同じ値をプロットする（等価線のため）
            mode="markers",
            showlegend=True,
            name='候補点',
            marker=dict(
                color='yellow',  # 候補点の色を指定
                size=10         # 候補点のサイズを指定
            ),
        )
    )
    return fig

def show_ia_importance(
    model,
    study,
    target
):
    """Show Optuna parameter importances for an inverse-analysis target.

    Args:
        model: Fitted multi-output model pipeline.
        study: Optuna study containing trial objective values.
        target: Target column whose objective index is visualized.

    Returns:
        go.Figure: Feature-importance bar chart.
    """
    target_cols = model.target_cols
    imp = optuna.visualization._param_importances._get_importances_infos(
        study,
        evaluator=None,
        params=None,
        target=lambda t: t.values[target_cols.index(target)],
        target_name=None
    )
    
    feature_importances = np.array(imp[0].importance_values)
    feature_names = np.array(imp[0].param_names)
    
    return show_importances(
        feature_names,
        feature_importances,
        n_bar = 15
    )

def show_ia_result_pareto(
    model,
    df_trials,
    target1,
    target2,
    target_item1=None,
    target_item2=None
):
    """Show actual and candidate values on a two-target Pareto scatter plot.

    Args:
        model: Fitted multi-output model pipeline.
        df_trials: Trial candidate dataframe containing prediction columns.
        target1: Target column used for the x-axis.
        target2: Target column used for the y-axis.
        target_item1: Classification label for ``target1``. Defaults to None.
        target_item2: Classification label for ``target2``. Defaults to None.

    Returns:
        go.Figure: Plotly Pareto scatter plot.
    """
    child_model1 = model.models[target1]
    child_model2 = model.models[target2]
    y1 = _get_training_y(child_model1)
    y2 = _get_training_y(child_model2)

    if child_model1.task=="classification":
        col_idx1 = child_model1.item2idx[target_item1]
        y1 = (y1==col_idx1).astype(int)

    if child_model2.task=="classification":
        col_idx2 = child_model2.item2idx[target_item2]
        y2 = (y2==col_idx2).astype(int)
    
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=np.asarray(y1).ravel(),  # 候補点の予測値の平均
            y=np.asarray(y2).ravel(),  # 同じ値をプロットする（等価線のため）
            mode="markers",
            showlegend=True,
            name='候補点',
            marker=dict(
                color='blue',  # 候補点の色を指定
                size=10         # 候補点のサイズを指定
            ),
        )
    )
   
    if child_model1.task=="classification":
        if target_item1 is not None:
            x = df_trials["pred_"+target1+"_"+target_item1].values.ravel()
        else:
            x = df_trials["pred_"+target1+"_"+child_model1.target_items[-1]].values.ravel()
    else:
        x = df_trials["pred_"+target1].values.ravel()

    if child_model2.task=="classification":
        if target_item2 is not None:
            y = df_trials["pred_"+target2+"_"+target_item2].values.ravel()
        else:
            y = df_trials["pred_"+target2+"_"+child_model2.target_items[-1]].values.ravel()
    else:
        y = df_trials["pred_"+target2].values.ravel()

    fig.add_trace(
        go.Scatter(
            x=x,  # 候補点の予測値の平均
            y=y,  # 同じ値をプロットする（等価線のため）
            mode="markers",
            showlegend=True,
            name='候補点',
            marker=dict(
                color='red',  # 候補点の色を指定
                size=10         # 候補点のサイズを指定
            ),
        )
    )
    
    fig.update_layout(
        title='Parato Plot',
        xaxis_title=target1,
        yaxis_title=target2,
        showlegend=False,
        width=650,
        height=600
    )
    return fig