import numpy as np
import plotly.graph_objects as go
import optuna
from typing import List, Optional, Union, Dict, Tuple, Callable, Any

from machine_learning.models.machine_learning.training import fit_model, tune_model, cv_fit
from machine_learning.models.machine_learning.utils import get_cat_unique_values, feature_names_from_pipeline, label_encode
from machine_learning.models.machine_learning.explainability import get_shap_values, get_importances, get_pfi_values, get_shap_scatter, get_pd_and_ice, get_pd_and_ice_2d
from machine_learning.models.machine_learning.pipelines import make_pipeline

from machine_learning.models.visualization import show_importances, yy_plot_ml, show_pd_and_ice, show_pd_2d, show_shap_scatter, show_shap_beeswarm

def show_ia_result_with_pd(
    df_trials,
    model,
    target_col,
    feature_col,
    target_item
):
    xpd, xticks = get_pd_and_ice(
        df_trials,
        model.models[target_col],
        target=feature_col,
        unique_dict= {},
        bounds=[
            model.models[target_col].X[feature_col].min(),
            model.models[target_col].X[feature_col].max()
        ]
    )
    
    X=model.models[target_col].X
    y=model.models[target_col].y
    
    if model.models[target_col].task=="classification":
        col_idx = model.models[target_col].item2idx[target_item]
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
    
    
    if model.models[target_col].task=="classification":
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
    X_PD, xticks1, xticks2 = get_pd_and_ice_2d(
        df_trials,
        model.models[target_col],
        targets=feature_cols,
        unique_dict= {},
        bounds=[
            [model.models[target_col].X[feature_cols[0]].min(), model.models[target_col].X[feature_cols[0]].max()],
            [model.models[target_col].X[feature_cols[1]].min(), model.models[target_col].X[feature_cols[1]].max()]
        ]
    )
    
    X=model.models[target_col].X
    y=model.models[target_col].y
    
    if model.models[target_col].task=="classification":
        col_idx = model.models[target_col].item2idx[target_item]
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
    y1 = model.models[target1].y
    y2 = model.models[target2].y

    if model.models[target1].task=="classification":
        col_idx1 = model.models[target1].item2idx[target_item1]
        y1 = (y1==col_idx1).astype(int)

    if model.models[target2].task=="classification":
        col_idx2 = model.models[target2].item2idx[target_item2]
        y2 = (y2==col_idx2).astype(int)
    
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=y1.values.ravel(),  # 候補点の予測値の平均
            y=y2.values.ravel(),  # 同じ値をプロットする（等価線のため）
            mode="markers",
            showlegend=True,
            name='候補点',
            marker=dict(
                color='blue',  # 候補点の色を指定
                size=10         # 候補点のサイズを指定
            ),
        )
    )
   
    if model.models[target1].task=="classification":
        if target_item1 is not None:
            x = df_trials["pred_"+target1+"_"+target_item1].values.ravel()
        else:
            x = df_trials["pred_"+target1+"_"+model.models[target1].target_items[-1]].values.ravel()
    else:
        x = df_trials["pred_"+target1].values.ravel()

    if model.models[target2].task=="classification":
        if target_item2 is not None:
            y = df_trials["pred_"+target2+"_"+target_item2].values.ravel()
        else:
            y = df_trials["pred_"+target2+"_"+model.models[target2].target_items[-1]].values.ravel()
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