"""FastAPIのXAIレスポンスをPlotlyで可視化する補助関数。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .machine_learning_plots import show_importances


def _as_mapping(payload: Any, payload_name: str) -> Mapping[str, Any]:
    """APIレスポンスまたはPydanticモデルを辞書として取得する。

    Args:
        payload: ``response.json()``の結果、HTTPレスポンス、またはPydanticモデル。
        payload_name: エラーメッセージに表示する入力名。

    Returns:
        XAIレスポンスを表すマッピング。

    Raises:
        TypeError: マッピングへ変換できない場合。
    """

    if isinstance(payload, Mapping):
        return payload

    model_dump = getattr(payload, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump(mode="json")
        if isinstance(dumped, Mapping):
            return dumped

    json_method = getattr(payload, "json", None)
    if callable(json_method):
        dumped = json_method()
        if isinstance(dumped, Mapping):
            return dumped

    raise TypeError(
        f"{payload_name} must be a mapping, an HTTP response with json(), "
        "or a Pydantic model with model_dump()."
    )


def show_xai_importance(
    response: Any,
    n_bar: int = 15,
) -> go.Figure:
    """FastAPIの重要度レスポンスを棒グラフで表示する。

    Args:
        response: ``/xai/{target}/importance``のレスポンスまたは
            ``response.json()``の結果。
        n_bar: 表示する上位特徴量数。

    Returns:
        特徴量重要度のPlotly図オブジェクト。

    Raises:
        ValueError: レスポンス形式または``n_bar``が不正な場合。
    """

    if n_bar < 1:
        raise ValueError("n_bar must be at least 1.")

    payload = _as_mapping(response, "response")
    items = payload.get("items")
    if not isinstance(items, list):
        raise ValueError("XAI importance response must contain an 'items' list.")

    if not items:
        return go.Figure()

    feature_names: list[str] = []
    feature_importances: list[float] = []
    for item in items:
        if not isinstance(item, Mapping) or "feature" not in item or "value" not in item:
            raise ValueError("Each importance item must contain 'feature' and 'value'.")
        feature_names.append(str(item["feature"]))
        feature_importances.append(float(item["value"]))

    fig = show_importances(
        feature_names=feature_names,
        feature_importances=np.asarray(feature_importances, dtype=float),
        n_bar=n_bar,
    )
    target = payload.get("target", "target")
    method = payload.get("method", "importance")
    fig.update_layout(title=f"{method} importance: {target}")
    return fig


def _resolve_shap_columns(
    frame: pd.DataFrame,
    payload: Mapping[str, Any],
    target_item: Any | None,
) -> list[str]:
    """SHAPレスポンスから描画対象列を解決する。"""

    value_columns = payload.get("value_columns")
    if value_columns is None:
        value_columns = [
            str(column)
            for column in frame.columns
            if str(column) == "shap" or str(column).startswith("shap_")
        ]
    if not isinstance(value_columns, list) or not value_columns:
        raise ValueError("XAI SHAP response must contain at least one SHAP value column.")

    value_columns = [str(column) for column in value_columns]
    missing = [column for column in value_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"SHAP value columns are missing from records: {missing}")

    if target_item is None:
        return value_columns

    requested = str(target_item)
    candidates = [requested, f"shap_{requested}"]
    for candidate in candidates:
        if candidate in value_columns:
            return [candidate]
    raise ValueError(
        f"SHAP column for target_item={target_item!r} is unavailable. "
        f"Available: {value_columns}"
    )


def show_xai_shap_scatter(
    response: Any,
    target_item: Any | None = None,
) -> go.Figure:
    """FastAPIのSHAPレスポンスを散布図で表示する。

    Args:
        response: ``/xai/{target}/shap``のレスポンスまたは
            ``response.json()``の結果。
        target_item: 分類で表示するクラス名またはSHAP列名。省略時は
            レスポンスに含まれる全SHAP列を表示する。

    Returns:
        特徴量値とSHAP値のPlotly散布図。

    Raises:
        ValueError: 必須フィールドまたはSHAP列が不足している場合。
    """

    payload = _as_mapping(response, "response")
    feature = payload.get("feature")
    records = payload.get("records")
    if not isinstance(feature, str) or not feature:
        raise ValueError("XAI SHAP response must contain a non-empty 'feature'.")
    if not isinstance(records, list):
        raise ValueError("XAI SHAP response must contain a 'records' list.")

    frame = pd.DataFrame(records)
    if feature not in frame.columns:
        raise ValueError(f"Feature {feature!r} is missing from SHAP records.")
    shap_columns = _resolve_shap_columns(frame, payload, target_item)

    fig = go.Figure()
    for shap_column in shap_columns:
        fig.add_trace(
            go.Scatter(
                x=frame[feature],
                y=frame[shap_column],
                mode="markers",
                name=shap_column,
                showlegend=len(shap_columns) > 1,
            )
        )

    fig.add_hline(y=0.0, line_width=1)
    target = payload.get("target", "target")
    fig.update_layout(
        title=f"SHAP: {target} / {feature}",
        xaxis_title=feature,
        yaxis_title="SHAP value",
        width=650,
        height=600,
    )
    return fig


def show_xai_pd_and_ice(
    response: Any,
    *,
    series_name: str | None = None,
    ice: bool = True,
    max_ice: int | None = None,
) -> go.Figure:
    """FastAPIのPDPレスポンスをPDP/ICE図として表示する。

    Args:
        response: ``/xai/{target}/pdp``のレスポンスまたは
            ``response.json()``の結果。
        series_name: 分類などの複数系列から表示する系列名。省略時は全系列。
        ice: レスポンスに含まれるICE曲線を表示するかどうか。
        max_ice: 描画するICE曲線数の上限。``None``では全曲線を表示する。

    Returns:
        PDPと任意のICE曲線を含むPlotly図オブジェクト。

    Raises:
        ValueError: レスポンス形式、系列名、または配列長が不正な場合。
    """

    if max_ice is not None and max_ice < 1:
        raise ValueError("max_ice must be at least 1 when specified.")

    payload = _as_mapping(response, "response")
    feature = payload.get("feature")
    x_values = payload.get("x_values")
    series = payload.get("series")
    if not isinstance(feature, str) or not feature:
        raise ValueError("XAI PDP response must contain a non-empty 'feature'.")
    if not isinstance(x_values, list):
        raise ValueError("XAI PDP response must contain an 'x_values' list.")
    if not isinstance(series, list) or not series:
        raise ValueError("XAI PDP response must contain a non-empty 'series' list.")

    selected_series = series
    if series_name is not None:
        selected_series = [
            item
            for item in series
            if isinstance(item, Mapping) and str(item.get("name")) == series_name
        ]
        if not selected_series:
            available = [
                str(item.get("name"))
                for item in series
                if isinstance(item, Mapping)
            ]
            raise ValueError(
                f"Unknown PDP series {series_name!r}. Available: {available}"
            )

    fig = go.Figure()
    for item in selected_series:
        if not isinstance(item, Mapping):
            raise ValueError("Each PDP series must be a mapping.")
        name = str(item.get("name", "output"))
        pd_values = item.get("pd_values")
        if not isinstance(pd_values, list) or len(pd_values) != len(x_values):
            raise ValueError(
                f"PDP series {name!r} must have the same length as x_values."
            )

        if ice:
            ice_values = item.get("ice_values") or []
            if not isinstance(ice_values, list):
                raise ValueError(f"ICE values for series {name!r} must be a list.")
            if max_ice is not None:
                ice_values = ice_values[:max_ice]
            for values in ice_values:
                if not isinstance(values, list) or len(values) != len(x_values):
                    raise ValueError(
                        f"Each ICE curve for series {name!r} must match x_values."
                    )
                fig.add_trace(
                    go.Scatter(
                        x=x_values,
                        y=values,
                        mode="lines",
                        line={"color": "rgba(120,120,120,0.2)", "width": 1},
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )

        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=pd_values,
                mode="lines",
                line={"width": 3},
                name=name,
            )
        )

    target = payload.get("target", "target")
    fig.update_layout(
        title=f"PDP / ICE: {target} / {feature}",
        xaxis_title=feature,
        yaxis_title="Partial dependence",
        width=650,
        height=600,
        showlegend=True,
    )
    return fig


__all__ = [
    "show_xai_importance",
    "show_xai_pd_and_ice",
    "show_xai_shap_scatter",
]
