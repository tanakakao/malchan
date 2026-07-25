"""Tests for plotting cached FastAPI XAI responses."""

import pytest

pytest.importorskip("plotly")

from malchan.visualization import (
    show_xai_importance,
    show_xai_pd_and_ice,
    show_xai_shap_scatter,
)


def test_show_xai_importance_accepts_api_payload() -> None:
    """Create an importance bar chart directly from the API response."""

    fig = show_xai_importance(
        {
            "model_id": "model-1",
            "target": "property",
            "method": "shap",
            "combined": True,
            "items": [
                {"feature": "temperature", "value": 0.8},
                {"feature": "time", "value": 0.3},
            ],
        },
        n_bar=2,
    )

    assert len(fig.data) == 1
    assert list(fig.data[0].y) == ["temperature", "time"]
    assert list(fig.data[0].x) == [0.8, 0.3]
    assert fig.layout.title.text == "shap importance: property"


def test_show_xai_shap_scatter_supports_multiple_classes() -> None:
    """Plot all class-specific SHAP columns or select one class."""

    payload = {
        "model_id": "model-1",
        "target": "quality",
        "feature": "temperature",
        "value_columns": ["shap_OK", "shap_NG"],
        "records": [
            {"temperature": 100.0, "shap_OK": 0.2, "shap_NG": -0.2},
            {"temperature": 120.0, "shap_OK": 0.5, "shap_NG": -0.5},
        ],
    }

    all_classes = show_xai_shap_scatter(payload)
    selected_class = show_xai_shap_scatter(payload, target_item="OK")

    assert len(all_classes.data) == 2
    assert [trace.name for trace in all_classes.data] == ["shap_OK", "shap_NG"]
    assert len(selected_class.data) == 1
    assert selected_class.data[0].name == "shap_OK"
    assert list(selected_class.data[0].x) == [100.0, 120.0]
    assert list(selected_class.data[0].y) == [0.2, 0.5]


def test_show_xai_pd_and_ice_accepts_api_payload() -> None:
    """Create PDP and ICE traces from the serialized endpoint response."""

    fig = show_xai_pd_and_ice(
        {
            "model_id": "model-1",
            "target": "property",
            "feature": "temperature",
            "x_values": [80.0, 100.0, 120.0],
            "series": [
                {
                    "name": "property",
                    "pd_values": [1.0, 1.5, 2.0],
                    "ice_values": [
                        [0.9, 1.4, 1.9],
                        [1.1, 1.6, 2.1],
                    ],
                }
            ],
        }
    )

    assert len(fig.data) == 3
    assert fig.data[-1].name == "property"
    assert list(fig.data[-1].x) == [80.0, 100.0, 120.0]
    assert list(fig.data[-1].y) == [1.0, 1.5, 2.0]


def test_show_xai_pd_and_ice_selects_one_series() -> None:
    """Select one output series when a classification response has several."""

    fig = show_xai_pd_and_ice(
        {
            "target": "quality",
            "feature": "temperature",
            "x_values": [80.0, 100.0],
            "series": [
                {"name": "OK", "pd_values": [0.3, 0.7], "ice_values": None},
                {"name": "NG", "pd_values": [0.7, 0.3], "ice_values": None},
            ],
        },
        series_name="OK",
        ice=False,
    )

    assert len(fig.data) == 1
    assert fig.data[0].name == "OK"


def test_xai_plot_helpers_reject_invalid_payloads() -> None:
    """Raise useful errors for malformed response dictionaries."""

    with pytest.raises(ValueError, match="items"):
        show_xai_importance({})

    with pytest.raises(ValueError, match="feature"):
        show_xai_shap_scatter({"records": []})

    with pytest.raises(ValueError, match="series"):
        show_xai_pd_and_ice({"feature": "x1", "x_values": [], "series": []})
