"""Tests for compact compositional labels in SHAP beeswarm figures."""

from malchan.visualization.xai_beeswarm import (
    _compact_compositional_feature_name,
    show_xai_shap_beeswarm,
)


def test_compact_compositional_feature_names() -> None:
    """Generated log-ratio names should become concise axis labels."""

    assert (
        _compact_compositional_feature_name("compositional_0__ilr__balance_2")
        == "組成1 · ILR2"
    )
    assert (
        _compact_compositional_feature_name("compositional_1__clr__SiO2")
        == "組成2 · CLR:SiO2"
    )
    assert (
        _compact_compositional_feature_name("compositional_0__alr__Al2O3_over_SiO2")
        == "組成1 · ALR:Al2O3/SiO2"
    )
    assert _compact_compositional_feature_name("temperature") == "temperature"


def test_long_component_names_are_bounded() -> None:
    """Long raw component names must not expand the beeswarm plot margin."""

    label = _compact_compositional_feature_name(
        "compositional_0__clr__extremely_long_composition_component_name"
    )

    assert label.startswith("組成1 · CLR:")
    assert label.endswith("…")
    assert len(label) < 30


def test_beeswarm_uses_short_axis_labels_and_full_hover_names() -> None:
    """Axis labels are compact while hover data retains the exact transformed name."""

    features = [
        "temperature",
        "compositional_0__ilr__balance_1",
        "compositional_0__ilr__balance_2",
    ]
    response = {
        "target": "property",
        "features": features,
        "records": [
            {
                "temperature": 100.0,
                "compositional_0__ilr__balance_1": -0.2,
                "compositional_0__ilr__balance_2": 0.4,
            },
            {
                "temperature": 200.0,
                "compositional_0__ilr__balance_1": 0.3,
                "compositional_0__ilr__balance_2": -0.1,
            },
        ],
        "cat_cols": [],
        "output_names": ["shap"],
        "shap_values": {
            "shap": [
                [0.05, 0.4, 0.2],
                [-0.02, -0.3, 0.1],
            ]
        },
    }

    figure = show_xai_shap_beeswarm(response, n_shap_top=3)

    ticktext = [str(value) for value in figure.layout.yaxis.ticktext]
    assert "組成1 · ILR1" in ticktext
    assert "組成1 · ILR2" in ticktext
    assert "temperature" in ticktext
    assert all("compositional_" not in value for value in ticktext)

    hover_names = [str(value) for value in figure.data[0].customdata]
    assert "compositional_0__ilr__balance_1" in hover_names
    assert "compositional_0__ilr__balance_2" in hover_names
    assert "%{customdata}" in figure.data[0].hovertemplate
