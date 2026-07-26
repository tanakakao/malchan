"""Tests for exporting and visualizing all cached SHAP values."""

from types import SimpleNamespace

import pandas as pd
import pytest

from malchan.app.services.xai_shap_service import get_xai_shap_values
from malchan.visualization import show_xai_shap_beeswarm


class _Service:
    """Minimal model service exposing one registered model."""

    def __init__(self, child: object) -> None:
        self.registered = SimpleNamespace(
            model=child,
            info=SimpleNamespace(target_cols=["quality"]),
        )

    def _get_registered(self, model_id: str) -> object:
        assert model_id == "model-1"
        return self.registered


def _child_with_multiclass_shap() -> object:
    return SimpleNamespace(
        all_cols=["temperature", "material"],
        cat_cols=["material"],
        importances={
            "shap_pd": {
                "temperature": pd.DataFrame(
                    {
                        "temperature": [100.0, 120.0, 140.0],
                        "shap_OK": [0.2, 0.4, 0.6],
                        "shap_NG": [-0.2, -0.4, -0.6],
                    }
                ),
                "material": pd.DataFrame(
                    {
                        "material": ["A", "B", "A"],
                        "shap_OK": [0.1, -0.1, 0.2],
                        "shap_NG": [-0.1, 0.1, -0.2],
                    }
                ),
            }
        },
    )


def test_get_xai_shap_values_reconstructs_feature_matrices() -> None:
    """Reconstruct aligned raw values and one SHAP matrix per class."""

    response = get_xai_shap_values(
        _Service(_child_with_multiclass_shap()),
        "model-1",
        "quality",
    )

    assert response.features == ["temperature", "material"]
    assert response.cat_cols == ["material"]
    assert response.output_names == ["shap_OK", "shap_NG"]
    assert response.records == [
        {"temperature": 100.0, "material": "A"},
        {"temperature": 120.0, "material": "B"},
        {"temperature": 140.0, "material": "A"},
    ]
    assert response.shap_values["shap_OK"] == [
        [0.2, 0.1],
        [0.4, -0.1],
        [0.6, 0.2],
    ]


def test_get_xai_shap_values_rejects_misaligned_frames() -> None:
    """Reject feature caches whose sample rows cannot be aligned."""

    child = _child_with_multiclass_shap()
    child.importances["shap_pd"]["material"] = child.importances["shap_pd"][
        "material"
    ].iloc[:2]

    with pytest.raises(ValueError, match="same number of rows"):
        get_xai_shap_values(_Service(child), "model-1", "quality")


def test_show_xai_shap_beeswarm_selects_class_matrix() -> None:
    """Build a Plotly beeswarm from a class-specific FastAPI SHAP matrix."""

    response = get_xai_shap_values(
        _Service(_child_with_multiclass_shap()),
        "model-1",
        "quality",
    )

    figure = show_xai_shap_beeswarm(
        response,
        n_shap_top=2,
        target_item="OK",
    )

    assert len(figure.data) == 1
    assert len(figure.data[0].x) == 6
    assert figure.layout.title.text == "SHAP beeswarm: quality / shap_OK"


def test_show_xai_shap_beeswarm_validates_matrix_shape() -> None:
    """Reject a response whose SHAP matrix does not align with raw records."""

    payload = {
        "target": "quality",
        "features": ["x1", "x2"],
        "cat_cols": [],
        "output_names": ["shap"],
        "records": [{"x1": 1.0, "x2": 2.0}],
        "shap_values": {"shap": [[0.1]]},
    }

    with pytest.raises(ValueError, match="expected"):
        show_xai_shap_beeswarm(payload)
