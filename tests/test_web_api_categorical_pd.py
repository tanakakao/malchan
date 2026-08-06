"""Regression tests for categorical partial-dependence web figures."""

from __future__ import annotations

import pandas as pd

from malchan.visualization.web_api_plots import show_model_pd_and_ice


class _CategoricalChildModel:
    task = "regression"
    target_col = "y"
    target_items = None

    def __init__(self) -> None:
        self._X = pd.DataFrame(
            {
                "material": ["A", "B", "A"],
                "material_grade": [1.0, 2.0, 3.0],
            }
        )
        self._y = pd.DataFrame({"y": [11.0, 22.0, 13.0]})
        self.unique_cols = {"material": ["A", "B"]}

    def _get_X(self) -> pd.DataFrame:
        return self._X

    def _get_y(self) -> pd.DataFrame:
        return self._y

    def _shared_attr(self, name: str):
        return getattr(self, name)

    def predict(self, X: pd.DataFrame, proba: bool = False) -> pd.DataFrame:
        del proba
        category_effect = X["material"].map({"A": 10.0, "B": 20.0})
        numeric_effect = pd.to_numeric(X["material_grade"], errors="raise")
        return pd.DataFrame({"y": category_effect + numeric_effect})


class _MultiOutputModel:
    def __init__(self) -> None:
        self.models = {"y": _CategoricalChildModel()}


def test_categorical_pd_changes_only_the_exact_feature_column() -> None:
    """A categorical feature must not overwrite similarly named numeric columns."""

    figure = show_model_pd_and_ice(
        _MultiOutputModel(),
        "y",
        "material",
        ice=False,
    )

    pd_trace = next(
        trace
        for trace in figure.data
        if str(getattr(trace, "name", "")).startswith("Partial Dependence")
    )
    assert list(pd_trace.x) == ["A", "B"]
    assert list(pd_trace.y) == [12.0, 22.0]
    assert pd_trace.mode == "lines+markers"
    assert figure.layout.xaxis.type == "category"
    assert list(figure.layout.xaxis.categoryarray) == ["A", "B"]
