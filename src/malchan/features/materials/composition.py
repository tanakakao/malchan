"""組成式を材料特徴量バックエンドの入力へ変換する。"""

from __future__ import annotations

from collections.abc import Hashable
from typing import Any

import numpy as np
import pandas as pd
from pymatgen.core import Composition as PymatgenComposition
from sklearn.base import BaseEstimator, TransformerMixin


def _as_series(X: Any) -> pd.Series:
    """1列のDataFrameまたは配列をSeriesへ正規化する。"""
    if isinstance(X, pd.DataFrame):
        return X.iloc[:, 0]
    return pd.Series(np.ravel(X))


def composition_cache_key(comp: Any, *, ndigits: int = 12) -> Hashable:
    """組成dictまたはCompositionから安定したキャッシュキーを作る。"""
    if comp is None:
        return ("__NONE__",)

    if hasattr(comp, "as_dict"):
        try:
            comp = comp.as_dict()
        except Exception:
            return ("__BAD__", str(comp))

    if isinstance(comp, dict):
        if not comp:
            return ("__EMPTY__",)
        items: list[tuple[str, Any]] = []
        for element, amount in comp.items():
            try:
                value: Any = round(float(amount), ndigits)
            except (TypeError, ValueError):
                value = amount
            items.append((str(element), value))
        return tuple(sorted(items))

    return ("__BAD__", str(comp))


class FormulaToFractionDict(BaseEstimator, TransformerMixin):
    """組成式を元素分率dictへ変換する。"""

    def __init__(self, invalid: str = "empty"):
        self.invalid = invalid

    def fit(self, X: Any, y: Any = None) -> "FormulaToFractionDict":
        return self

    def _formula_to_dict(self, formula: Any) -> dict[str, float]:
        if formula is None or (isinstance(formula, float) and np.isnan(formula)):
            return {}
        try:
            composition = PymatgenComposition(str(formula))
            amounts = composition.get_el_amt_dict()
            total = sum(amounts.values())
            return {str(element): amount / total for element, amount in amounts.items()}
        except Exception:
            if self.invalid == "error":
                raise
            return {}

    def transform(self, X: Any) -> pd.DataFrame:
        series = _as_series(X)
        return series.apply(self._formula_to_dict).to_frame(name="comp_dict")

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return np.array(["comp_dict"], dtype=object)


class FormulaToComposition(BaseEstimator, TransformerMixin):
    """組成式をpymatgenのCompositionへ変換する。"""

    def __init__(self, invalid: str = "empty"):
        self.invalid = invalid

    def fit(self, X: Any, y: Any = None) -> "FormulaToComposition":
        return self

    def _parse(self, formula: Any) -> PymatgenComposition | None:
        if formula is None or (isinstance(formula, float) and np.isnan(formula)):
            return None
        try:
            return PymatgenComposition(str(formula))
        except Exception:
            if self.invalid == "error":
                raise
            return None

    def transform(self, X: Any) -> pd.DataFrame:
        series = _as_series(X).apply(self._parse)
        return pd.DataFrame({"composition": series}, index=series.index)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return np.array(["composition"], dtype=object)


__all__ = [
    "FormulaToComposition",
    "FormulaToFractionDict",
    "composition_cache_key",
]
