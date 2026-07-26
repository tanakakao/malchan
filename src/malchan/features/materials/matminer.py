"""Matminerによる組成特徴量生成。"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


def _featurizer_factories() -> dict[str, Callable[[], Any]]:
    """利用可能なMatminer featurizerの生成関数を返す。"""
    try:
        from matminer.featurizers.composition import (
            BandCenter,
            ElementFraction,
            ElementProperty,
            IonProperty,
            Meredig,
            Miedema,
            Stoichiometry,
            TMetalFraction,
            ValenceOrbital,
            YangSolidSolution,
        )
    except ImportError as exc:
        raise ImportError(
            "Matminer特徴量を使用するにはmatminerを導入してください。"
        ) from exc

    return {
        "ElementProperty": lambda: ElementProperty.from_preset("magpie", impute_nan=True),
        "ValenceOrbital": lambda: ValenceOrbital(impute_nan=True),
        "IonProperty": lambda: IonProperty(impute_nan=True),
        "YangSolidSolution": YangSolidSolution,
        "TMetalFraction": TMetalFraction,
        "Stoichiometry": Stoichiometry,
        "Meredig": Meredig,
        "BandCenter": BandCenter,
        "Miedema": Miedema,
        "ElementFraction": ElementFraction,
    }


def resolve_matminer_featurizers(names: list[str]) -> list[Any]:
    """APIで指定された名前をMatminer featurizerへ変換する。"""
    factories = _featurizer_factories()
    unknown = [name for name in names if name not in factories]
    if unknown:
        raise ValueError(
            f"未対応のMatminer特徴量です: {unknown}. "
            f"利用可能: {sorted(factories)}"
        )
    return [factories[name]() for name in names]


class MatminerCompositionFeaturizer(BaseEstimator, TransformerMixin):
    """複数のMatminer featurizerをまとめて適用する。"""

    def __init__(
        self,
        featurizers: list[Any] = [],
        input_col: str = "composition",
        prefix: str = "mm__",
        n_jobs: int = 1,
        use_cache: bool = True,
        cache_max: int = 20000,
    ):
        self.featurizers = featurizers
        self.input_col = input_col
        self.prefix = prefix
        self.n_jobs = n_jobs
        self.use_cache = use_cache
        self.cache_max = cache_max

    @staticmethod
    def _key(composition: Any) -> str | None:
        try:
            return str(composition.reduced_formula)
        except Exception:
            return None

    def fit(self, X: pd.DataFrame, y: Any = None) -> "MatminerCompositionFeaturizer":
        try:
            from matminer.featurizers.base import MultipleFeaturizer
        except ImportError as exc:
            raise ImportError(
                "Matminer特徴量を使用するにはmatminerを導入してください。"
            ) from exc

        self.mm_ = MultipleFeaturizer(self.featurizers)
        try:
            self.mm_.set_n_jobs(self.n_jobs)
        except Exception:
            pass

        self.cols_in_ = list(self.mm_.feature_labels())
        self.cols_out_ = [f"{self.prefix}{column}" for column in self.cols_in_]
        self._cache_ = {} if self.use_cache else None
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        compositions = X.iloc[:, 0].tolist()
        feature_count = len(self.cols_in_)
        nan_row = [np.nan] * feature_count

        if not self.use_cache:
            rows = self.mm_.featurize_many(compositions, ignore_errors=True, pbar=False)
            rows = [nan_row if row is None else row for row in rows]
            return pd.DataFrame(rows, index=X.index, columns=self.cols_out_)

        rows: list[list[float] | None] = [None] * len(compositions)
        uncached: list[tuple[int, str | None, Any]] = []
        for index, composition in enumerate(compositions):
            if composition is None:
                rows[index] = nan_row
                continue
            key = self._key(composition)
            if key is not None and key in self._cache_:
                rows[index] = self._cache_[key]
            else:
                uncached.append((index, key, composition))

        if uncached:
            unique_keys: list[str] = []
            unique_compositions: list[Any] = []
            seen: set[str] = set()
            for _, key, composition in uncached:
                if key is None or key in self._cache_ or key in seen:
                    continue
                seen.add(key)
                unique_keys.append(key)
                unique_compositions.append(composition)

            if unique_compositions:
                values = self.mm_.featurize_many(
                    unique_compositions,
                    ignore_errors=True,
                    pbar=False,
                )
                for key, value in zip(unique_keys, values):
                    self._cache_[key] = nan_row if value is None else value
                if self.cache_max and len(self._cache_) > self.cache_max:
                    self._cache_.clear()

            for index, key, composition in uncached:
                if key is None:
                    value = self.mm_.featurize(composition, ignore_errors=True)
                    rows[index] = nan_row if value is None else value
                else:
                    rows[index] = self._cache_.get(key, nan_row)

        return pd.DataFrame(rows, index=X.index, columns=self.cols_out_)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return np.array(self.cols_out_, dtype=object)


__all__ = [
    "MatminerCompositionFeaturizer",
    "resolve_matminer_featurizers",
]
