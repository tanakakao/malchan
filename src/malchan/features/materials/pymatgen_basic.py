"""pymatgenの元素情報による軽量な組成基本統計。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


SUPPORTED_PYMATGEN_PROPERTIES = (
    "atomic_number",
    "atomic_mass",
    "atomic_radius",
    "electronegativity",
    "row",
    "group",
)

COMPOSITION_SUMMARY_FEATURES = (
    "n_elements",
    "reduced_num_atoms",
    "reduced_formula_weight",
    "composition_entropy",
    "min_fraction",
    "max_fraction",
)


class PymatgenBasicCompositionFeaturizer(BaseEstimator, TransformerMixin):
    """元素物性を組成分率で集約し、組成の基本統計を生成する。

    選択した元素物性について、組成分率で重み付けした平均、標準偏差、
    最小、最大、範囲を生成する。加えて元素数、還元組成の原子数と式量、
    組成エントロピー、元素分率の最小・最大を常に出力する。

    Args:
        props: 使用するpymatgen元素物性名。
        stats: 各元素物性へ適用する統計量。
        prefix: 出力特徴量名の接頭辞。
        use_cache: 同一組成の計算結果を再利用するか。
        cache_max: キャッシュ件数の上限。超過時はキャッシュを初期化する。
    """

    def __init__(
        self,
        props: list[str] | tuple[str, ...] = (),
        stats: tuple[str, ...] = ("mean", "std", "min", "max", "range"),
        prefix: str = "pmg__",
        use_cache: bool = True,
        cache_max: int = 20000,
    ):
        self.props = props
        self.stats = stats
        self.prefix = prefix
        self.use_cache = use_cache
        self.cache_max = cache_max

    @staticmethod
    def _weighted_stats(values: np.ndarray, weights: np.ndarray) -> dict[str, float]:
        """有限値だけを用いて重み付き統計量を計算する。"""

        mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
        if not np.any(mask):
            return {
                "mean": np.nan,
                "std": np.nan,
                "min": np.nan,
                "max": np.nan,
                "range": np.nan,
            }

        values = values[mask]
        weights = weights[mask]
        weights = weights / weights.sum()
        mean = float(np.sum(weights * values))
        variance = float(np.sum(weights * (values - mean) ** 2))
        minimum = float(np.min(values))
        maximum = float(np.max(values))
        return {
            "mean": mean,
            "std": float(np.sqrt(variance)),
            "min": minimum,
            "max": maximum,
            "range": maximum - minimum,
        }

    @staticmethod
    def _as_float(value: Any) -> float:
        """pymatgenの単位付き値やNoneをfloatへ安全に変換する。"""

        if value is None:
            return np.nan
        try:
            return float(value)
        except (TypeError, ValueError):
            return np.nan

    @classmethod
    def _element_property(cls, element: Any, prop: str) -> float:
        """Elementから指定した基本物性を取得する。"""

        if prop == "atomic_number":
            return cls._as_float(element.Z)
        if prop == "atomic_mass":
            return cls._as_float(element.atomic_mass)
        if prop == "atomic_radius":
            return cls._as_float(element.atomic_radius)
        if prop == "electronegativity":
            return cls._as_float(element.X)
        if prop == "row":
            return cls._as_float(element.row)
        if prop == "group":
            return cls._as_float(element.group)
        raise ValueError(f"Unknown pymatgen property: {prop}")

    def fit(self, X: pd.DataFrame, y: Any = None) -> "PymatgenBasicCompositionFeaturizer":
        """出力列とキャッシュを初期化する。"""

        properties = tuple(self.props) or SUPPORTED_PYMATGEN_PROPERTIES
        unknown = [prop for prop in properties if prop not in SUPPORTED_PYMATGEN_PROPERTIES]
        if unknown:
            raise ValueError(
                f"Unknown pymatgen properties: {unknown}. "
                f"Available: {list(SUPPORTED_PYMATGEN_PROPERTIES)}"
            )

        self.props_ = list(dict.fromkeys(properties))
        self.cols_out_ = [
            *(f"{self.prefix}{name}" for name in COMPOSITION_SUMMARY_FEATURES),
            *(
                f"{self.prefix}{prop}__{stat}"
                for prop in self.props_
                for stat in self.stats
            ),
        ]
        self._cache_ = {} if self.use_cache else None
        return self

    def _composition_features(self, composition: Any) -> list[float]:
        """1組成から基本統計を生成する。"""

        reduced = composition.reduced_composition
        fractional = composition.fractional_composition
        elements = list(fractional.elements)
        weights = np.array([float(fractional[element]) for element in elements], dtype=float)
        positive_weights = weights[np.isfinite(weights) & (weights > 0)]
        entropy = (
            float(-np.sum(positive_weights * np.log(positive_weights)))
            if positive_weights.size
            else np.nan
        )

        features: list[float] = [
            float(len(elements)),
            self._as_float(reduced.num_atoms),
            self._as_float(reduced.weight),
            entropy,
            float(np.min(positive_weights)) if positive_weights.size else np.nan,
            float(np.max(positive_weights)) if positive_weights.size else np.nan,
        ]

        for prop in self.props_:
            values = np.array(
                [self._element_property(element, prop) for element in elements],
                dtype=float,
            )
            statistics = self._weighted_stats(values, weights)
            features.extend(statistics.get(stat, np.nan) for stat in self.stats)
        return features

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """pymatgen Composition列を数値特徴量へ変換する。"""

        rows: list[list[float]] = []
        for composition in X.iloc[:, 0].tolist():
            if composition is None:
                rows.append([np.nan] * len(self.cols_out_))
                continue

            key: str | None = None
            if self.use_cache:
                try:
                    key = str(composition.reduced_formula)
                except Exception:
                    key = None
                if key is not None and key in self._cache_:
                    rows.append(self._cache_[key])
                    continue

            try:
                features = self._composition_features(composition)
            except Exception:
                features = [np.nan] * len(self.cols_out_)
            rows.append(features)

            if self.use_cache and key is not None:
                self._cache_[key] = features
                if self.cache_max and len(self._cache_) > self.cache_max:
                    self._cache_.clear()

        return pd.DataFrame(rows, index=X.index, columns=self.cols_out_)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        """出力特徴量名を返す。"""

        return np.array(self.cols_out_, dtype=object)


__all__ = [
    "COMPOSITION_SUMMARY_FEATURES",
    "PymatgenBasicCompositionFeaturizer",
    "SUPPORTED_PYMATGEN_PROPERTIES",
]
