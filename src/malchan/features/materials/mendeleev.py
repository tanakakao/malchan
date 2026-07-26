"""Mendeleev元素プロパティによる組成特徴量生成。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class MendeleevCompositionFeaturizer(BaseEstimator, TransformerMixin):
    """元素プロパティを組成分率で集約する。"""

    def __init__(
        self,
        props: list[str] = [],
        stats: tuple[str, ...] = ("mean", "std", "min", "max", "range"),
        prefix: str = "md__",
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

    def fit(self, X: pd.DataFrame, y: Any = None) -> "MendeleevCompositionFeaturizer":
        try:
            from mendeleev.fetch import fetch_table
        except ImportError as exc:
            raise ImportError(
                "Mendeleev特徴量を使用するにはmendeleevを導入してください。"
            ) from exc

        properties = self.props or [
            "atomic_number",
            "atomic_weight",
            "atomic_radius",
            "covalent_radius_cordero",
            "electron_affinity",
            "boiling_point",
            "density",
        ]
        self.props_ = list(properties)

        periodic_table = fetch_table("elements").set_index("symbol", drop=False)
        self._prop_map_: dict[str, dict[str, Any]] = {}
        block_map = {"s": 0.0, "p": 1.0, "d": 2.0, "f": 3.0}
        for prop in self.props_:
            if prop not in periodic_table.columns:
                raise ValueError(f"Unknown mendeleev property: {prop}")
            if prop == "block":
                series = periodic_table[prop].map(block_map).astype(float)
            else:
                series = periodic_table[prop]
            self._prop_map_[prop] = series.to_dict()

        self.cols_out_ = [
            f"{self.prefix}{prop}__{stat}"
            for prop in self.props_
            for stat in self.stats
        ]
        self._cache_ = {} if self.use_cache else None
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
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
                fractions = composition.fractional_composition.get_el_amt_dict()
            except Exception:
                rows.append([np.nan] * len(self.cols_out_))
                continue

            symbols = list(fractions)
            weights = np.array([fractions[symbol] for symbol in symbols], dtype=float)
            features: list[float] = []
            for prop in self.props_:
                prop_map = self._prop_map_[prop]
                values = np.array(
                    [prop_map.get(symbol, np.nan) for symbol in symbols],
                    dtype=float,
                )
                statistics = self._weighted_stats(values, weights)
                features.extend(statistics.get(stat, np.nan) for stat in self.stats)

            rows.append(features)
            if self.use_cache and key is not None:
                self._cache_[key] = features
                if self.cache_max and len(self._cache_) > self.cache_max:
                    self._cache_.clear()

        return pd.DataFrame(rows, index=X.index, columns=self.cols_out_)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return np.array(self.cols_out_, dtype=object)


__all__ = ["MendeleevCompositionFeaturizer"]
