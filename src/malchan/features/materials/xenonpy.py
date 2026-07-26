"""XenonPyによる組成特徴量生成。"""

from __future__ import annotations

import shutil
from collections.abc import Hashable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from .composition import composition_cache_key


def _load_xenonpy_components() -> tuple[Any, Any, Any]:
    """XenonPyの設定、preset、Compositionsを遅延importする。"""
    try:
        from xenonpy._conf import __cfg_root__
        from xenonpy.datatools import preset
        from xenonpy.descriptor import Compositions
    except ImportError as exc:
        raise ImportError(
            "XenonPy特徴量を使用するにはxenonpyを導入してください。"
        ) from exc
    return __cfg_root__, preset, Compositions


def prepare_xenonpy_preset(source: str | Path = "elements_completed.pd.xz") -> bool:
    """配布済み元素データが存在する場合にXenonPyのdatasetへ登録する。

    Args:
        source: ``elements_completed.pd.xz`` の配置場所。

    Returns:
        データをコピーしてインデックスを更新した場合はTrue。
    """
    source_path = Path(source)
    if not source_path.exists():
        return False

    config_root, preset, _ = _load_xenonpy_components()
    destination_dir = Path(config_root) / "dataset"
    destination_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, destination_dir / source_path.name)
    preset._make_index(prefix=["dataset"])
    return True


class XenoCompositionsTransformer(BaseEstimator, TransformerMixin):
    """XenonPy Compositionsをsklearn Transformerとして扱う。"""

    def __init__(
        self,
        prefix: str = "comp__",
        sample_in_fit: int = 1,
        *,
        n_jobs: int = 1,
        featurizers: str | list[str] = "classic",
        on_errors: str = "nan",
        cache: bool = True,
        key_round_digits: int = 12,
    ):
        self.prefix = prefix
        self.sample_in_fit = sample_in_fit
        self.n_jobs = n_jobs
        self.featurizers = featurizers
        self.on_errors = on_errors
        self.cache = cache
        self.key_round_digits = key_round_digits

    def fit(self, X: pd.DataFrame, y: Any = None) -> "XenoCompositionsTransformer":
        _, _, compositions_cls = _load_xenonpy_components()
        self._comp = compositions_cls(
            n_jobs=self.n_jobs,
            featurizers=self.featurizers,
            on_errors=self.on_errors,
        )

        series = X.iloc[:, 0]
        if self.sample_in_fit and len(series) > self.sample_in_fit:
            series = series.iloc[: self.sample_in_fit]

        probe = list(series.values)
        probe.append({"Si": 1.0})
        frame = self._comp.transform(probe)
        self._cols_in_ = list(frame.columns)
        self._cols_out_ = [f"{self.prefix}{column}" for column in self._cols_in_]
        self._cache: dict[Hashable, np.ndarray] = {}
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        series = X.iloc[:, 0]
        if series.empty:
            return pd.DataFrame(
                index=series.index,
                columns=self._cols_out_,
                dtype=float,
            )

        keys = [
            composition_cache_key(value, ndigits=self.key_round_digits)
            for value in series.values
        ]
        unique: dict[Hashable, Any] = {}
        for key, value in zip(keys, series.values, strict=False):
            unique.setdefault(key, value)

        if self.cache:
            pending = [
                (key, value)
                for key, value in unique.items()
                if key not in self._cache
            ]
        else:
            pending = list(unique.items())
            self._cache = {}

        if pending:
            frame = self._comp.transform([value for _, value in pending])
            for column in self._cols_in_:
                if column not in frame.columns:
                    frame[column] = np.nan
            values = frame[self._cols_in_].to_numpy(dtype=float, copy=False)
            for (key, _), row in zip(pending, values, strict=False):
                self._cache[key] = row

        array = np.vstack([self._cache[key] for key in keys])
        return pd.DataFrame(array, index=series.index, columns=self._cols_out_)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return np.array(self._cols_out_, dtype=object)


__all__ = [
    "XenoCompositionsTransformer",
    "prepare_xenonpy_preset",
]
