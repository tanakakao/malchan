"""分子fingerprintの解決と特徴量名保持。"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


SUPPORTED_FINGERPRINTS = (
    "AtomPair",
    "Autocorr",
    "Avalon",
    "E3FP",
    "ECFP",
    "MACCS",
    "MORSE",
    "PhysChem",
    "PubChem",
    "RDF",
    "RDKit",
)
THREE_DIMENSIONAL_FINGERPRINTS = frozenset({"Autocorr", "E3FP", "MORSE", "RDF"})


def _fingerprint_factories() -> dict[str, Callable[[], Any]]:
    """利用可能なscikit-fingerprints実装の生成関数を返す。"""
    try:
        from skfp.fingerprints import (
            AtomPairFingerprint,
            AutocorrFingerprint,
            AvalonFingerprint,
            E3FPFingerprint,
            ECFPFingerprint,
            MACCSFingerprint,
            MORSEFingerprint,
            PhysiochemicalPropertiesFingerprint,
            PubChemFingerprint,
            RDFFingerprint,
            RDKit2DDescriptorsFingerprint,
        )
    except ImportError as exc:
        raise ImportError(
            "分子fingerprintを使用するにはscikit-fingerprintsを導入してください。"
        ) from exc

    return {
        "ECFP": lambda: ECFPFingerprint(count=True),
        "MACCS": MACCSFingerprint,
        "RDKit": RDKit2DDescriptorsFingerprint,
        "PubChem": PubChemFingerprint,
        "AtomPair": AtomPairFingerprint,
        "Avalon": AvalonFingerprint,
        "PhysChem": PhysiochemicalPropertiesFingerprint,
        "Autocorr": lambda: AutocorrFingerprint(use_3D=True),
        "E3FP": E3FPFingerprint,
        "MORSE": MORSEFingerprint,
        "RDF": RDFFingerprint,
    }


def available_fingerprints() -> tuple[str, ...]:
    """依存パッケージを読み込まず、対応fingerprint名を返す。"""
    return SUPPORTED_FINGERPRINTS


def resolve_fingerprints(names: Sequence[str]) -> list[Any]:
    """APIで指定された名前をfingerprint Transformerへ変換する。"""
    unknown = [name for name in names if name not in SUPPORTED_FINGERPRINTS]
    if unknown:
        raise ValueError(
            f"未対応のfingerprintです: {unknown}. 利用可能: {list(SUPPORTED_FINGERPRINTS)}"
        )
    factories = _fingerprint_factories()
    return [factories[name]() for name in names]


def requires_conformers(names: Sequence[str]) -> bool:
    """指定fingerprintが3次元配座を必要とするか判定する。"""
    return any(name in THREE_DIMENSIONAL_FINGERPRINTS for name in names)


class PassthroughNames(BaseEstimator, TransformerMixin):
    """匿名のfingerprint配列へ安定した特徴量名を付与する。"""

    def fit(self, X: Any, y: Any = None) -> "PassthroughNames":
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: Any) -> Any:
        return X

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        if input_features is None:
            return np.array(
                [f"feat_{index}" for index in range(self.n_features_in_)],
                dtype=object,
            )
        return np.asarray(input_features, dtype=object)


__all__ = [
    "PassthroughNames",
    "SUPPORTED_FINGERPRINTS",
    "THREE_DIMENSIONAL_FINGERPRINTS",
    "available_fingerprints",
    "requires_conformers",
    "resolve_fingerprints",
]
