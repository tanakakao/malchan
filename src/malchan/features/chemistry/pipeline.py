"""SMILES列向けの特徴量生成Pipeline。"""

from __future__ import annotations

from collections.abc import Sequence

from sklearn.pipeline import Pipeline, make_union
from sklearn.preprocessing import StandardScaler

from .fingerprints import (
    PassthroughNames,
    requires_conformers,
    resolve_fingerprints,
)
from .smiles import SmilesToMol


def make_smiles_preprocess(
    fingerprints: Sequence[str] = (),
) -> Pipeline | None:
    """SMILES列向けのfingerprint生成Pipelineを作成する。

    Args:
        fingerprints: 使用するfingerprint名。空の場合はNoneを返す。

    Returns:
        SMILES変換、fingerprint生成、スケーリングをまとめたPipeline。

    Raises:
        ImportError: chemistry依存パッケージが未導入の場合。
        ValueError: 未対応のfingerprint名が指定された場合。
    """
    if not fingerprints:
        return None

    fingerprint_names = tuple(fingerprints)
    selected = resolve_fingerprints(fingerprint_names)
    return Pipeline(
        [
            (
                "to_mol",
                SmilesToMol(
                    generate_conformers=requires_conformers(fingerprint_names)
                ),
            ),
            ("fp", make_union(*selected)),
            ("sc", StandardScaler(with_mean=False)),
            ("names", PassthroughNames()),
        ]
    )


__all__ = ["make_smiles_preprocess"]
