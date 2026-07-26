"""化学特徴量生成の公開API。"""

from importlib import import_module
from typing import Any


_LAZY_EXPORTS = {
    "PassthroughNames": "malchan.features.chemistry.fingerprints",
    "SmilesToMol": "malchan.features.chemistry.smiles",
    "available_fingerprints": "malchan.features.chemistry.fingerprints",
    "make_smiles_preprocess": "malchan.features.chemistry.pipeline",
    "requires_conformers": "malchan.features.chemistry.fingerprints",
    "resolve_fingerprints": "malchan.features.chemistry.fingerprints",
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    """化学特徴量実装を必要になった時点で読み込む。"""
    if name not in _LAZY_EXPORTS:
        raise AttributeError(
            f"module 'malchan.features.chemistry' has no attribute {name!r}"
        )

    module = import_module(_LAZY_EXPORTS[name])
    value = getattr(module, name)
    globals()[name] = value
    return value
