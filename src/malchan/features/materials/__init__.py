"""材料特徴量生成の公開API。"""

from importlib import import_module
from typing import Any

_LAZY_EXPORTS = {
    "FormulaToComposition": "malchan.features.materials.composition",
    "FormulaToFractionDict": "malchan.features.materials.composition",
    "MatminerCompositionFeaturizer": "malchan.features.materials.matminer",
    "MendeleevCompositionFeaturizer": "malchan.features.materials.mendeleev",
    "PymatgenBasicCompositionFeaturizer": "malchan.features.materials.pymatgen_basic",
    "SUPPORTED_PYMATGEN_PROPERTIES": "malchan.features.materials.pymatgen_basic",
    "XenoCompositionsTransformer": "malchan.features.materials.xenonpy",
    "make_comp_preprocess": "malchan.features.materials.pipeline",
    "prepare_xenonpy_preset": "malchan.features.materials.xenonpy",
    "resolve_matminer_featurizers": "malchan.features.materials.matminer",
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    """任意依存を必要な機能の利用時まで読み込まない。"""
    if name not in _LAZY_EXPORTS:
        raise AttributeError(
            f"module 'malchan.features.materials' has no attribute {name!r}"
        )

    module = import_module(_LAZY_EXPORTS[name])
    value = getattr(module, name)
    globals()[name] = value
    return value
