"""材料特徴量実装のモジュール境界を確認するテスト。"""

from __future__ import annotations

import subprocess
import sys

import pytest


def test_tabular_preprocess_does_not_import_xenonpy() -> None:
    """組成列を使わない前処理ではXenonPyを要求しない。"""
    script = """
import sys
from malchan.models.pipelines.preprocess_pipeline import make_preprocess

pipeline = make_preprocess(
    model_name='線形回帰',
    num_cols=['x'],
    cat_cols=[],
    smiles_cols=[],
    comp_cols=[],
)
assert pipeline is not None
print(any(name == 'xenonpy' or name.startswith('xenonpy.') for name in sys.modules))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False"


def test_material_implementations_are_not_exported_from_preprocess_pipeline() -> None:
    """移動済み材料クラスを旧モジュールへ互換公開しない。"""
    from malchan.models.pipelines import preprocess_pipeline

    removed_names = {
        "FormulaToComposition",
        "FormulaToFractionDict",
        "MatminerCompositionFeaturizer",
        "MendeleevCompositionFeaturizer",
        "XenoCompositionsTransformer",
        "make_comp_preprocess",
        "xenonpy_prest",
    }

    assert removed_names.isdisjoint(vars(preprocess_pipeline))


def test_composition_transformers_are_available_from_materials() -> None:
    """組成式変換をmaterials公開APIから利用できる。"""
    pytest.importorskip("pymatgen")

    from malchan.features.materials import (
        FormulaToComposition,
        FormulaToFractionDict,
    )

    assert FormulaToComposition is not None
    assert FormulaToFractionDict is not None


def test_matminer_pipeline_uses_selected_features() -> None:
    """Matminerバックエンドを選択した場合だけ関連モジュールを構築する。"""
    pytest.importorskip("matminer")

    from malchan.features.materials import make_comp_preprocess

    pipeline = make_comp_preprocess(
        method="matminer",
        feats=["Stoichiometry", "TMetalFraction"],
    )

    assert pipeline is not None
    assert list(pipeline.named_steps) == ["f2c", "mm", "imp", "sc"]
    assert len(pipeline.named_steps["mm"].featurizers) == 2


def test_material_pipeline_rejects_unknown_backend() -> None:
    """未対応バックエンドは明示的に拒否する。"""
    from malchan.features.materials import make_comp_preprocess

    with pytest.raises(ValueError, match="comp_method"):
        make_comp_preprocess(method="unknown")
