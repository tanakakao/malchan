"""化学特徴量実装のモジュール境界を確認するテスト。"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"


def test_tabular_preprocess_does_not_import_chemistry_backends() -> None:
    """分子列を使わない前処理ではRDKitとscikit-fingerprintsを要求しない。"""
    code = """
import sys
from malchan.models.pipelines.preprocess_pipeline import make_preprocess

pipeline = make_preprocess(
    model_name="線形回帰",
    num_cols=["x"],
    cat_cols=[],
    smiles_cols=[],
    comp_cols=[],
)
assert pipeline is not None
assert not any(name == "skfp" or name.startswith("skfp.") for name in sys.modules)
assert not any(name == "rdkit" or name.startswith("rdkit.") for name in sys.modules)
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(SRC_PATH), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_chemistry_implementations_are_not_exported_from_preprocess_pipeline() -> None:
    """移動済みchemistry実装を旧モジュールへ互換公開しない。"""
    from malchan.models.pipelines import preprocess_pipeline

    removed_names = {
        "FINGERPRINTS",
        "PassthroughNames",
        "SmilesToMol",
        "make_smiles_preprocess",
    }
    assert removed_names.isdisjoint(vars(preprocess_pipeline))


def test_smiles_transformer_is_available_from_chemistry() -> None:
    """SMILES変換器をchemistry公開APIから利用できる。"""
    from malchan.features.chemistry import SmilesToMol

    assert SmilesToMol is not None


def test_empty_fingerprint_selection_returns_none() -> None:
    """fingerprint未指定ではchemistry依存を読み込まずNoneを返す。"""
    from malchan.features.chemistry import make_smiles_preprocess

    assert make_smiles_preprocess() is None


def test_smiles_pipeline_uses_selected_fingerprints() -> None:
    """選択したfingerprintだけでSMILES Pipelineを構築する。"""
    pytest.importorskip("skfp")

    from malchan.features.chemistry import make_smiles_preprocess

    pipeline = make_smiles_preprocess(["ECFP", "MACCS"])

    assert pipeline is not None
    assert list(pipeline.named_steps) == ["to_mol", "fp", "sc", "names"]
    assert len(pipeline.named_steps["fp"].transformer_list) == 2
    assert pipeline.named_steps["to_mol"].generate_conformers is False


def test_three_dimensional_fingerprint_enables_conformers() -> None:
    """3次元fingerprint指定時は配座生成を有効にする。"""
    pytest.importorskip("skfp")

    from malchan.features.chemistry import make_smiles_preprocess

    pipeline = make_smiles_preprocess(["E3FP"])

    assert pipeline is not None
    assert pipeline.named_steps["to_mol"].generate_conformers is True


def test_smiles_pipeline_rejects_unknown_fingerprint() -> None:
    """未対応fingerprintは明示的に拒否する。"""
    pytest.importorskip("skfp")

    from malchan.features.chemistry import make_smiles_preprocess

    with pytest.raises(ValueError, match="fingerprint"):
        make_smiles_preprocess(["unknown"])
