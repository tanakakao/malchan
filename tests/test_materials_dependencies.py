"""材料系オプション依存の互換性を確認するテスト。"""

import pytest


def test_scikit_fingerprints_imports_with_mordredcommunity() -> None:
    """scikit-fingerprintsの公開APIを問題なくimportできることを確認する。"""
    pytest.importorskip("skfp")
    pytest.importorskip("mordred")

    from skfp.fingerprints import ECFPFingerprint, MordredFingerprint

    assert ECFPFingerprint is not None
    assert MordredFingerprint is not None
