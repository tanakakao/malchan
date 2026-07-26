"""モデル系オプション依存の互換性を確認するテスト。"""

import pytest


def test_optuna_sklearn_integration_imports() -> None:
    """OptunaSearchCVの公開APIを問題なくimportできることを確認する。"""
    pytest.importorskip("optuna")
    pytest.importorskip("optuna_integration")

    from optuna.integration import OptunaSearchCV

    assert OptunaSearchCV is not None
