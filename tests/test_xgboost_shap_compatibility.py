"""Regression tests for XGBoost and SHAP compatibility."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from malchan.models.explainability import get_shap_values
from malchan.models.pipelines.predictor_pipeline import make_model


@pytest.mark.parametrize("task", ["regression", "classification"])
def test_make_model_disables_native_categorical_by_default(task: str) -> None:
    """malchanの前処理済み数値特徴量ではXGBoostのカテゴリ機能を無効化する。"""
    model = make_model(
        "XGBoost",
        task=task,
        kwargs={"n_estimators": 5, "max_depth": 2, "random_state": 0},
    )

    assert model.enable_categorical is False


def test_make_model_preserves_explicit_categorical_setting() -> None:
    """利用者が明示したXGBoost設定は上書きしない。"""
    model = make_model(
        "XGBoost",
        task="regression",
        kwargs={"enable_categorical": True},
    )

    assert model.enable_categorical is True


def test_xgboost_regression_shap_runs_with_malchan_defaults() -> None:
    """XGBoost 3.3以降とSHAPの組み合わせでもTreeExplainerが実行できる。"""
    X = pd.DataFrame(
        {
            "raw_material_1": [0.1, 0.2, 0.4, 0.8, 1.6, 3.2],
            "temperature": [80.0, 90.0, 100.0, 110.0, 120.0, 130.0],
        }
    )
    y = 2.0 * X["raw_material_1"] + 0.1 * X["temperature"]
    model = make_model(
        "XGBoost",
        task="regression",
        kwargs={
            "n_estimators": 5,
            "max_depth": 2,
            "learning_rate": 0.1,
            "random_state": 0,
            "verbosity": 0,
        },
    )
    model.fit(X, y)

    values, base_values, explainer, sample = get_shap_values(model, X)

    assert values is not None
    assert base_values is not None
    assert explainer is not None
    assert sample is not None
    assert np.asarray(values).shape == X.shape
