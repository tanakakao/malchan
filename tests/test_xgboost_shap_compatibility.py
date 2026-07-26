"""Regression tests for XGBoost and SHAP compatibility."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from malchan.models.explainability import get_shap_values


def test_make_model_disables_native_categorical_by_default() -> None:
    """XGBoost生成時にカテゴリ機能を既定で無効化する。"""
    module_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "malchan"
        / "models"
        / "pipelines"
        / "predictor_pipeline.py"
    )
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    make_model = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "make_model"
    )

    xgboost_branch = next(
        node
        for node in make_model.body
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == "model_name"
        and any(
            isinstance(comparator, ast.Constant)
            and comparator.value == "XGBoost"
            for comparator in node.test.comparators
        )
    )
    setdefault_calls = [
        node
        for node in ast.walk(xgboost_branch)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "kwargs"
        and node.func.attr == "setdefault"
    ]

    assert any(
        len(call.args) == 2
        and isinstance(call.args[0], ast.Constant)
        and call.args[0].value == "enable_categorical"
        and isinstance(call.args[1], ast.Constant)
        and call.args[1].value is False
        for call in setdefault_calls
    )


def test_xgboost_regression_shap_runs_with_native_categorical_disabled() -> None:
    """XGBoost 3.3以降とSHAPの組み合わせでもTreeExplainerが実行できる。"""
    X = pd.DataFrame(
        {
            "raw_material_1": [0.1, 0.2, 0.4, 0.8, 1.6, 3.2],
            "temperature": [80.0, 90.0, 100.0, 110.0, 120.0, 130.0],
        }
    )
    y = 2.0 * X["raw_material_1"] + 0.1 * X["temperature"]
    model = XGBRegressor(
        n_estimators=5,
        max_depth=2,
        learning_rate=0.1,
        random_state=0,
        verbosity=0,
        enable_categorical=False,
    ).fit(X, y)

    values, base_values, explainer, sample = get_shap_values(model, X)

    assert values is not None
    assert base_values is not None
    assert explainer is not None
    assert sample is not None
    assert np.asarray(values).shape == X.shape
