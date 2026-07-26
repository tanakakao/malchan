"""Regression tests for XAI dependency isolation."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from malchan.models.explainability import get_shap_values


FEATURE_BACKENDS = {"matminer", "pymatgen", "rdkit", "skfp", "xenonpy"}


def test_explainability_does_not_import_feature_backends() -> None:
    """Generic XAI must not require chemistry or materials backends."""
    module_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "malchan"
        / "models"
        / "explainability.py"
    )
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(
                alias.name.split(".", maxsplit=1)[0]
                for alias in node.names
            )
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".", maxsplit=1)[0])

    assert imported_roots.isdisjoint(FEATURE_BACKENDS)


def test_linear_shap_runs_without_material_backends() -> None:
    """Linear-regression SHAP uses only generic model dependencies."""
    X = pd.DataFrame(
        {
            "raw_material_1": [0.1, 0.2, 0.4, 0.8, 1.6],
            "temperature": [80.0, 90.0, 100.0, 110.0, 120.0],
        }
    )
    y = 2.0 * X["raw_material_1"] + 0.1 * X["temperature"]
    model = LinearRegression().fit(X, y)

    values, base_values, explainer, sample = get_shap_values(model, X)

    assert values is not None
    assert base_values is not None
    assert explainer is not None
    assert sample is not None
    assert np.asarray(values).shape == X.shape
