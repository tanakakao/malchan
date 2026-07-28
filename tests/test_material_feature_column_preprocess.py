"""Regression tests for multiple SMILES and composition columns."""

import importlib.util

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("imblearn") is None,
    reason="Material preprocessing tests require the models extra.",
)


def _identity_pipeline():
    """Return a cloneable one-column transformer without optional chemistry packages."""

    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer

    return Pipeline(
        [
            (
                "identity",
                FunctionTransformer(
                    lambda frame: frame.to_numpy(dtype=float),
                    validate=False,
                ),
            )
        ]
    )


def test_smiles_and_composition_columns_are_transformed_independently() -> None:
    """Every raw material string column should keep the original sample count."""

    from malchan.models.pipelines.preprocess_pipeline import (
        make_categorical_preprocess,
        make_numeric_preprocess,
        make_preprocess_pipeline,
    )

    frame = pd.DataFrame(
        {
            "smiles_a": [1.0, 2.0, 3.0],
            "smiles_b": [10.0, 20.0, 30.0],
            "formula_a": [100.0, 200.0, 300.0],
            "formula_b": [1000.0, 2000.0, 3000.0],
        }
    )
    preprocess = make_preprocess_pipeline(
        num_process=make_numeric_preprocess(),
        cat_process=make_categorical_preprocess("Ridge"),
        smiles_process=_identity_pipeline(),
        comp_process=_identity_pipeline(),
        smiles_cols=["smiles_a", "smiles_b"],
        comp_cols=["formula_a", "formula_b"],
    )

    transformed = preprocess.fit_transform(frame)

    assert transformed.shape == (3, 4)
    np.testing.assert_allclose(
        transformed,
        frame[["smiles_a", "smiles_b", "formula_a", "formula_b"]].to_numpy(),
    )
    transformer_names = [
        name
        for name, _, _ in preprocess.named_steps["column_preprocess"].transformers_
    ]
    assert transformer_names == ["smiles_0", "smiles_1", "comp_0", "comp_1"]
