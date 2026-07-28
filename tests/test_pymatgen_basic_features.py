"""Tests for lightweight pymatgen composition statistics."""

import importlib.util

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("pymatgen") is None,
    reason="Pymatgen feature tests require the materials extra.",
)


def _composition_frame(formulas: list[str]) -> pd.DataFrame:
    """Convert formulas to the one-column Composition frame used by featurizers."""

    from malchan.features.materials.composition import FormulaToComposition

    return FormulaToComposition(invalid="error").fit_transform(pd.DataFrame({"formula": formulas}))


def test_pymatgen_basic_featurizer_generates_summary_and_weighted_statistics() -> None:
    """Selected properties should receive five statistics plus fixed summaries."""

    from malchan.features.materials.pymatgen_basic import (
        COMPOSITION_SUMMARY_FEATURES,
        PymatgenBasicCompositionFeaturizer,
    )

    frame = _composition_frame(["LiFePO4", "NaCl"])
    featurizer = PymatgenBasicCompositionFeaturizer(
        props=["atomic_number", "electronegativity"],
        prefix="comp__",
    )

    transformed = featurizer.fit_transform(frame)

    assert transformed.shape == (2, len(COMPOSITION_SUMMARY_FEATURES) + 10)
    assert transformed.columns.tolist() == [
        "comp__n_elements",
        "comp__reduced_num_atoms",
        "comp__reduced_formula_weight",
        "comp__composition_entropy",
        "comp__min_fraction",
        "comp__max_fraction",
        "comp__atomic_number__mean",
        "comp__atomic_number__std",
        "comp__atomic_number__min",
        "comp__atomic_number__max",
        "comp__atomic_number__range",
        "comp__electronegativity__mean",
        "comp__electronegativity__std",
        "comp__electronegativity__min",
        "comp__electronegativity__max",
        "comp__electronegativity__range",
    ]
    assert transformed.loc[0, "comp__n_elements"] == 4.0
    assert transformed.loc[0, "comp__reduced_num_atoms"] == 7.0
    assert np.isfinite(transformed.to_numpy(dtype=float)).all()


def test_pymatgen_basic_statistics_are_invariant_to_formula_scaling() -> None:
    """Equivalent formulas should produce identical reduced-composition features."""

    from malchan.features.materials.pymatgen_basic import PymatgenBasicCompositionFeaturizer

    frame = _composition_frame(["LiFePO4", "Li2Fe2P2O8"])
    transformed = PymatgenBasicCompositionFeaturizer(
        props=["atomic_number", "atomic_mass", "row", "group"],
    ).fit_transform(frame)

    np.testing.assert_allclose(
        transformed.iloc[0].to_numpy(dtype=float),
        transformed.iloc[1].to_numpy(dtype=float),
        rtol=1e-12,
        atol=1e-12,
    )


def test_pymatgen_basic_featurizer_rejects_unknown_properties() -> None:
    """Unknown property names should fail during fitting with an actionable message."""

    from malchan.features.materials.pymatgen_basic import PymatgenBasicCompositionFeaturizer

    frame = _composition_frame(["LiFePO4", "NaCl"])

    with pytest.raises(ValueError, match="Unknown pymatgen properties"):
        PymatgenBasicCompositionFeaturizer(props=["unknown"]).fit(frame)


def test_composition_pipeline_uses_pymatgen_as_the_default_app_method() -> None:
    """The default composition preprocessing pipeline should use pymatgen statistics."""

    from malchan.features.materials.pipeline import (
        SUPPORTED_COMPOSITION_METHODS,
        make_comp_preprocess,
    )

    pipeline = make_comp_preprocess(feats=["atomic_number"])

    assert SUPPORTED_COMPOSITION_METHODS == ("pymatgen", "matminer", "mendeleev")
    assert "xenonpy" not in SUPPORTED_COMPOSITION_METHODS
    assert "pmg" in pipeline.named_steps
