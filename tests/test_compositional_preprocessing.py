import numpy as np
import pandas as pd
import pytest

from malchan.preprocessing.compositional import (
    ALRTransformer,
    CLRTransformer,
    ILRTransformer,
    make_compositional_preprocess,
)


def test_ilr_matches_helmert_coordinates_and_reduces_dimension() -> None:
    frame = pd.DataFrame(
        {
            "A": [0.5, 0.2],
            "B": [0.3, 0.3],
            "C": [0.2, 0.5],
        }
    )
    transformer = ILRTransformer().fit(frame)

    transformed = transformer.transform(frame)

    expected_first = np.log(frame["A"] / frame["B"]) / np.sqrt(2.0)
    expected_second = (
        np.log(frame["A"]) + np.log(frame["B"]) - 2.0 * np.log(frame["C"])
    ) / np.sqrt(6.0)
    assert transformed.shape == (2, 2)
    np.testing.assert_allclose(transformed[:, 0], expected_first)
    np.testing.assert_allclose(transformed[:, 1], expected_second)
    assert transformer.get_feature_names_out().tolist() == [
        "ilr__balance_1",
        "ilr__balance_2",
    ]
    assert transformer.get_balance_definitions() == [
        {"feature": "ilr__balance_1", "positive": ["A"], "negative": ["B"]},
        {"feature": "ilr__balance_2", "positive": ["A", "B"], "negative": ["C"]},
    ]


def test_ilr_is_invariant_to_total_scale() -> None:
    fractions = np.asarray([[0.5, 0.3, 0.2], [0.1, 0.4, 0.5]])
    percentages = fractions * 100.0

    transformer = ILRTransformer().fit(fractions)

    np.testing.assert_allclose(
        transformer.transform(fractions),
        transformer.transform(percentages),
    )


def test_zero_replacement_keeps_log_ratio_outputs_finite() -> None:
    data = np.asarray([[0.6, 0.4, 0.0], [0.0, 0.2, 0.8]])

    transformed = ILRTransformer(zero_replacement=1e-4).fit_transform(data)

    assert np.all(np.isfinite(transformed))


def test_zero_without_replacement_is_rejected() -> None:
    data = np.asarray([[0.6, 0.4, 0.0]])

    with pytest.raises(ValueError, match="zero_replacement"):
        ILRTransformer(zero_replacement=None).fit(data)


def test_negative_and_empty_compositions_are_rejected() -> None:
    with pytest.raises(ValueError, match="負の値"):
        ILRTransformer().fit([[0.5, -0.1, 0.6]])
    with pytest.raises(ValueError, match="少なくとも1つ正の値"):
        ILRTransformer().fit([[0.0, 0.0, 0.0]])


def test_clr_and_alr_are_available_from_same_factory() -> None:
    frame = pd.DataFrame({"A": [0.5], "B": [0.3], "C": [0.2]})

    clr = CLRTransformer().fit_transform(frame)
    alr_transformer = ALRTransformer(reference="C").fit(frame)
    alr = alr_transformer.transform(frame)

    assert clr.shape == (1, 3)
    np.testing.assert_allclose(clr.sum(axis=1), 0.0, atol=1e-12)
    np.testing.assert_allclose(alr, [[np.log(0.5 / 0.2), np.log(0.3 / 0.2)]])
    assert alr_transformer.get_feature_names_out().tolist() == [
        "alr__A_over_C",
        "alr__B_over_C",
    ]
    assert make_compositional_preprocess("ILR") is not None
    assert make_compositional_preprocess("CLR") is not None
    assert make_compositional_preprocess("ALR") is not None


def test_unknown_method_and_scaler_are_rejected() -> None:
    with pytest.raises(ValueError, match="compositional_method"):
        make_compositional_preprocess("unknown")
    with pytest.raises(ValueError, match="compositional_scale_type"):
        make_compositional_preprocess("ILR", scale_type="unknown")


def test_make_preprocess_routes_composition_columns_only_through_log_ratio_branch() -> None:
    from malchan.models.pipelines.preprocess_pipeline import make_preprocess

    frame = pd.DataFrame(
        {
            "A": [50.0, 20.0, 30.0],
            "B": [30.0, 30.0, 40.0],
            "C": [20.0, 50.0, 30.0],
            "temperature": [1000.0, 1100.0, 1200.0],
        }
    )
    preprocess = make_preprocess(
        model_name="Ridge",
        num_cols=["A", "B", "C", "temperature"],
        compositional_groups=[["A", "B", "C"]],
        compositional_method="ILR",
    )

    transformed = preprocess.fit_transform(frame)

    assert transformed.shape == (3, 3)
    column_transformer = preprocess.named_steps["column_preprocess"]
    transformer_names = [name for name, _, _ in column_transformer.transformers_]
    assert transformer_names == ["num_cat", "compositional_0"]
    num_cat_columns = column_transformer.transformers_[0][2]
    assert num_cat_columns == ["temperature"]


def test_multiple_compositional_groups_are_transformed_independently() -> None:
    from malchan.models.pipelines.preprocess_pipeline import make_preprocess

    frame = pd.DataFrame(
        {
            "A": [0.6, 0.4],
            "B": [0.4, 0.6],
            "X": [0.5, 0.2],
            "Y": [0.3, 0.3],
            "Z": [0.2, 0.5],
        }
    )
    preprocess = make_preprocess(
        model_name="Ridge",
        num_cols=list(frame.columns),
        compositional_groups=[["A", "B"], ["X", "Y", "Z"]],
    )

    transformed = preprocess.fit_transform(frame)

    assert transformed.shape == (2, 3)
    assert [
        name
        for name, _, _ in preprocess.named_steps["column_preprocess"].transformers_
    ] == ["compositional_0", "compositional_1"]


def test_compositional_groups_reject_duplicate_assignments() -> None:
    from malchan.models.pipelines.preprocess_pipeline import make_preprocess

    with pytest.raises(ValueError, match="複数"):
        make_preprocess(
            model_name="Ridge",
            num_cols=["A", "B", "C"],
            compositional_groups=[["A", "B"], ["B", "C"]],
        )
