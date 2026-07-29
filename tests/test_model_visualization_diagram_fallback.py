"""Tests for framework-independent trained-model structures."""

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from malchan.app.services.estimator_structure import build_estimator_structure


def _sample_pipeline() -> Pipeline:
    """Return a representative preprocessing and predictor pipeline."""

    return Pipeline(
        steps=[
            (
                "preprocess",
                ColumnTransformer(
                    transformers=[
                        (
                            "numeric",
                            Pipeline(
                                steps=[
                                    ("imputer", SimpleImputer(strategy="mean")),
                                ]
                            ),
                            ["x", "temperature"],
                        ),
                    ]
                ),
            ),
            ("predictor", RandomForestRegressor(n_estimators=5, random_state=0)),
        ]
    )


def test_native_structure_preserves_pipeline_and_column_branches() -> None:
    """Pipeline steps and ColumnTransformer columns should remain explicit."""

    structure = build_estimator_structure(_sample_pipeline())

    assert structure.kind == "pipeline"
    assert structure.class_name == "Pipeline"
    assert [child.name for child in structure.children] == ["preprocess", "predictor"]

    preprocess = structure.children[0]
    assert preprocess.kind == "branch"
    assert preprocess.class_name == "ColumnTransformer"
    assert len(preprocess.children) == 1

    numeric = preprocess.children[0]
    assert numeric.name == "numeric"
    assert numeric.columns == ["x", "temperature"]
    assert numeric.kind == "pipeline"
    assert numeric.children[0].name == "imputer"
    assert numeric.children[0].class_name == "SimpleImputer"
    assert numeric.children[0].parameters["strategy"] == "mean"

    predictor = structure.children[1]
    assert predictor.kind == "estimator"
    assert predictor.class_name == "RandomForestRegressor"
    assert predictor.parameters["n_estimators"] == "5"
    assert predictor.parameters["random_state"] == "0"


def test_native_structure_marks_passthrough_and_dropped_columns() -> None:
    """ColumnTransformer string directives should be rendered as terminal nodes."""

    transformer = ColumnTransformer(
        transformers=[
            ("keep", "passthrough", ["x"]),
            ("remove", "drop", ["unused"]),
        ]
    )

    structure = build_estimator_structure(transformer, name="columns")

    assert structure.kind == "branch"
    assert structure.children[0].kind == "passthrough"
    assert structure.children[0].columns == ["x"]
    assert structure.children[1].kind == "dropped"
    assert structure.children[1].columns == ["unused"]
