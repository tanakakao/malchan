import pandas as pd

from malchan.llm import (
    ComparisonConfigSuggestion,
    DatasetSummary,
    ModelSpec,
    ModelSpecRegistry,
    SuggestionValidator,
    TrainingConfigSuggestion,
    TrainingSuggestion,
)


def _regression_summary(n_rows: int = 8) -> DatasetSummary:
    df = pd.DataFrame(
        {
            "x1": list(range(n_rows)),
            "x2": list(range(n_rows, 2 * n_rows)),
            "y": [float(value) for value in range(n_rows)],
        }
    )
    return DatasetSummary.from_dataframe(
        df,
        target_cols=["y"],
        tasks=["regression"],
        num_cols=["x1", "x2"],
        cat_cols=[],
    )


def _registry() -> ModelSpecRegistry:
    return ModelSpecRegistry(
        [
            ModelSpec(
                name="K近傍法",
                task="regression",
                family="neighbors",
                allowed_params=frozenset({"n_neighbors", "weights"}),
            ),
            ModelSpec(
                name="PLS回帰",
                task="regression",
                family="pls",
                allowed_params=frozenset({"n_components"}),
            ),
            ModelSpec(
                name="Ridge",
                task="regression",
                family="linear",
                allowed_params=frozenset({"alpha"}),
            ),
        ]
    )


def test_validator_adjusts_fold_sensitive_parameters_and_components():
    suggestion = TrainingSuggestion(
        training=TrainingConfigSuggestion(
            model_names_by_target={"y": ["K近傍法"]},
            model_params_by_target={"y": {"n_neighbors": 100}},
            decomposition=True,
            dec_n_components=100,
        ),
        comparison=ComparisonConfigSuggestion(n_splits=20),
    )

    result = SuggestionValidator(_registry()).validate(
        _regression_summary(),
        suggestion,
    )

    assert result.status == "adjusted"
    assert result.suggestion.comparison.n_splits == 8
    assert result.suggestion.training.dec_n_components == 2
    assert (
        result.suggestion.training.model_params_by_target["y"][
            "n_neighbors"
        ]
        == 7
    )
    assert suggestion.comparison.n_splits == 20


def test_validator_rejects_unknown_model():
    suggestion = TrainingSuggestion(
        training=TrainingConfigSuggestion(
            model_names_by_target={"y": ["Unknown"]},
            model_params_by_target={"y": {"bad_param": 1}},
        )
    )

    result = SuggestionValidator(_registry()).validate(
        _regression_summary(),
        suggestion,
    )

    assert result.status == "rejected"
    assert any(issue.code == "unknown_model" for issue in result.issues)


def test_validator_rejects_parameter_not_supported_by_registered_model():
    suggestion = TrainingSuggestion(
        training=TrainingConfigSuggestion(
            model_names_by_target={"y": ["Ridge"]},
            model_params_by_target={"y": {"max_depth": 4}},
        )
    )

    result = SuggestionValidator(_registry()).validate(
        _regression_summary(),
        suggestion,
    )

    assert result.status == "rejected"
    assert any(
        issue.code == "unknown_model_parameter"
        for issue in result.issues
    )


def test_validator_rejects_excessive_polynomial_expansion():
    suggestion = TrainingSuggestion(
        training=TrainingConfigSuggestion(
            model_names_by_target={"y": ["Ridge"]},
            poly=True,
            poly_degree=8,
            poly_interaction_only=False,
        )
    )

    result = SuggestionValidator(
        _registry(),
        max_polynomial_features=10,
    ).validate(_regression_summary(), suggestion)

    assert result.status == "rejected"
    assert any(
        issue.code == "polynomial_feature_explosion"
        for issue in result.issues
    )


def test_validator_requires_comparison_for_multiple_non_ensemble_models():
    suggestion = TrainingSuggestion(
        training=TrainingConfigSuggestion(
            model_names_by_target={"y": ["Ridge", "PLS回帰"]},
        )
    )

    result = SuggestionValidator(_registry()).validate(
        _regression_summary(),
        suggestion,
    )

    assert result.status == "rejected"
    assert any(
        issue.code == "multiple_models_without_ensemble"
        for issue in result.issues
    )
