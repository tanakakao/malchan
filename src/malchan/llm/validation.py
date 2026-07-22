"""Semantic validation and safe adjustment of LLM-generated suggestions."""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any, Literal

from .registry import ModelSpec, ModelSpecRegistry
from .suggestions import TrainingSuggestion
from .summary import DatasetSummary

IssueSeverity = Literal["error", "warning", "adjustment"]
ValidationStatus = Literal["accepted", "adjusted", "rejected"]


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    """One semantic validation finding."""

    severity: IssueSeverity
    code: str
    field: str
    message: str
    original_value: Any = None
    adjusted_value: Any = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SuggestionValidationResult:
    """Validated suggestion and all issues found during validation."""

    status: ValidationStatus
    suggestion: TrainingSuggestion
    issues: list[ValidationIssue]

    @property
    def is_valid(self) -> bool:
        """Return whether the suggestion can be applied."""

        return self.status != "rejected"

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "suggestion": self.suggestion.to_dict(),
            "issues": [issue.to_dict() for issue in self.issues],
        }


class SuggestionValidator:
    """Validate LLM suggestions against malchan capabilities and dataset size."""

    def __init__(
        self,
        registry: ModelSpecRegistry,
        *,
        max_polynomial_features: int = 5_000,
    ) -> None:
        if max_polynomial_features < 1:
            raise ValueError("max_polynomial_features must be positive.")
        self.registry = registry
        self.max_polynomial_features = max_polynomial_features

    def validate(
        self,
        summary: DatasetSummary,
        suggestion: TrainingSuggestion,
    ) -> SuggestionValidationResult:
        """Return a validated copy without mutating ``suggestion``."""

        adjusted = copy.deepcopy(suggestion)
        issues: list[ValidationIssue] = []
        self._validate_targets(summary, adjusted, issues)
        self._validate_ensemble(summary, adjusted, issues)
        self._validate_comparison(summary, adjusted, issues)
        self._validate_preprocessing(summary, adjusted, issues)
        self._validate_models(summary, adjusted, issues)

        if any(issue.severity == "error" for issue in issues):
            status: ValidationStatus = "rejected"
        elif any(issue.severity == "adjustment" for issue in issues):
            status = "adjusted"
        else:
            status = "accepted"
        return SuggestionValidationResult(
            status=status,
            suggestion=adjusted,
            issues=issues,
        )

    @staticmethod
    def _validate_targets(
        summary: DatasetSummary,
        suggestion: TrainingSuggestion,
        issues: list[ValidationIssue],
    ) -> None:
        declared = set(summary.target_columns)
        model_targets = set(suggestion.training.model_names_by_target)
        unknown = sorted(model_targets.difference(declared))
        for target in unknown:
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="unknown_target",
                    field=f"training.model_names_by_target.{target}",
                    message=f"Unknown target {target!r} in model suggestion.",
                )
            )
        for target in summary.target_columns:
            names = suggestion.training.model_names_by_target.get(target)
            if not names:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        code="missing_model_selection",
                        field=f"training.model_names_by_target.{target}",
                        message=(
                            f"At least one model must be suggested for target "
                            f"{target!r}."
                        ),
                    )
                )

        mapping_fields = {
            "model_params_by_target": (
                suggestion.training.model_params_by_target
            ),
            "base_model_params_by_target": (
                suggestion.training.base_model_params_by_target
            ),
        }
        for field_name, mapping in mapping_fields.items():
            for target in sorted(set(mapping).difference(declared)):
                issues.append(
                    ValidationIssue(
                        severity="error",
                        code="unknown_target",
                        field=f"training.{field_name}.{target}",
                        message=f"Unknown target {target!r} in {field_name}.",
                    )
                )

    @staticmethod
    def _validate_ensemble(
        summary: DatasetSummary,
        suggestion: TrainingSuggestion,
        issues: list[ValidationIssue],
    ) -> None:
        config = suggestion.training
        allowed_ensemble_types = {
            "アンサンブル",
            "スタッキング",
            "バギング",
            "ブースティング",
        }
        if not config.ensemble:
            for target, names in config.model_names_by_target.items():
                if len(names) > 1:
                    issues.append(
                        ValidationIssue(
                            severity="error",
                            code="multiple_models_without_ensemble",
                            field=(
                                f"training.model_names_by_target.{target}"
                            ),
                            message=(
                                "A non-ensemble training suggestion must select "
                                "exactly one model per target. Put additional "
                                "candidates in comparison.model_names."
                            ),
                        )
                    )
            if config.ens_type is not None:
                original = config.ens_type
                config.ens_type = None
                issues.append(
                    ValidationIssue(
                        severity="adjustment",
                        code="unused_ensemble_type",
                        field="training.ens_type",
                        message=(
                            "ens_type was removed because ensemble is false."
                        ),
                        original_value=original,
                        adjusted_value=None,
                    )
                )
            if config.base_model is not None:
                original = config.base_model
                config.base_model = None
                issues.append(
                    ValidationIssue(
                        severity="adjustment",
                        code="unused_base_model",
                        field="training.base_model",
                        message=(
                            "base_model was removed because ensemble is false."
                        ),
                        original_value=original,
                        adjusted_value=None,
                    )
                )
            return

        if config.ens_type not in allowed_ensemble_types:
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="invalid_ensemble_type",
                    field="training.ens_type",
                    message=(
                        "ensemble=true requires one of: "
                        f"{sorted(allowed_ensemble_types)}."
                    ),
                    original_value=config.ens_type,
                )
            )
        if config.ens_type == "スタッキング" and not config.base_model:
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="missing_stacking_meta_model",
                    field="training.base_model",
                    message=(
                        "Stacking requires an explicit final estimator in "
                        "base_model."
                    ),
                )
            )
        for target in summary.target_columns:
            names = config.model_names_by_target.get(target) or []
            params = config.model_params_by_target.get(target)
            if len(names) > 1 and params is not None:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        code="ambiguous_ensemble_parameters",
                        field=(
                            f"training.model_params_by_target.{target}"
                        ),
                        message=(
                            "The current training schema has one parameter "
                            "dictionary per target, so parameters for multiple "
                            "ensemble members are ambiguous. Use default member "
                            "parameters for now."
                        ),
                    )
                )

    def _validate_comparison(
        self,
        summary: DatasetSummary,
        suggestion: TrainingSuggestion,
        issues: list[ValidationIssue],
    ) -> None:
        config = suggestion.comparison
        if config.tuning and config.tune_best:
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="conflicting_tuning_modes",
                    field="comparison",
                    message="tuning and tune_best cannot both be true.",
                )
            )

        if config.method not in {"kfold", "loo"}:
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="unknown_cv_method",
                    field="comparison.method",
                    message=(
                        "Unsupported cross-validation method: "
                        f"{config.method!r}."
                    ),
                )
            )
            return

        if summary.n_rows < 2:
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="insufficient_rows",
                    field="comparison.method",
                    message=(
                        "At least two rows are required for cross-validation."
                    ),
                )
            )
            return

        if config.method == "loo":
            self._validate_trial_selection(
                config.tuning_trials,
                "comparison.tuning_trials",
                issues,
            )
            return

        maximum_splits = summary.n_rows
        classification_targets = [
            target
            for target, task in summary.tasks_by_target.items()
            if task == "classification"
        ]
        class_limits = [
            summary.minimum_class_count(target)
            for target in classification_targets
        ]
        class_limits = [limit for limit in class_limits if limit is not None]
        if class_limits:
            maximum_splits = min(maximum_splits, min(class_limits))

        if maximum_splits < 2:
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="insufficient_class_members",
                    field="comparison.n_splits",
                    message=(
                        "Each classification class needs at least two rows for "
                        "k-fold validation."
                    ),
                )
            )
        elif config.n_splits < 2:
            original = config.n_splits
            config.n_splits = 2
            issues.append(
                ValidationIssue(
                    severity="adjustment",
                    code="n_splits_too_small",
                    field="comparison.n_splits",
                    message=(
                        "n_splits was increased to the minimum valid value."
                    ),
                    original_value=original,
                    adjusted_value=2,
                )
            )
        elif config.n_splits > maximum_splits:
            original = config.n_splits
            config.n_splits = maximum_splits
            issues.append(
                ValidationIssue(
                    severity="adjustment",
                    code="n_splits_too_large",
                    field="comparison.n_splits",
                    message=(
                        "n_splits was reduced to fit the available rows or "
                        "class counts."
                    ),
                    original_value=original,
                    adjusted_value=maximum_splits,
                )
            )

        self._validate_trial_selection(
            config.tuning_trials,
            "comparison.tuning_trials",
            issues,
        )

    @staticmethod
    def _validate_trial_selection(
        value: int | Mapping[str, int],
        field: str,
        issues: list[ValidationIssue],
    ) -> None:
        values = [value] if isinstance(value, int) else list(value.values())
        if not values or any(
            not isinstance(item, int) or item < 1 for item in values
        ):
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="invalid_trial_count",
                    field=field,
                    message=(
                        "Every tuning trial count must be a positive integer."
                    ),
                    original_value=value,
                )
            )

    def _validate_preprocessing(
        self,
        summary: DatasetSummary,
        suggestion: TrainingSuggestion,
        issues: list[ValidationIssue],
    ) -> None:
        config = suggestion.training
        if config.poly_degree < 1:
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="invalid_polynomial_degree",
                    field="training.poly_degree",
                    message="poly_degree must be at least 1.",
                    original_value=config.poly_degree,
                )
            )
        elif config.poly:
            estimated = self._estimated_polynomial_features(
                summary.n_features,
                config.poly_degree,
                config.poly_interaction_only,
            )
            if estimated > self.max_polynomial_features:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        code="polynomial_feature_explosion",
                        field="training.poly",
                        message=(
                            "Polynomial preprocessing would create "
                            f"approximately {estimated} features, exceeding "
                            "the configured limit of "
                            f"{self.max_polynomial_features}."
                        ),
                        original_value=estimated,
                    )
                )

        if config.decomposition:
            maximum_components = min(summary.n_rows, summary.n_features)
            if maximum_components < 1:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        code="invalid_decomposition",
                        field="training.decomposition",
                        message=(
                            "Decomposition requires at least one row and one "
                            "feature."
                        ),
                    )
                )
            elif config.dec_n_components < 1:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        code="invalid_component_count",
                        field="training.dec_n_components",
                        message="dec_n_components must be at least 1.",
                    )
                )
            elif config.dec_n_components > maximum_components:
                original = config.dec_n_components
                config.dec_n_components = maximum_components
                issues.append(
                    ValidationIssue(
                        severity="adjustment",
                        code="component_count_too_large",
                        field="training.dec_n_components",
                        message=(
                            "dec_n_components was reduced to fit the dataset "
                            "dimensions."
                        ),
                        original_value=original,
                        adjusted_value=maximum_components,
                    )
                )

            if config.decomposition_method == "NMF":
                negative_columns = [
                    name
                    for name in summary.numeric_columns
                    if (summary.column(name).numeric_min or 0.0) < 0.0
                ]
                if negative_columns:
                    issues.append(
                        ValidationIssue(
                            severity="error",
                            code="nmf_requires_nonnegative_data",
                            field="training.decomposition_method",
                            message=(
                                "NMF cannot be used because negative values "
                                f"exist in {negative_columns}."
                            ),
                        )
                    )

        tasks = set(summary.tasks_by_target.values())
        if config.sampling_method is not None and tasks == {"regression"}:
            original = config.sampling_method
            config.sampling_method = None
            issues.append(
                ValidationIssue(
                    severity="adjustment",
                    code="sampling_ignored_for_regression",
                    field="training.sampling_method",
                    message=(
                        "Classification sampling was removed from a "
                        "regression-only workflow."
                    ),
                    original_value=original,
                    adjusted_value=None,
                )
            )
        if config.sampling_method == "SMOTE":
            small_targets = [
                target
                for target, task in summary.tasks_by_target.items()
                if task == "classification"
                and (summary.minimum_class_count(target) or 0) < 6
            ]
            if small_targets:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        code="insufficient_samples_for_smote",
                        field="training.sampling_method",
                        message=(
                            "SMOTE with its usual neighbor count requires at "
                            "least six rows in every class; insufficient "
                            f"targets: {small_targets}."
                        ),
                    )
                )

    def _validate_models(
        self,
        summary: DatasetSummary,
        suggestion: TrainingSuggestion,
        issues: list[ValidationIssue],
    ) -> None:
        min_train_samples = self._minimum_training_rows(summary, suggestion)
        for target, names in (
            suggestion.training.model_names_by_target.items()
        ):
            task = summary.tasks_by_target.get(target)
            if task is None:
                continue
            if len(names) != len(set(names)):
                issues.append(
                    ValidationIssue(
                        severity="error",
                        code="duplicate_model_name",
                        field=f"training.model_names_by_target.{target}",
                        message=(
                            "Duplicate model names were suggested for target "
                            f"{target!r}."
                        ),
                    )
                )
            for name in names:
                spec = self.registry.find(task, name)
                if spec is None:
                    issues.append(
                        ValidationIssue(
                            severity="error",
                            code="unknown_model",
                            field=(
                                f"training.model_names_by_target.{target}"
                            ),
                            message=(
                                f"Model {name!r} is not registered for task "
                                f"{task!r}."
                            ),
                        )
                    )
                    continue
                if summary.n_rows < spec.recommended_min_samples:
                    issues.append(
                        ValidationIssue(
                            severity="warning",
                            code="small_dataset_for_model",
                            field=(
                                f"training.model_names_by_target.{target}"
                            ),
                            message=(
                                f"Model {name!r} is usually safer with at "
                                f"least {spec.recommended_min_samples} rows."
                            ),
                        )
                    )

                params = (
                    suggestion.training.model_params_by_target.get(target)
                )
                if params is None:
                    continue
                self._validate_model_params(
                    target=target,
                    spec=spec,
                    params=params,
                    min_train_samples=min_train_samples,
                    n_features=summary.n_features,
                    issues=issues,
                )

        self._validate_comparison_models(summary, suggestion, issues)

    def _validate_comparison_models(
        self,
        summary: DatasetSummary,
        suggestion: TrainingSuggestion,
        issues: list[ValidationIssue],
    ) -> None:
        selection = suggestion.comparison.model_names
        if selection is None:
            return
        selections = (
            {target: selection for target in summary.target_columns}
            if isinstance(selection, list)
            else dict(selection)
        )
        for target, names in selections.items():
            if len(names) != len(set(names)):
                issues.append(
                    ValidationIssue(
                        severity="error",
                        code="duplicate_model_name",
                        field=f"comparison.model_names.{target}",
                        message=(
                            "Duplicate comparison models were suggested for "
                            f"target {target!r}."
                        ),
                    )
                )
            task = summary.tasks_by_target.get(target)
            if task is None:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        code="unknown_target",
                        field=f"comparison.model_names.{target}",
                        message=f"Unknown comparison target {target!r}.",
                    )
                )
                continue
            for name in names:
                if self.registry.find(task, name) is None:
                    issues.append(
                        ValidationIssue(
                            severity="error",
                            code="unknown_model",
                            field=f"comparison.model_names.{target}",
                            message=(
                                f"Model {name!r} is not registered for task "
                                f"{task!r}."
                            ),
                        )
                    )

    @staticmethod
    def _validate_model_params(
        *,
        target: str,
        spec: ModelSpec,
        params: dict[str, Any],
        min_train_samples: int,
        n_features: int,
        issues: list[ValidationIssue],
    ) -> None:
        unknown_params = sorted(
            name for name in params if not spec.accepts_parameter(name)
        )
        if unknown_params:
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="unknown_model_parameter",
                    field=f"training.model_params_by_target.{target}",
                    message=(
                        f"Unsupported parameters for {spec.name!r}: "
                        f"{unknown_params}."
                    ),
                )
            )

        if spec.family == "neighbors" and "n_neighbors" in params:
            value = params["n_neighbors"]
            if isinstance(value, int) and value > min_train_samples:
                params["n_neighbors"] = max(min_train_samples, 1)
                issues.append(
                    ValidationIssue(
                        severity="adjustment",
                        code="n_neighbors_too_large",
                        field=(
                            "training.model_params_by_target."
                            f"{target}.n_neighbors"
                        ),
                        message=(
                            "n_neighbors was reduced to fit the smallest CV "
                            "training fold."
                        ),
                        original_value=value,
                        adjusted_value=params["n_neighbors"],
                    )
                )
        if spec.family == "pls" and "n_components" in params:
            value = params["n_components"]
            maximum = max(min(min_train_samples, n_features), 1)
            if isinstance(value, int) and value > maximum:
                params["n_components"] = maximum
                issues.append(
                    ValidationIssue(
                        severity="adjustment",
                        code="pls_components_too_large",
                        field=(
                            "training.model_params_by_target."
                            f"{target}.n_components"
                        ),
                        message=(
                            "PLS components were reduced to fit the data "
                            "dimensions."
                        ),
                        original_value=value,
                        adjusted_value=maximum,
                    )
                )

    @staticmethod
    def _minimum_training_rows(
        summary: DatasetSummary,
        suggestion: TrainingSuggestion,
    ) -> int:
        if suggestion.comparison.method == "loo":
            return max(summary.n_rows - 1, 1)
        splits = max(suggestion.comparison.n_splits, 2)
        return max(
            summary.n_rows - math.ceil(summary.n_rows / splits),
            1,
        )

    @staticmethod
    def _estimated_polynomial_features(
        n_features: int,
        degree: int,
        interaction_only: bool,
    ) -> int:
        if n_features <= 0:
            return 0
        if interaction_only:
            maximum_degree = min(degree, n_features)
            return sum(
                math.comb(n_features, value)
                for value in range(1, maximum_degree + 1)
            )
        return math.comb(n_features + degree, degree) - 1
