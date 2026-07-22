"""Typed, provider-independent configuration suggestions for malchan."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

CrossValidationMethod = Literal["kfold", "loo"]
TaskType = Literal["regression", "classification"]
ModelNameSelection = list[str] | dict[str, list[str]]
MetricSelection = str | dict[str, str]
TrialSelection = int | dict[str, int]


@dataclass(slots=True)
class TrainingConfigSuggestion:
    """Suggested arguments corresponding to malchan model training options."""

    model_names_by_target: dict[str, list[str]] = field(default_factory=dict)
    model_params_by_target: dict[str, dict[str, Any] | None] = field(default_factory=dict)
    base_model_params_by_target: dict[str, dict[str, Any] | None] = field(default_factory=dict)
    impute: bool = False
    tuning: bool = False
    ensemble: bool = False
    ens_type: str | None = None
    base_model: str | None = None
    num_impute_type: str | None = None
    num_scale_type: str | None = None
    cat_impute: bool = False
    poly: bool = False
    poly_degree: int = 1
    poly_interaction_only: bool = True
    decomposition: bool = False
    decomposition_method: str = "PCA"
    dec_n_components: int = 2
    sampling_method: str | None = None
    compute_xai: bool = True


@dataclass(slots=True)
class ComparisonConfigSuggestion:
    """Suggested arguments corresponding to ``compare()`` and its API schema."""

    model_names: ModelNameSelection | None = None
    model_params: dict[str, Any] | None = None
    method: CrossValidationMethod = "kfold"
    n_splits: int = 5
    metric: MetricSelection | None = None
    tuning: bool = False
    tune_best: bool = True
    tuning_trials: TrialSelection = 30
    tuning_verbose: int = 0
    continue_on_error: bool = True
    activate_best: bool = False


@dataclass(slots=True)
class TrainingSuggestion:
    """Complete review-first suggestion returned by a future LLM planner."""

    training: TrainingConfigSuggestion = field(default_factory=TrainingConfigSuggestion)
    comparison: ComparisonConfigSuggestion = field(default_factory=ComparisonConfigSuggestion)
    reasoning_summary: str = ""
    warnings: list[str] = field(default_factory=list)
    confidence: float | None = None
    raw_plan: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly dictionary for APIs, logs, and tests."""

        return asdict(self)


def training_suggestion_from_mapping(value: Mapping[str, Any]) -> TrainingSuggestion:
    """Convert a serialized planner response to typed suggestion objects."""

    data = dict(value)
    training = data.get("training") or data.get("training_config") or {}
    comparison = data.get("comparison") or data.get("comparison_config") or {}
    confidence = data.get("confidence")
    if confidence is not None:
        confidence = float(confidence)
    return TrainingSuggestion(
        training=(
            training
            if isinstance(training, TrainingConfigSuggestion)
            else TrainingConfigSuggestion(**dict(training))
        ),
        comparison=(
            comparison
            if isinstance(comparison, ComparisonConfigSuggestion)
            else ComparisonConfigSuggestion(**dict(comparison))
        ),
        reasoning_summary=str(data.get("reasoning_summary") or ""),
        warnings=[str(item) for item in data.get("warnings") or []],
        confidence=confidence,
        raw_plan=dict(data.get("raw_plan") or {}),
    )
