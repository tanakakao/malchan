"""Schemas for model comparison and best-model tuning endpoints."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


ModelNameSelection = list[str] | dict[str, list[str]]
MetricSelection = str | dict[str, str]
TrialSelection = int | dict[str, int]


class CompareModelsRequest(BaseModel):
    """Request for comparing candidate estimators of a registered model."""

    model_config = ConfigDict(extra="forbid")

    model_names: ModelNameSelection | None = None
    model_params: dict[str, Any] | None = None
    method: Literal["kfold", "loo"] = "kfold"
    n_splits: int = Field(default=5, ge=2)
    metric: MetricSelection | None = None
    tuning: bool = False
    tune_best: bool = False
    tuning_trials: TrialSelection = 30
    tuning_verbose: int = Field(default=0, ge=0)
    continue_on_error: bool = True
    activate_best: bool = False

    @model_validator(mode="after")
    def validate_comparison(self) -> "CompareModelsRequest":
        """Validate candidate, metric, and tuning settings."""

        if self.tuning and self.tune_best:
            raise ValueError(
                "Use either tuning=true for every candidate or tune_best=true "
                "for two-stage tuning, not both."
            )
        self._validate_model_names(self.model_names)
        self._validate_metric(self.metric)
        self._validate_trials(self.tuning_trials, "tuning_trials")
        return self

    @staticmethod
    def _validate_model_names(value: ModelNameSelection | None) -> None:
        """Reject empty or duplicate candidate selections."""

        if value is None:
            return
        selections = {"__all__": value} if isinstance(value, list) else value
        if not selections:
            raise ValueError("model_names mapping must not be empty.")
        for target, names in selections.items():
            if not names or any(not name.strip() for name in names):
                raise ValueError(
                    f"At least one non-empty model name is required for {target!r}."
                )
            if len(names) != len(set(names)):
                raise ValueError(
                    f"model_names for {target!r} must not contain duplicates."
                )

    @staticmethod
    def _validate_metric(value: MetricSelection | None) -> None:
        """Reject blank ranking metrics."""

        if value is None:
            return
        metrics = [value] if isinstance(value, str) else list(value.values())
        if not metrics or any(not metric.strip() for metric in metrics):
            raise ValueError("metric values must not be empty.")

    @staticmethod
    def _validate_trials(value: TrialSelection, field_name: str) -> None:
        """Require positive Optuna trial counts."""

        values = [value] if isinstance(value, int) else list(value.values())
        if not values or any(trials < 1 for trials in values):
            raise ValueError(f"{field_name} values must be at least 1.")


class TuneBestModelRequest(BaseModel):
    """Request for tuning selected models from the latest comparison result."""

    model_config = ConfigDict(extra="forbid")

    targets: list[str] = Field(default_factory=list)
    n_trials: TrialSelection = 30
    verbose: int = Field(default=0, ge=0)
    evaluate: bool = True
    activate_best: bool = False

    @model_validator(mode="after")
    def validate_tuning(self) -> "TuneBestModelRequest":
        """Validate target names and per-target trial counts."""

        if len(self.targets) != len(set(self.targets)):
            raise ValueError("targets must not contain duplicates.")
        if any(not target.strip() for target in self.targets):
            raise ValueError("targets must not contain empty names.")
        CompareModelsRequest._validate_trials(self.n_trials, "n_trials")
        return self


class TargetComparisonResponse(BaseModel):
    """Comparison and optional tuning result for one target column."""

    target: str
    metric: str
    higher_is_better: bool
    ranking: list[dict[str, Any]]
    failures: dict[str, str]
    best_model_name: str | None = None
    best_params: Any = None
    best_is_tuned: bool = False
    best_cv_scores: dict[str, list[dict[str, Any]]] | None = None


class ModelComparisonResponse(BaseModel):
    """Latest comparison state for a registered model."""

    model_id: str
    targets: dict[str, TargetComparisonResponse]
