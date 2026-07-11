"""Schemas for Optuna-based inverse analysis endpoints."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class InverseObjective(BaseModel):
    """One target optimized by direction or toward a requested value/class."""

    model_config = ConfigDict(extra="forbid")

    target: str = Field(min_length=1)
    direction: Literal["min", "max"] | None = None
    target_value: Any | None = None

    @model_validator(mode="after")
    def validate_objective(self) -> "InverseObjective":
        """Require exactly one objective definition."""

        has_direction = self.direction is not None
        has_target = self.target_value is not None
        if has_direction == has_target:
            raise ValueError(
                "Specify exactly one of direction or target_value for each objective."
            )
        return self


class NumericSearchRange(BaseModel):
    """Search range for one numeric feature."""

    model_config = ConfigDict(extra="forbid")

    min: float
    max: float
    dtype: Literal["float", "int"] | None = None
    step: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_range(self) -> "NumericSearchRange":
        """Validate range ordering and integer-specific values."""

        if self.max <= self.min:
            raise ValueError("max must be greater than min.")
        if self.dtype == "int":
            if not float(self.min).is_integer() or not float(self.max).is_integer():
                raise ValueError("Integer search ranges require integral min and max.")
            if self.step is not None and not float(self.step).is_integer():
                raise ValueError("Integer search ranges require an integral step.")
        return self


class SumConstraint(BaseModel):
    """Equality constraint requiring selected numeric features to sum to a value."""

    model_config = ConfigDict(extra="forbid")

    columns: list[str] = Field(min_length=1)
    value: float

    @model_validator(mode="after")
    def validate_columns(self) -> "SumConstraint":
        """Reject duplicate columns."""

        if len(self.columns) != len(set(self.columns)):
            raise ValueError("Sum-constraint columns must not contain duplicates.")
        return self


class InverseAnalysisRequest(BaseModel):
    """Request for candidate search against a registered model."""

    model_config = ConfigDict(extra="forbid")

    objectives: list[InverseObjective] = Field(min_length=1)
    sampler_type: Literal[
        "TPE",
        "MOTPE",
        "CmaEs",
        "GP",
        "QMS",
        "NSGAII",
        "NSGAIII",
    ] = "TPE"
    bounds: dict[str, NumericSearchRange] = Field(default_factory=dict)
    categories: dict[str, list[Any]] = Field(default_factory=dict)
    fixed_values: dict[str, Any] = Field(default_factory=dict)
    sum_constraint: SumConstraint | None = None
    trials: int = Field(default=250, ge=1)
    n_candidates: int = Field(default=10, ge=1)

    @model_validator(mode="after")
    def validate_request(self) -> "InverseAnalysisRequest":
        """Validate objective uniqueness and candidate count."""

        targets = [objective.target for objective in self.objectives]
        if len(targets) != len(set(targets)):
            raise ValueError("Inverse-analysis objectives must use unique targets.")
        if self.n_candidates > self.trials:
            raise ValueError("n_candidates must not exceed trials.")
        empty_categories = [
            column for column, values in self.categories.items() if not values
        ]
        if empty_categories:
            raise ValueError(
                f"Category candidate lists must not be empty: {empty_categories}"
            )
        return self


class InverseAnalysisResponse(BaseModel):
    """Ranked candidates and summary metadata from inverse analysis."""

    model_id: str
    objectives: list[InverseObjective]
    candidates: list[dict[str, Any]]
    n_trials: int
    n_completed_trials: int
    pareto_size: int
