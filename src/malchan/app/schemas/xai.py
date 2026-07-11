"""Schemas for cached SHAP, importance, and partial-dependence endpoints."""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

XaiStatus = Literal[
    "not_requested",
    "computing",
    "ready",
    "partial",
    "failed",
    "unavailable",
]
XaiImportanceMethod = Literal["model", "pfi", "shap"]


class RecomputeXaiRequest(BaseModel):
    """Request for explicitly rebuilding cached XAI data."""

    model_config = ConfigDict(extra="forbid")

    targets: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_targets(self) -> "RecomputeXaiRequest":
        """Reject duplicate or empty target names."""

        if len(self.targets) != len(set(self.targets)):
            raise ValueError("targets must not contain duplicates.")
        if any(not target.strip() for target in self.targets):
            raise ValueError("targets must not contain empty names.")
        return self


class XaiTargetSummary(BaseModel):
    """Availability metadata for one target's cached explanations."""

    target: str
    status: XaiStatus
    computed_at: datetime | None = None
    error: str | None = None
    features: list[str] = Field(default_factory=list)
    importance_methods: list[XaiImportanceMethod] = Field(default_factory=list)
    shap_features: list[str] = Field(default_factory=list)
    pdp_features: list[str] = Field(default_factory=list)


class XaiSummaryResponse(BaseModel):
    """Cached XAI state for a registered model."""

    model_id: str
    status: XaiStatus
    targets: dict[str, XaiTargetSummary]


class XaiImportanceItem(BaseModel):
    """One feature-importance value."""

    feature: str
    value: float


class XaiImportanceResponse(BaseModel):
    """Cached feature importance for one target and method."""

    model_id: str
    target: str
    method: XaiImportanceMethod
    combined: bool
    items: list[XaiImportanceItem]


class XaiShapResponse(BaseModel):
    """Cached SHAP scatter records for one raw feature."""

    model_id: str
    target: str
    feature: str
    value_columns: list[str]
    records: list[dict[str, Any]]


class XaiPdpSeries(BaseModel):
    """One PDP output series and optional cached ICE curves."""

    name: str
    pd_values: list[float]
    ice_values: list[list[float]] | None = None


class XaiPdpResponse(BaseModel):
    """Cached partial-dependence data for one target and feature."""

    model_id: str
    target: str
    feature: str
    x_values: list[Any]
    series: list[XaiPdpSeries]
