"""Pydantic schemas for model training and prediction endpoints."""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TrainModelRequest(BaseModel):
    """Request body used to train one output model."""

    model_config = ConfigDict(extra="forbid")

    data: list[dict[str, Any]] = Field(min_length=2)
    target_col: str = Field(min_length=1)
    task: Literal["regression", "classification"]
    num_cols: list[str] = Field(default_factory=list)
    cat_cols: list[str] = Field(default_factory=list)
    model_names: list[str] = Field(min_length=1)
    smiles_cols: list[str] = Field(default_factory=list)
    fingerprints: list[str] = Field(default_factory=list)
    comp_cols: list[str] = Field(default_factory=list)
    comp_method: str | None = None
    comp_feats: list[str] = Field(default_factory=list)
    impute: bool = False
    tuning: bool = False
    ensemble: bool = False
    ens_type: str | None = None
    base_model: str | None = None
    model_params: dict[str, Any] | None = None
    base_model_param: dict[str, Any] | None = None
    num_impute_type: str | None = None
    num_scale_type: str | None = None
    cat_impute: bool = False
    poly: bool = False
    poly_degree: int = Field(default=1, ge=1)
    poly_interaction_only: bool = True
    decomposition: bool = False
    decomposition_method: str = "PCA"
    dec_n_components: int = Field(default=2, ge=1)
    sampling_method: str | None = None

    @property
    def feature_columns(self) -> list[str]:
        """Return all raw input columns in model input order."""

        return [
            *self.num_cols,
            *self.cat_cols,
            *self.smiles_cols,
            *self.comp_cols,
        ]

    @model_validator(mode="after")
    def validate_training_columns(self) -> "TrainModelRequest":
        """Validate feature declarations against every input row."""

        feature_columns = self.feature_columns
        if not feature_columns:
            raise ValueError("At least one feature column must be specified.")
        if len(feature_columns) != len(set(feature_columns)):
            raise ValueError("Feature columns must not contain duplicates.")
        if self.target_col in feature_columns:
            raise ValueError(
                "target_col must not also be declared as a feature column."
            )

        required = set(feature_columns) | {self.target_col}
        for row_index, row in enumerate(self.data):
            missing = sorted(required.difference(row))
            if missing:
                raise ValueError(
                    f"data[{row_index}] is missing required columns: {missing}"
                )
        return self


class PredictRequest(BaseModel):
    """Request body used to run inference with a registered model."""

    model_config = ConfigDict(extra="forbid")

    data: list[dict[str, Any]] = Field(min_length=1)
    proba: bool = False
    decode_labels: bool = True


class HealthResponse(BaseModel):
    """Health-check response."""

    status: str
    service: str
    version: str


class ModelInfo(BaseModel):
    """Metadata for one in-memory model."""

    model_id: str
    target_col: str
    task: Literal["regression", "classification"]
    feature_columns: list[str]
    model_names: list[str]
    created_at: datetime


class ModelListResponse(BaseModel):
    """Response containing all currently registered models."""

    models: list[ModelInfo]


class PredictionResponse(BaseModel):
    """Prediction records returned by a model."""

    model_id: str
    predictions: list[dict[str, Any]]
