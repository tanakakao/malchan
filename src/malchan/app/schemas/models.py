"""Pydantic schemas for model training and prediction endpoints."""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

TaskType = Literal["regression", "classification"]
EnsembleType = Literal["アンサンブル", "スタッキング", "バギング", "ブースティング"]
ModelParameters = dict[str, Any] | list[dict[str, Any]]


class TrainModelRequest(BaseModel):
    """Request body used to train a single-output or multi-output model."""

    model_config = ConfigDict(extra="forbid")

    data: list[dict[str, Any]] = Field(min_length=2)
    target_col: str | None = Field(default=None, min_length=1)
    task: TaskType | None = None
    target_cols: list[str] = Field(default_factory=list)
    tasks: list[TaskType] = Field(default_factory=list)
    num_cols: list[str] = Field(default_factory=list)
    cat_cols: list[str] = Field(default_factory=list)
    model_names: list[str] = Field(default_factory=list)
    model_names_by_target: dict[str, list[str]] = Field(default_factory=dict)
    smiles_cols: list[str] = Field(default_factory=list)
    fingerprints: list[str] = Field(default_factory=list)
    comp_cols: list[str] = Field(default_factory=list)
    comp_method: str | None = None
    comp_feats: list[str] = Field(default_factory=list)
    impute: bool = False
    tuning: bool = False
    ensemble: bool = False
    ens_type: EnsembleType | None = None
    base_model: str | None = None
    model_params: ModelParameters | None = None
    model_params_by_target: dict[str, ModelParameters | None] = Field(
        default_factory=dict
    )
    base_model_param: dict[str, Any] | None = None
    base_model_params_by_target: dict[str, dict[str, Any] | None] = Field(default_factory=dict)
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
    compute_xai: bool = True

    @property
    def feature_columns(self) -> list[str]:
        """Return all raw input columns in model input order."""

        return [
            *self.num_cols,
            *self.cat_cols,
            *self.smiles_cols,
            *self.comp_cols,
        ]

    @property
    def resolved_target_cols(self) -> list[str]:
        """Return target columns for either request style."""

        if self.target_cols:
            return list(self.target_cols)
        if self.target_col is None:
            raise RuntimeError("Training targets have not been validated.")
        return [self.target_col]

    @property
    def resolved_tasks(self) -> list[TaskType]:
        """Return task names aligned with :attr:`resolved_target_cols`."""

        if self.tasks:
            return list(self.tasks)
        if self.task is None:
            raise RuntimeError("Training tasks have not been validated.")
        return [self.task]

    @property
    def is_multi_output(self) -> bool:
        """Return whether more than one target is requested."""

        return len(self.resolved_target_cols) > 1

    @property
    def model_names_for_targets(self) -> dict[str, list[str]]:
        """Return model-name lists keyed by target."""

        return {
            target: list(self.model_names_by_target.get(target, self.model_names))
            for target in self.resolved_target_cols
        }

    @property
    def resolved_model_params(self) -> list[ModelParameters | None]:
        """Return predictor parameter settings aligned with target order.

        A single parameter dictionary is normalized to a one-element list for
        ensemble training because the predictor builder expects one dictionary
        per member model.
        """

        resolved: list[ModelParameters | None] = []
        for target in self.resolved_target_cols:
            params = self.model_params_by_target.get(target, self.model_params)
            if self.ensemble and isinstance(params, dict):
                params = [params]
            resolved.append(params)
        return resolved

    @property
    def resolved_base_model_params(self) -> list[dict[str, Any] | None]:
        """Return base-model parameter dictionaries aligned with target order."""

        return [
            self.base_model_params_by_target.get(target, self.base_model_param)
            for target in self.resolved_target_cols
        ]

    @model_validator(mode="after")
    def validate_training_columns(self) -> "TrainModelRequest":
        """Validate output, ensemble, and column settings against every row."""

        uses_single_fields = self.target_col is not None or self.task is not None
        uses_multi_fields = bool(self.target_cols) or bool(self.tasks)
        if uses_single_fields and uses_multi_fields:
            raise ValueError("Use either target_col/task or target_cols/tasks, not both.")
        if not uses_single_fields and not uses_multi_fields:
            raise ValueError("Specify target_col/task or target_cols/tasks.")
        if uses_single_fields and (self.target_col is None or self.task is None):
            raise ValueError("target_col and task must be specified together.")
        if uses_multi_fields:
            if not self.target_cols or not self.tasks:
                raise ValueError("target_cols and tasks must be specified together.")
            if len(self.target_cols) != len(self.tasks):
                raise ValueError("target_cols and tasks must have the same length.")

        target_cols = self.resolved_target_cols
        if any(not target.strip() for target in target_cols):
            raise ValueError("Target column names must not be empty.")
        if len(target_cols) != len(set(target_cols)):
            raise ValueError("Target columns must not contain duplicates.")

        feature_columns = self.feature_columns
        if not feature_columns:
            raise ValueError("At least one feature column must be specified.")
        if len(feature_columns) != len(set(feature_columns)):
            raise ValueError("Feature columns must not contain duplicates.")
        overlapping = sorted(set(target_cols).intersection(feature_columns))
        if overlapping:
            raise ValueError(
                f"Target columns must not also be feature columns: {overlapping}"
            )

        target_set = set(target_cols)
        target_mappings = (
            ("model_names_by_target", self.model_names_by_target),
            ("model_params_by_target", self.model_params_by_target),
            ("base_model_params_by_target", self.base_model_params_by_target),
        )
        for field_name, mapping in target_mappings:
            unknown = sorted(set(mapping).difference(target_set))
            if unknown:
                raise ValueError(f"{field_name} contains unknown targets: {unknown}")

        names_by_target = self.model_names_for_targets
        for target, names in names_by_target.items():
            if not names or any(not name.strip() for name in names):
                raise ValueError(
                    f"At least one model name is required for target {target!r}."
                )

        if self.ensemble:
            if self.ens_type is None:
                raise ValueError("ens_type is required when ensemble is true.")
            if self.ens_type == "スタッキング" and (
                self.base_model is None or not self.base_model.strip()
            ):
                raise ValueError(
                    "base_model is required when ens_type is 'スタッキング'."
                )

            for target, names in names_by_target.items():
                if self.ens_type in {"アンサンブル", "スタッキング"} and len(names) < 2:
                    raise ValueError(
                        f"{self.ens_type} requires at least two model names "
                        f"for target {target!r}."
                    )

                params = self.model_params_by_target.get(target, self.model_params)
                if isinstance(params, dict) and len(names) > 1:
                    raise ValueError(
                        "Ensemble model_params must be a list aligned with "
                        f"model names for target {target!r}."
                    )
                if isinstance(params, list) and len(params) != len(names):
                    raise ValueError(
                        "Ensemble model_params must have the same length as "
                        f"model names for target {target!r}."
                    )

        required = set(feature_columns) | target_set
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
    """Metadata for one in-memory single-output or multi-output model."""

    model_id: str
    target_cols: list[str]
    tasks: list[TaskType]
    feature_columns: list[str]
    model_names_by_target: dict[str, list[str]]
    created_at: datetime
    target_col: str | None = None
    task: TaskType | None = None
    model_names: list[str] = Field(default_factory=list)
    xai_status: str = "not_requested"


class ModelListResponse(BaseModel):
    """Response containing all currently registered models."""

    models: list[ModelInfo]


class PredictionResponse(BaseModel):
    """Prediction records returned by a model."""

    model_id: str
    predictions: list[dict[str, Any]]
