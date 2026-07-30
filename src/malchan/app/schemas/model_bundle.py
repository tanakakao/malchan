"""Schemas for downloading and restoring model artifacts."""

from typing import Any

from pydantic import BaseModel, Field

from .models import ModelInfo


class ModelBundleImportResponse(BaseModel):
    """Metadata returned after a downloaded model is restored in memory."""

    model: ModelInfo
    original_model_id: str
    num_cols: list[str] = Field(default_factory=list)
    cat_cols: list[str] = Field(default_factory=list)
    smiles_cols: list[str] = Field(default_factory=list)
    comp_cols: list[str] = Field(default_factory=list)
    training_rows: list[dict[str, Any]] = Field(default_factory=list)


__all__ = ["ModelBundleImportResponse"]
