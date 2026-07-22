"""Provider-independent foundations for future LLM-assisted workflows.

This initial module deliberately performs no network calls.  It exposes model
capability metadata, local dataset summarization, typed configuration
suggestions, and semantic validation that OpenAI or Gemini adapters can use in a
later phase.
"""

from .registry import ModelSpec, ModelSpecRegistry, build_runtime_model_registry
from .suggestions import (
    ComparisonConfigSuggestion,
    TrainingConfigSuggestion,
    TrainingSuggestion,
    training_suggestion_from_mapping,
)
from .summary import ColumnSummary, DatasetSummary
from .validation import (
    SuggestionValidationResult,
    SuggestionValidator,
    ValidationIssue,
)

__all__ = [
    "ColumnSummary",
    "ComparisonConfigSuggestion",
    "DatasetSummary",
    "ModelSpec",
    "ModelSpecRegistry",
    "SuggestionValidationResult",
    "SuggestionValidator",
    "TrainingConfigSuggestion",
    "TrainingSuggestion",
    "ValidationIssue",
    "build_runtime_model_registry",
    "training_suggestion_from_mapping",
]
