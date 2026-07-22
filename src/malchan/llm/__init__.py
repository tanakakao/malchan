"""LLM-assisted configuration planning for malchan.

The public API separates provider calls from dataset summarization and semantic
validation. OpenAI and Gemini produce reviewable settings only; malchan remains
responsible for validating, fitting, comparing, and tuning models.
"""

from .client import (
    BaseLLMClient,
    GeminiClient,
    LLMResponse,
    OpenAIClient,
    make_llm_client,
)
from .configs import (
    LLMConfig,
    LLMContextConfig,
    coerce_llm_config,
    coerce_llm_context,
)
from .parser import (
    parse_json_payload,
    response_model_name,
    schema_from_response_model,
    validate_structured_payload,
)
from .planner import (
    TrainingPlanResult,
    build_training_planner_prompt,
    plan_training_configuration,
    training_suggestion_json_schema,
)
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
    "BaseLLMClient",
    "ColumnSummary",
    "ComparisonConfigSuggestion",
    "DatasetSummary",
    "GeminiClient",
    "LLMConfig",
    "LLMContextConfig",
    "LLMResponse",
    "ModelSpec",
    "ModelSpecRegistry",
    "OpenAIClient",
    "SuggestionValidationResult",
    "SuggestionValidator",
    "TrainingConfigSuggestion",
    "TrainingPlanResult",
    "TrainingSuggestion",
    "ValidationIssue",
    "build_runtime_model_registry",
    "build_training_planner_prompt",
    "coerce_llm_config",
    "coerce_llm_context",
    "make_llm_client",
    "parse_json_payload",
    "plan_training_configuration",
    "response_model_name",
    "schema_from_response_model",
    "training_suggestion_from_mapping",
    "training_suggestion_json_schema",
    "validate_structured_payload",
]
