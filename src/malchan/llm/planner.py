"""Review-first LLM planner for malchan training and comparison settings."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .client import BaseLLMClient, LLMResponse, make_llm_client
from .configs import LLMContextConfig, coerce_llm_context
from .parser import parse_json_payload
from .registry import ModelSpecRegistry, build_runtime_model_registry
from .suggestions import TrainingSuggestion, training_suggestion_from_mapping
from .summary import DatasetSummary
from .validation import SuggestionValidationResult, SuggestionValidator


def _to_jsonable(value: Any) -> Any:
    """Convert optional context and existing settings to prompt-safe values."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_to_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return _to_jsonable(value.tolist())
    return repr(value)


def training_suggestion_json_schema() -> dict[str, Any]:
    """Return the provider schema used for training-setting suggestions.

    Model names and estimator parameter keys depend on the runtime registry, so
    the provider schema focuses on structure while :class:`SuggestionValidator`
    enforces malchan-specific semantics after generation.
    """

    nullable_string = {"type": ["string", "null"]}
    nullable_object = {
        "type": ["object", "null"],
        "additionalProperties": True,
    }
    model_name_selection = {
        "anyOf": [
            {"type": "array", "items": {"type": "string"}},
            {
                "type": "object",
                "additionalProperties": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            {"type": "null"},
        ]
    }
    metric_selection = {
        "anyOf": [
            {"type": "string"},
            {"type": "object", "additionalProperties": {"type": "string"}},
            {"type": "null"},
        ]
    }
    trial_selection = {
        "anyOf": [
            {"type": "integer", "minimum": 1},
            {
                "type": "object",
                "additionalProperties": {"type": "integer", "minimum": 1},
            },
        ]
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "training": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "model_names_by_target": {
                        "type": "object",
                        "additionalProperties": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 1,
                        },
                    },
                    "model_params_by_target": {
                        "type": "object",
                        "additionalProperties": nullable_object,
                    },
                    "base_model_params_by_target": {
                        "type": "object",
                        "additionalProperties": nullable_object,
                    },
                    "impute": {"type": "boolean"},
                    "tuning": {"type": "boolean"},
                    "ensemble": {"type": "boolean"},
                    "ens_type": nullable_string,
                    "base_model": nullable_string,
                    "num_impute_type": nullable_string,
                    "num_scale_type": nullable_string,
                    "cat_impute": {"type": "boolean"},
                    "poly": {"type": "boolean"},
                    "poly_degree": {"type": "integer", "minimum": 1},
                    "poly_interaction_only": {"type": "boolean"},
                    "decomposition": {"type": "boolean"},
                    "decomposition_method": {"type": "string"},
                    "dec_n_components": {"type": "integer", "minimum": 1},
                    "sampling_method": nullable_string,
                    "compute_xai": {"type": "boolean"},
                },
                "required": ["model_names_by_target"],
            },
            "comparison": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "model_names": model_name_selection,
                    "model_params": nullable_object,
                    "method": {"type": "string", "enum": ["kfold", "loo"]},
                    "n_splits": {"type": "integer", "minimum": 2},
                    "metric": metric_selection,
                    "tuning": {"type": "boolean"},
                    "tune_best": {"type": "boolean"},
                    "tuning_trials": trial_selection,
                    "tuning_verbose": {"type": "integer", "minimum": 0},
                    "continue_on_error": {"type": "boolean"},
                    "activate_best": {"type": "boolean"},
                },
            },
            "reasoning_summary": {"type": "string"},
            "warnings": {"type": "array", "items": {"type": "string"}},
            "confidence": {
                "type": ["number", "null"],
                "minimum": 0.0,
                "maximum": 1.0,
            },
        },
        "required": ["training", "comparison", "reasoning_summary", "warnings", "confidence"],
    }


def _available_model_payload(
    summary: DatasetSummary,
    registry: ModelSpecRegistry,
) -> dict[str, list[dict[str, Any]]]:
    payload: dict[str, list[dict[str, Any]]] = {}
    for task in dict.fromkeys(summary.tasks_by_target.values()):
        payload[task] = registry.to_prompt_payload(task)
    return payload


def build_training_planner_prompt(
    *,
    goal: str,
    summary: DatasetSummary,
    registry: ModelSpecRegistry,
    llm_context: LLMContextConfig | Mapping[str, Any] | None = None,
    existing_suggestion: TrainingSuggestion | Mapping[str, Any] | None = None,
) -> str:
    """Build a JSON-only prompt without including raw dataframe rows."""

    if not str(goal).strip():
        raise ValueError("goal must not be empty.")
    context = coerce_llm_context(llm_context)
    existing_payload: dict[str, Any]
    if existing_suggestion is None:
        existing_payload = {}
    elif isinstance(existing_suggestion, TrainingSuggestion):
        existing_payload = _to_jsonable(existing_suggestion.to_dict())
    else:
        existing_payload = _to_jsonable(dict(existing_suggestion))

    payload = {
        "role": (
            "You are a malchan configuration planner for supervised machine "
            "learning workflows in materials and manufacturing data analysis."
        ),
        "goal": str(goal),
        "privacy": {
            "raw_data_policy": context.raw_data_policy,
            "raw_rows_included": False,
            "instruction": (
                "Use only the supplied statistical summary. Never request or "
                "invent raw observations, credentials, or confidential values."
            ),
        },
        "domain_context": _to_jsonable(context.to_dict()),
        "dataset_summary": summary.to_dict(),
        "available_models": _available_model_payload(summary, registry),
        "available_settings": {
            "numeric_imputation": [None, "Multiple", "mean", "median", "most_frequent", "knn"],
            "numeric_scaling": [None, "StandardScaler", "MinMaxScaler", "centering", "MaxAbsScaler"],
            "decomposition": ["PCA", "KernalPCA", "NMF", "ICA"],
            "classification_sampling": [None, "OverSampling", "UnderSampling", "SMOTE", "重みづけ"],
            "ensemble_types": ["アンサンブル", "スタッキング", "バギング", "ブースティング"],
            "cv_methods": ["kfold", "loo"],
            "regression_metrics": ["RMSE", "MAE", "MAPE", "R2"],
            "classification_metrics": ["ACCURACY", "PRECISION", "RECALL", "F1"],
        },
        "important_rules": [
            "Return JSON only and follow output_schema.",
            "Do not include API keys, credentials, raw data rows, or executable code.",
            "Use exact model names from available_models and exact malchan setting values.",
            "For ordinary non-ensemble training, choose exactly one initial model per target.",
            "Put alternative model families in comparison.model_names rather than training.model_names_by_target.",
            (
                "Prefer tune_best=true over tuning every comparison candidate unless "
                "the user explicitly requests exhaustive tuning."
            ),
            (
                "Use simple, regularized models for very small or high-dimensional "
                "datasets unless there is a strong reason not to."
            ),
            "Scaling is usually important for linear, neighbor, SVM, neural-network, Gaussian-process, and PLS models.",
            "Do not recommend SMOTE when a class is too small for neighbor-based synthesis.",
            (
                "Do not invent estimator parameter names. Omit model parameters when "
                "uncertain and let malchan defaults or Optuna handle them."
            ),
            (
                "If the evidence is insufficient or settings involve tradeoffs, add a "
                "warning rather than claiming certainty."
            ),
            (
                "This is a review-first recommendation. Do not claim that training, "
                "comparison, or tuning has already run."
            ),
        ],
        "existing_suggestion": existing_payload,
        "output_schema": training_suggestion_json_schema(),
    }
    return "Plan malchan settings from the following JSON payload.\n" + json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
    )


@dataclass(slots=True)
class TrainingPlanResult:
    """Planner output, semantic validation, and provider metadata."""

    prompt: str
    suggestion: TrainingSuggestion
    validation: SuggestionValidationResult
    response: LLMResponse | None = None
    offline: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly review payload."""

        response_metadata = None
        if self.response is not None:
            response_metadata = {
                "provider": self.response.provider,
                "model": self.response.model,
                "usage": self.response.usage,
                "request_id": self.response.request_id,
            }
        return {
            "suggestion": self.suggestion.to_dict(),
            "validation": self.validation.to_dict(),
            "response": response_metadata,
            "offline": self.offline,
        }


def plan_training_configuration(
    *,
    goal: str,
    summary: DatasetSummary,
    registry: ModelSpecRegistry | None = None,
    llm_config: Any | None = None,
    llm_context: LLMContextConfig | Mapping[str, Any] | None = None,
    planner_response: Any | None = None,
    existing_suggestion: TrainingSuggestion | Mapping[str, Any] | None = None,
    client: BaseLLMClient | None = None,
    validator: SuggestionValidator | None = None,
) -> TrainingPlanResult:
    """Generate and semantically validate a malchan configuration suggestion.

    ``planner_response`` provides a deterministic offline path for tests, CI,
    demonstrations, and regulated environments. When supplied, no provider SDK
    is imported and no external request is made.
    """

    resolved_registry = registry or build_runtime_model_registry()
    prompt = build_training_planner_prompt(
        goal=goal,
        summary=summary,
        registry=resolved_registry,
        llm_context=llm_context,
        existing_suggestion=existing_suggestion,
    )

    response: LLMResponse | None = None
    if planner_response is not None:
        payload = parse_json_payload(planner_response)
        offline = True
    else:
        resolved_client = client or make_llm_client(llm_config)
        response = resolved_client.generate_json(
            prompt,
            schema=training_suggestion_json_schema(),
            schema_name="malchan_training_suggestion",
            strict=False,
        )
        payload = parse_json_payload(response.text)
        response.parsed = payload
        offline = False

    if not isinstance(payload, Mapping):
        raise ValueError("Planner response must be a JSON object.")
    payload_dict = dict(payload)
    payload_dict.setdefault("training", {})
    payload_dict.setdefault("comparison", {})
    payload_dict.setdefault("reasoning_summary", "")
    payload_dict.setdefault("warnings", [])
    payload_dict.setdefault("confidence", None)

    suggestion = training_suggestion_from_mapping(payload_dict)
    suggestion.raw_plan = dict(payload_dict)
    resolved_validator = validator or SuggestionValidator(resolved_registry)
    validation = resolved_validator.validate(summary, suggestion)
    return TrainingPlanResult(
        prompt=prompt,
        suggestion=validation.suggestion,
        validation=validation,
        response=response,
        offline=offline,
    )
