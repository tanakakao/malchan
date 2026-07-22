# malchan LLM planning

`malchan.llm` proposes machine-learning settings for review. The LLM does not fit
models, run cross-validation, tune hyperparameters, or activate a model.

## Installation

```bash
pip install -e ".[llm]"
```

Set one provider key in the server or notebook environment:

```bash
export OPENAI_API_KEY="..."
# or
export GEMINI_API_KEY="..."
```

The provider model name is explicit because available model names can change.

## OpenAI

```python
from malchan.llm import DatasetSummary, LLMConfig, plan_training_configuration

summary = DatasetSummary.from_dataframe(
    df,
    target_cols=["strength"],
    tasks=["regression"],
    num_cols=["temperature", "time"],
    cat_cols=["atmosphere"],
)

result = plan_training_configuration(
    goal="Prioritize stable prediction and interpretability for a small dataset.",
    summary=summary,
    llm_config=LLMConfig(
        provider="openai",
        model="YOUR_OPENAI_MODEL",
    ),
)

print(result.validation.status)
print(result.suggestion.to_dict())
```

## Gemini

Only the provider configuration changes:

```python
result = plan_training_configuration(
    goal="Compare robust regression candidates and tune only the best family.",
    summary=summary,
    llm_config=LLMConfig(
        provider="gemini",
        model="YOUR_GEMINI_MODEL",
    ),
)
```

## Offline and CI use

Pass an explicit planner response to avoid importing a provider SDK or making a
network request:

```python
result = plan_training_configuration(
    goal="Use a regularized linear baseline.",
    summary=summary,
    planner_response={
        "training": {
            "model_names_by_target": {"strength": ["Ridge"]},
            "num_scale_type": "StandardScaler",
        },
        "comparison": {
            "model_names": ["Ridge", "ランダムフォレスト回帰"],
            "method": "kfold",
            "n_splits": 5,
            "tune_best": True,
            "tuning_trials": 50,
        },
        "reasoning_summary": "Use Ridge as an interpretable baseline.",
        "warnings": [],
        "confidence": 0.7,
    },
)
```

## Privacy and safety boundary

The planner prompt contains `DatasetSummary`, model capabilities, and user-supplied
domain notes. Raw dataframe rows are not included. API keys are read from
environment variables by default and are redacted from `safe_dict()` output.

Every generated plan is passed through `SuggestionValidator`. The result is
classified as `accepted`, `adjusted`, or `rejected`; only the validated suggestion
should be presented for manual application.
