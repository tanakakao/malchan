import json

import pandas as pd

from malchan.llm.client import BaseLLMClient, LLMResponse
from malchan.llm.planner import (
    build_training_planner_prompt,
    plan_training_configuration,
    training_suggestion_json_schema,
)
from malchan.llm.registry import ModelSpec, ModelSpecRegistry
from malchan.llm.summary import DatasetSummary


class RecordingClient(BaseLLMClient):
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def generate_json(
        self,
        prompt,
        *,
        schema=None,
        schema_name="malchan_response",
        strict=None,
    ):
        self.calls.append(
            {
                "prompt": prompt,
                "schema": schema,
                "schema_name": schema_name,
                "strict": strict,
            }
        )
        return LLMResponse(
            text=json.dumps(self.payload),
            provider="fake",
            model="fake-model",
        )


def _summary():
    df = pd.DataFrame(
        {
            "x": list(range(8)),
            "y": [float(value) for value in range(8)],
        }
    )
    return DatasetSummary.from_dataframe(
        df,
        target_cols=["y"],
        tasks=["regression"],
        num_cols=["x"],
        cat_cols=[],
    )


def _registry():
    return ModelSpecRegistry(
        [
            ModelSpec(
                name="K近傍法",
                task="regression",
                family="neighbors",
                allowed_params=frozenset({"n_neighbors"}),
            ),
            ModelSpec(
                name="Ridge",
                task="regression",
                family="linear",
                allowed_params=frozenset({"alpha"}),
            ),
        ]
    )


def _payload():
    return {
        "training": {
            "model_names_by_target": {"y": ["K近傍法"]},
            "model_params_by_target": {"y": {"n_neighbors": 100}},
            "num_scale_type": "StandardScaler",
        },
        "comparison": {
            "model_names": ["Ridge", "K近傍法"],
            "method": "kfold",
            "n_splits": 20,
            "tune_best": True,
            "tuning_trials": 30,
        },
        "reasoning_summary": (
            "Compare a linear baseline and a local nonlinear model."
        ),
        "warnings": [],
        "confidence": 0.7,
    }


def test_prompt_contains_summary_but_not_raw_rows():
    prompt = build_training_planner_prompt(
        goal="Prefer interpretable models",
        summary=_summary(),
        registry=_registry(),
    )

    assert '"raw_rows_included": false' in prompt
    assert '"n_rows": 8' in prompt
    assert '"K近傍法"' in prompt
    assert "0.0, 1.0, 2.0" not in prompt


def test_offline_planner_response_skips_provider_and_is_validated():
    result = plan_training_configuration(
        goal="Build a robust regression model",
        summary=_summary(),
        registry=_registry(),
        planner_response=_payload(),
    )

    assert result.offline is True
    assert result.response is None
    assert result.validation.status == "adjusted"
    assert result.suggestion.comparison.n_splits == 8
    assert (
        result.suggestion.training.model_params_by_target["y"]["n_neighbors"]
        == 7
    )


def test_online_path_passes_structured_schema_to_client():
    client = RecordingClient(_payload())

    result = plan_training_configuration(
        goal="Build a robust regression model",
        summary=_summary(),
        registry=_registry(),
        client=client,
    )

    assert result.offline is False
    assert client.calls[0]["schema_name"] == "malchan_training_suggestion"
    assert client.calls[0]["schema"] == training_suggestion_json_schema()
    assert client.calls[0]["strict"] is False
