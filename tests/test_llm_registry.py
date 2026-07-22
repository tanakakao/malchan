import json

import pytest

from malchan.llm import ModelSpec, ModelSpecRegistry


def test_registry_registers_and_filters_models():
    registry = ModelSpecRegistry(
        [
            ModelSpec(
                name="Ridge",
                task="regression",
                family="linear",
            ),
            ModelSpec(
                name="Logistic",
                task="classification",
                family="linear",
            ),
        ]
    )

    assert registry.names("regression") == ["Ridge"]
    assert registry.get("classification", "Logistic").family == "linear"
    assert len(registry.to_prompt_payload()) == 2


def test_registry_rejects_duplicate_keys():
    registry = ModelSpecRegistry(
        [ModelSpec(name="Ridge", task="regression", family="linear")]
    )

    with pytest.raises(ValueError, match="already registered"):
        registry.register(
            ModelSpec(name="Ridge", task="regression", family="linear")
        )


def test_model_spec_validates_allowed_parameters():
    spec = ModelSpec(
        name="KNN",
        task="regression",
        family="neighbors",
        allowed_params=frozenset({"n_neighbors"}),
    )

    assert spec.accepts_parameter("n_neighbors")
    assert not spec.accepts_parameter("max_depth")


def test_model_spec_to_dict_serializes_non_json_default_values():
    spec = ModelSpec(
        name="Gaussian process",
        task="regression",
        family="gaussian_process",
        default_params={"kernel": object()},
    )

    payload = spec.to_dict()

    assert isinstance(payload["default_params"]["kernel"], str)
    json.dumps(payload)
