import pytest

from malchan.llm.configs import LLMConfig, LLMContextConfig


def test_llm_config_masks_secrets():
    config = LLMConfig(
        provider="openai",
        model="test-model",
        api_key="secret-key",
        proxy="http://user:pass@example.test:8080",
        client_kwargs={"authorization": "Bearer secret"},
        request_kwargs={"metadata": {"access_token": "token"}},
    )

    safe = config.safe_dict()

    assert safe["api_key"] == "***"
    assert safe["proxy"] == "***"
    assert safe["client_kwargs"]["authorization"] == "***"
    assert safe["request_kwargs"]["metadata"]["access_token"] == "***"


def test_llm_config_requires_explicit_model():
    with pytest.raises(ValueError, match="model"):
        LLMConfig(model="")


def test_llm_context_validates_priority_range():
    with pytest.raises(ValueError, match="priority"):
        LLMContextConfig(priorities={"accuracy": 1.2})
