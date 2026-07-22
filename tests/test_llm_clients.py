from types import SimpleNamespace

from malchan.llm.client import BaseLLMClient, GeminiClient, LLMResponse, OpenAIClient
from malchan.llm.configs import LLMConfig


class FakeResponsesAPI:
    def __init__(self):
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            output_text='{"value": 1}',
            usage={"input_tokens": 10, "output_tokens": 5},
            _request_id="req-openai",
        )


class FakeOpenAIClient:
    def __init__(self):
        self.responses = FakeResponsesAPI()


class FakeGeminiModels:
    def __init__(self):
        self.kwargs = None

    def generate_content(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            text='{"value": 2}',
            usage_metadata={"prompt_token_count": 8, "candidates_token_count": 4},
            sdk_http_response=SimpleNamespace(
                headers={"x-goog-request-id": "req-gemini"}
            ),
        )


class FakeGeminiClient:
    def __init__(self):
        self.models = FakeGeminiModels()


class FakeStructuredClient(BaseLLMClient):
    def generate_json(
        self,
        prompt,
        *,
        schema=None,
        schema_name="malchan_response",
        strict=None,
    ):
        return LLMResponse(text='{"value": 7}', provider="fake", model="fake")


class FakeResponseModel:
    @classmethod
    def model_json_schema(cls):
        return {
            "type": "object",
            "properties": {"value": {"type": "integer"}},
        }

    @classmethod
    def model_validate(cls, value):
        return SimpleNamespace(value=value["value"])


def test_openai_adapter_uses_responses_structured_output():
    fake = FakeOpenAIClient()
    adapter = OpenAIClient(LLMConfig(model="openai-test"), client=fake)

    result = adapter.generate_json(
        "prompt",
        schema={"type": "object"},
        schema_name="test_schema",
        strict=True,
    )

    request = fake.responses.kwargs
    assert request["text"]["format"]["type"] == "json_schema"
    assert request["text"]["format"]["strict"] is True
    assert request["store"] is False
    assert result.request_id == "req-openai"
    assert result.usage["input_tokens"] == 10


def test_gemini_adapter_uses_response_schema():
    fake = FakeGeminiClient()
    adapter = GeminiClient(
        LLMConfig(provider="gemini", model="gemini-test"),
        client=fake,
    )

    result = adapter.generate_json("prompt", schema={"type": "object"})

    request = fake.models.kwargs
    assert request["config"]["response_mime_type"] == "application/json"
    assert request["config"]["response_schema"] == {"type": "object"}
    assert result.request_id == "req-gemini"
    assert result.usage["prompt_token_count"] == 8


def test_base_client_validates_pydantic_compatible_response():
    response = FakeStructuredClient().generate_structured(
        "prompt",
        FakeResponseModel,
    )

    assert response.parsed.value == 7
