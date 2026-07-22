from malchan.llm.parser import (
    parse_json_payload,
    schema_from_response_model,
    validate_structured_payload,
)


class FakePydanticModel:
    def __init__(self, value):
        self.value = value

    @classmethod
    def model_json_schema(cls):
        return {
            "type": "object",
            "properties": {"value": {"type": "integer"}},
            "required": ["value"],
        }

    @classmethod
    def model_validate(cls, value):
        return cls(value["value"])


def test_parser_removes_json_fence():
    assert parse_json_payload('```json\n{"value": 3}\n```') == {"value": 3}


def test_optional_pydantic_compatible_validation():
    schema = schema_from_response_model(FakePydanticModel)
    parsed = validate_structured_payload('{"value": 4}', FakePydanticModel)

    assert schema["type"] == "object"
    assert parsed.value == 4
