import importlib.util

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("fastapi") is None
    or importlib.util.find_spec("httpx") is None,
    reason="Local SHAP API tests require web and test extras.",
)


class LocalShapService:
    """Small service double exposing only the local-SHAP route operation."""

    def compute_local_shap(self, model_id, request):
        from malchan.app.schemas import LocalShapResponse, LocalShapTargetResponse

        assert model_id == "model-1"
        assert request.data == [{"x": 3.0}]
        return LocalShapResponse(
            model_id=model_id,
            row_count=1,
            targets={
                "property": LocalShapTargetResponse(
                    target="property",
                    features=["x"],
                    output_names=["property"],
                    records=[{"x": 3.0}],
                    shap_values={"property": [[1.5]]},
                    base_values={"property": [0.25]},
                )
            },
        )


def test_local_shap_endpoint_returns_request_scoped_values() -> None:
    """The API should expose selected-row SHAP independently from cached XAI."""

    from fastapi.testclient import TestClient

    from malchan.app import create_app

    client = TestClient(create_app(model_service=LocalShapService()))
    response = client.post(
        "/api/models/model-1/xai/local",
        json={"data": [{"x": 3.0}]},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["row_count"] == 1
    assert payload["targets"]["property"]["shap_values"]["property"] == [[1.5]]
    assert payload["targets"]["property"]["base_values"]["property"] == [0.25]
