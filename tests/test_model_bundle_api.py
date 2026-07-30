"""Tests for signed downloadable model bundles."""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("fastapi") is None
    or importlib.util.find_spec("httpx") is None,
    reason="Model bundle API tests require the web and test extras.",
)


class BundlePipeline:
    """Pickle-compatible fitted model double."""

    def __init__(self) -> None:
        """Initialize raw feature metadata."""

        self.X = None
        self.num_cols = []
        self.cat_cols = []
        self.smiles_cols = []
        self.comp_cols = []

    def fit(self, **kwargs) -> None:
        """Record the subset required after bundle restoration."""

        self.num_cols = list(kwargs["num_cols"])
        self.cat_cols = list(kwargs["cat_cols"])
        self.smiles_cols = list(kwargs["smiles_cols"])
        self.comp_cols = list(kwargs["comp_cols"])
        columns = [
            *self.num_cols,
            *self.cat_cols,
            *self.smiles_cols,
            *self.comp_cols,
        ]
        self.X = kwargs["df"][columns].copy()

    def predict(self, X, proba=False, idx2item=False):
        """Return deterministic predictions after loading."""

        return pd.DataFrame({"y": [float(value) * 2 for value in X["x"]]})


def _payload() -> dict:
    """Return one minimal training request."""

    return {
        "data": [
            {"x": 1.0, "y": 2.0},
            {"x": 2.0, "y": 4.0},
        ],
        "target_col": "y",
        "task": "regression",
        "num_cols": ["x"],
        "cat_cols": [],
        "model_names": ["dummy"],
        "compute_xai": False,
    }


def _client(secret: str | None, ids: list[str], max_bytes: int = 1024 * 1024):
    """Create a client with deterministic model identifiers."""

    from fastapi.testclient import TestClient

    from malchan.app import AppSettings, create_app
    from malchan.app.services import InMemoryModelService

    iterator = iter(ids)
    service = InMemoryModelService(
        model_factory=BundlePipeline,
        id_factory=iterator.__next__,
    )
    settings = AppSettings(
        serve_frontend=False,
        model_bundle_secret=secret,
        model_bundle_max_bytes=max_bytes,
    )
    return TestClient(create_app(settings=settings, model_service=service))


def test_model_bundle_round_trip_restores_prediction_and_columns() -> None:
    """A downloaded bundle should restore a usable model without server storage."""

    client = _client("a" * 32, ["model-1", "model-2"])
    trained = client.post("/api/models", json=_payload())
    exported = client.get("/api/models/model-1/export")

    assert trained.status_code == 201
    assert exported.status_code == 200
    assert exported.headers["content-type"].startswith("application/vnd.malchan.model")
    assert exported.headers["cache-control"] == "no-store"
    assert ".malchan" in exported.headers["content-disposition"]

    assert client.delete("/api/models/model-1").status_code == 204
    restored = client.post(
        "/api/model-bundles/import",
        content=exported.content,
        headers={"Content-Type": "application/vnd.malchan.model"},
    )
    predicted = client.post(
        "/api/models/model-2/predict",
        json={"data": [{"x": 3.0}]},
    )

    assert restored.status_code == 201
    assert restored.json()["model"]["model_id"] == "model-2"
    assert restored.json()["original_model_id"] == "model-1"
    assert restored.json()["num_cols"] == ["x"]
    assert restored.json()["cat_cols"] == []
    assert predicted.json()["predictions"] == [{"y": 6.0}]


def test_model_bundle_rejects_modified_or_differently_signed_files() -> None:
    """HMAC verification should reject tampering and a different environment secret."""

    source = _client("a" * 32, ["model-1"])
    source.post("/api/models", json=_payload())
    bundle = source.get("/api/models/model-1/export").content

    tampered = bytearray(bundle)
    tampered[-1] ^= 1
    assert source.post("/api/model-bundles/import", content=bytes(tampered)).status_code == 422

    destination = _client("b" * 32, ["model-2"])
    response = destination.post("/api/model-bundles/import", content=bundle)
    assert response.status_code == 422
    assert "署名" in response.json()["detail"]


def test_model_bundle_requires_configured_secret_and_enforces_size_limit() -> None:
    """Unsafe unsigned operation and oversized uploads should remain disabled."""

    unsigned = _client(None, ["model-1"])
    unsigned.post("/api/models", json=_payload())
    unavailable = unsigned.get("/api/models/model-1/export")
    assert unavailable.status_code == 503
    assert "MALCHAN_MODEL_BUNDLE_SECRET" in unavailable.json()["detail"]

    limited = _client("a" * 32, ["model-1"], max_bytes=64)
    oversized = limited.post("/api/model-bundles/import", content=b"x" * 65)
    assert oversized.status_code == 413


def test_model_bundle_service_does_not_use_server_filesystem() -> None:
    """The implementation should serialize request and response bytes only in memory."""

    source = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "malchan"
        / "app"
        / "services"
        / "model_bundle_service.py"
    ).read_text(encoding="utf-8")

    assert "tempfile" not in source
    assert "NamedTemporaryFile" not in source
    assert "Path(" not in source
    assert "open(" not in source
    assert "pickle.loads(payload)" in source
    assert "hmac.compare_digest(supplied_signature, expected_signature)" in source
    assert source.index("hmac.compare_digest(supplied_signature, expected_signature)") < source.index("pickle.loads(payload)")
