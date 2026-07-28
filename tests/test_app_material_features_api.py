"""FastAPI SMILES and composition feature integration tests."""

import importlib.util

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("fastapi") is None
    or importlib.util.find_spec("httpx") is None,
    reason="FastAPI API tests require the web and test extras.",
)


class RecordingPipeline:
    """Record single-output fit arguments received from the model service."""

    def __init__(self) -> None:
        self.fit_kwargs = None

    def fit(self, **kwargs) -> None:
        self.fit_kwargs = kwargs


class RecordingMultiPipeline:
    """Record multi-output fit arguments received from the model service."""

    def __init__(self) -> None:
        self.fit_kwargs = None

    def fit(self, **kwargs) -> None:
        self.fit_kwargs = kwargs


def _records() -> list[dict]:
    """Return rows containing numeric, SMILES, composition, and target data."""

    return [
        {
            "temperature": 100.0,
            "smiles_a": "CCO",
            "smiles_b": "CC",
            "formula_a": "LiFePO4",
            "formula_b": "SiO2",
            "strength": 1.0,
            "cost": 4.0,
        },
        {
            "temperature": 200.0,
            "smiles_a": "CCN",
            "smiles_b": "CO",
            "formula_a": "LiCoO2",
            "formula_b": "Al2O3",
            "strength": 2.0,
            "cost": 3.0,
        },
        {
            "temperature": 300.0,
            "smiles_a": "CCC",
            "smiles_b": "CN",
            "formula_a": "NaCl",
            "formula_b": "TiO2",
            "strength": 3.0,
            "cost": 2.0,
        },
    ]


def _single_payload() -> dict:
    """Return a single-output material-feature training request."""

    return {
        "data": _records(),
        "target_col": "strength",
        "task": "regression",
        "num_cols": ["temperature"],
        "cat_cols": [],
        "smiles_cols": ["smiles_a", "smiles_b"],
        "fingerprints": ["ecfp", "maccs"],
        "comp_cols": ["formula_a", "formula_b"],
        "comp_method": "MATMINER",
        "comp_feats": ["elementproperty", "stoichiometry"],
        "model_names": ["model-a"],
        "compute_xai": False,
    }


def _make_client(
    pipeline: RecordingPipeline | None = None,
    multi_pipeline: RecordingMultiPipeline | None = None,
):
    """Create a TestClient backed by recording pipeline doubles."""

    from fastapi.testclient import TestClient

    from malchan.app import create_app
    from malchan.app.services import InMemoryModelService

    single = pipeline or RecordingPipeline()
    multi = multi_pipeline or RecordingMultiPipeline()
    service = InMemoryModelService(
        model_factory=lambda: single,
        multi_model_factory=lambda: multi,
        id_factory=lambda: "material-model",
    )
    return TestClient(create_app(model_service=service)), single, multi


def test_fastapi_forwards_normalized_smiles_and_composition_settings() -> None:
    """Single-output training should preserve material-feature configuration."""

    client, pipeline, _ = _make_client()

    response = client.post("/api/models", json=_single_payload())

    assert response.status_code == 201
    assert pipeline.fit_kwargs["smiles_cols"] == ["smiles_a", "smiles_b"]
    assert pipeline.fit_kwargs["fingerprints"] == ["ECFP", "MACCS"]
    assert pipeline.fit_kwargs["comp_cols"] == ["formula_a", "formula_b"]
    assert pipeline.fit_kwargs["comp_method"] == "matminer"
    assert pipeline.fit_kwargs["comp_feats"] == [
        "ElementProperty",
        "Stoichiometry",
    ]
    assert response.json()["feature_columns"] == [
        "temperature",
        "smiles_a",
        "smiles_b",
        "formula_a",
        "formula_b",
    ]


def test_fastapi_forwards_pymatgen_basic_statistics_settings() -> None:
    """Pymatgen basic properties should be accepted as an app composition method."""

    client, pipeline, _ = _make_client()
    payload = _single_payload()
    payload["comp_method"] = "PYMATGEN"
    payload["comp_feats"] = [
        "ATOMIC_NUMBER",
        "atomic_mass",
        "electronegativity",
    ]

    response = client.post("/api/models", json=payload)

    assert response.status_code == 201
    assert pipeline.fit_kwargs["comp_method"] == "pymatgen"
    assert pipeline.fit_kwargs["comp_feats"] == [
        "atomic_number",
        "atomic_mass",
        "electronegativity",
    ]


def test_fastapi_forwards_material_features_to_multi_output_pipeline() -> None:
    """Multi-output training should use the same shared feature configuration."""

    pipeline = RecordingMultiPipeline()
    client, _, _ = _make_client(multi_pipeline=pipeline)
    payload = _single_payload()
    payload.pop("target_col")
    payload.pop("task")
    payload["target_cols"] = ["strength", "cost"]
    payload["tasks"] = ["regression", "regression"]
    payload["model_names_by_target"] = {
        "strength": ["model-a"],
        "cost": ["model-b"],
    }
    payload.pop("model_names")

    response = client.post("/api/models", json=payload)

    assert response.status_code == 201
    assert pipeline.fit_kwargs["smiles_cols"] == ["smiles_a", "smiles_b"]
    assert pipeline.fit_kwargs["fingerprints"] == ["ECFP", "MACCS"]
    assert pipeline.fit_kwargs["comp_cols"] == ["formula_a", "formula_b"]
    assert pipeline.fit_kwargs["comp_method"] == "matminer"
    assert pipeline.fit_kwargs["comp_feats"] == [
        "ElementProperty",
        "Stoichiometry",
    ]


@pytest.mark.parametrize(
    ("patch", "expected_message"),
    [
        (
            {"smiles_cols": ["smiles_a"], "fingerprints": []},
            "fingerprints must contain at least one value",
        ),
        (
            {"smiles_cols": [], "fingerprints": ["ECFP"]},
            "smiles_cols must contain at least one column",
        ),
        (
            {"fingerprints": ["unknown"]},
            "Unsupported fingerprints",
        ),
        (
            {"comp_cols": ["formula_a"], "comp_method": None, "comp_feats": []},
            "comp_method is required",
        ),
        (
            {"comp_cols": [], "comp_method": "mendeleev", "comp_feats": []},
            "comp_cols is required",
        ),
        (
            {"comp_method": "pymatgen", "comp_feats": []},
            "comp_feats must contain at least one Pymatgen property",
        ),
        (
            {"comp_method": "pymatgen", "comp_feats": ["unknown"]},
            "Unsupported Pymatgen comp_feats",
        ),
        (
            {"comp_method": "matminer", "comp_feats": []},
            "comp_feats must contain at least one Matminer featurizer",
        ),
        (
            {"comp_method": "matminer", "comp_feats": ["unknown"]},
            "Unsupported Matminer comp_feats",
        ),
        (
            {"comp_method": "xenonpy", "comp_feats": []},
            "Unsupported comp_method",
        ),
        (
            {"comp_method": "unknown"},
            "Unsupported comp_method",
        ),
    ],
)
def test_fastapi_rejects_invalid_material_feature_combinations(
    patch: dict,
    expected_message: str,
) -> None:
    """Invalid chemistry settings should fail before model fitting starts."""

    client, pipeline, _ = _make_client()
    payload = _single_payload()
    payload.update(patch)

    response = client.post("/api/models", json=payload)

    assert response.status_code == 422
    assert expected_message in response.text
    assert pipeline.fit_kwargs is None
