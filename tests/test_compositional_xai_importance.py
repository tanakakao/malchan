"""Tests for default compositional feature-importance aggregation."""

from types import SimpleNamespace

import pytest

from malchan.app.schemas import XaiImportanceItem, XaiImportanceResponse
from malchan.app.services.compositional_xai_service import (
    _aggregate_compositional_importance,
    install_compositional_xai_service,
)


def _raw_response() -> XaiImportanceResponse:
    return XaiImportanceResponse(
        model_id="model-1",
        target="strength",
        method="shap",
        combined=False,
        items=[
            XaiImportanceItem(feature="temperature", value=0.4),
            XaiImportanceItem(
                feature="compositional_0__ilr__balance_1",
                value=-0.3,
            ),
            XaiImportanceItem(
                feature="compositional_0__ilr__balance_2",
                value=0.5,
            ),
        ],
    )


def test_compositional_importance_is_visible_as_group_by_default() -> None:
    """ILR coordinates should appear as one readable group-level importance."""

    child = SimpleNamespace(
        num_cols=["a", "b", "c", "temperature"],
        cat_cols=[],
        smiles_cols=[],
        comp_cols=[],
        compositional_groups=[["a", "b", "c"]],
    )

    response = _aggregate_compositional_importance(
        child,
        _raw_response(),
        top_n=20,
    )

    assert response.combined is True
    assert [item.feature for item in response.items] == [
        "組成比: a / b / c",
        "temperature",
    ]
    assert response.items[0].value == pytest.approx(0.8)
    assert response.items[1].value == pytest.approx(0.4)


def test_service_defaults_to_compositional_combined_importance() -> None:
    """The Web default combined request should use raw values for safe aggregation."""

    child = SimpleNamespace(
        num_cols=["a", "b", "c", "temperature"],
        cat_cols=[],
        smiles_cols=[],
        comp_cols=[],
        compositional_groups=[["a", "b", "c"]],
    )

    class FakeService:
        def __init__(self) -> None:
            self.calls = []
            self.registered = SimpleNamespace(
                model=child,
                info=SimpleNamespace(target_cols=["strength"]),
            )

        def _get_registered(self, model_id):
            assert model_id == "model-1"
            return self.registered

        def get_xai_importance(
            self,
            model_id,
            target,
            method,
            combined=True,
            top_n=None,
        ):
            self.calls.append(
                {
                    "model_id": model_id,
                    "target": target,
                    "method": method,
                    "combined": combined,
                    "top_n": top_n,
                }
            )
            return _raw_response()

    install_compositional_xai_service(FakeService)
    service = FakeService()

    response = service.get_xai_importance(
        "model-1",
        "strength",
        "shap",
    )

    assert service.calls == [
        {
            "model_id": "model-1",
            "target": "strength",
            "method": "shap",
            "combined": False,
            "top_n": None,
        }
    ]
    assert response.combined is True
    assert response.items[0].feature == "組成比: a / b / c"


def test_non_compositional_models_keep_existing_importance_behavior() -> None:
    """Models without compositional groups should not be post-processed."""

    child = SimpleNamespace(compositional_groups=[])

    class FakeService:
        def __init__(self) -> None:
            self.calls = []
            self.registered = SimpleNamespace(
                model=child,
                info=SimpleNamespace(target_cols=["strength"]),
            )

        def _get_registered(self, model_id):
            return self.registered

        def get_xai_importance(
            self,
            model_id,
            target,
            method,
            combined=True,
            top_n=None,
        ):
            self.calls.append(combined)
            return XaiImportanceResponse(
                model_id=model_id,
                target=target,
                method=method,
                combined=combined,
                items=[XaiImportanceItem(feature="temperature", value=0.4)],
            )

    install_compositional_xai_service(FakeService)
    service = FakeService()

    response = service.get_xai_importance(
        "model-1",
        "strength",
        "shap",
    )

    assert service.calls == [True]
    assert response.items[0].feature == "temperature"
