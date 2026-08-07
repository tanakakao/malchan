"""Regression tests for compositional SHAP detail and PDP availability."""

from types import SimpleNamespace

import numpy as np
import pandas as pd

from malchan.app.schemas import XaiSummaryResponse, XaiTargetSummary
from malchan.app.services.compositional_xai_detail_service import (
    _fallback_shap_values_response,
    install_compositional_xai_detail_service,
)
from malchan.app.services.xai_service import XaiNotReadyError
from malchan.pipeline.compositional_xai_extensions import (
    _compute_compositional_xai,
)


class CompositionalChild:
    """Small fitted-pipeline double with one simplex group and one ordinary feature."""

    def __init__(self) -> None:
        self.compositional_groups = [["a", "b", "c"]]
        self.all_cols = ["a", "b", "c", "temperature"]
        self.feature_names = [
            "temperature",
            "compositional_0__ilr__balance_1",
            "compositional_0__ilr__balance_2",
        ]
        self.X_sample = pd.DataFrame(
            [
                [100.0, 0.1, -0.2],
                [200.0, 0.3, 0.4],
            ],
            columns=self.feature_names,
        )
        self.shap_values = np.array(
            [
                [0.05, 0.2, -0.1],
                [0.10, -0.3, 0.4],
            ]
        )
        self.importances = None
        self.pdp_calls = []

    def _shared_attr(self, name):
        return getattr(self, name)

    def model_importance(self):
        return np.array([0.1, 0.2, 0.3])

    def pfi_importance(self):
        return np.array([0.2, 0.3, 0.4])

    def shap_importance(self):
        return np.sqrt((self.shap_values**2).sum(axis=0))

    def get_shap_scatter_data(self, feature):
        index = self.feature_names.index(feature)
        frame = self.X_sample[[feature]].copy()
        frame["shap"] = self.shap_values[:, index]
        return frame

    def get_pd_and_ice(self, feature):
        self.pdp_calls.append(feature)
        if feature in {"a", "b", "c"}:
            raise AssertionError("compositional components must not use standard PDP")
        return np.array([[1.0], [2.0]]), np.array([100.0, 200.0])

    def _combine_cat_importance(self, values):
        return values


def test_get_xai_skips_standard_pdp_for_compositional_components() -> None:
    child = CompositionalChild()

    cache = _compute_compositional_xai(child)

    assert child.pdp_calls == ["temperature"]
    assert list(cache["pd"]) == ["temperature"]
    assert list(cache["shap_pd"]) == child.feature_names


def test_beeswarm_can_be_rebuilt_directly_from_transformed_shap_arrays() -> None:
    child = CompositionalChild()

    response = _fallback_shap_values_response("model-1", "property", child)

    assert response.features == child.feature_names
    assert response.output_names == ["shap"]
    assert len(response.records) == 2
    assert np.allclose(
        np.asarray(response.shap_values["shap"]),
        child.shap_values,
    )


def test_service_falls_back_to_transformed_arrays_when_shap_pd_is_missing() -> None:
    child = CompositionalChild()
    registered = SimpleNamespace(
        model=child,
        info=SimpleNamespace(target_cols=["property"]),
    )

    class FakeService:
        def _get_registered(self, model_id):
            assert model_id == "model-1"
            return registered

        def get_xai_summary(self, model_id):
            return XaiSummaryResponse(
                model_id=model_id,
                status="ready",
                targets={
                    "property": XaiTargetSummary(
                        target="property",
                        status="ready",
                        features=["a", "b", "c", "temperature"],
                        importance_methods=["shap"],
                        shap_features=[],
                        pdp_features=[],
                    )
                },
            )

        def get_xai_shap_values(self, model_id, target):
            raise XaiNotReadyError(
                f"Cached SHAP values are unavailable for target {target!r}."
            )

    install_compositional_xai_detail_service(FakeService)
    service = FakeService()

    response = service.get_xai_shap_values("model-1", "property")
    summary = service.get_xai_summary("model-1")

    assert response.features == child.feature_names
    assert summary.targets["property"].features == ["temperature"]
    assert summary.targets["property"].pdp_features == ["temperature"]
    assert summary.targets["property"].shap_features == child.feature_names
