"""Tests for feature-importance recovery after partial XAI failures."""

from types import SimpleNamespace

import numpy as np

from malchan.app.services.xai_importance_fallback_service import (
    _recover_child_importance,
    install_xai_importance_fallback_service,
)


class RecoverableChild:
    """Model double exposing importance even when full XAI cache creation failed."""

    def __init__(self) -> None:
        self.shap_values = np.array([[0.2, -0.1], [0.3, 0.4]])
        self.importances = None

    def model_importance(self):
        return None

    def pfi_importance(self):
        return np.array([0.25, 0.5])

    def shap_importance(self):
        return np.sqrt((self.shap_values**2).sum(axis=0))


def test_recover_child_importance_collects_methods_independently() -> None:
    """PFI and SHAP importance survive even if another importance is unavailable."""

    child = RecoverableChild()

    available, errors = _recover_child_importance(child)

    assert available == ["pfi", "shap"]
    assert errors == []
    assert child.importances["pfi"].tolist() == [0.25, 0.5]
    assert np.allclose(
        child.importances["shap"],
        np.sqrt((child.shap_values**2).sum(axis=0)),
    )


def test_training_promotes_failed_xai_when_importance_is_recoverable() -> None:
    """A failed full-XAI pass should still become displayable after training."""

    child = RecoverableChild()
    info = SimpleNamespace(
        model_id="model-1",
        target_cols=["strength"],
        xai_status="failed",
    )
    registered = SimpleNamespace(model=child, info=info)

    class FakeService:
        def __init__(self) -> None:
            self._xai_states = {
                "model-1": {
                    "requested": True,
                    "status": "failed",
                    "targets": {
                        "strength": {
                            "target": "strength",
                            "status": "failed",
                            "computed_at": None,
                            "error": "ValueError: PDP failed for a compositional row.",
                        }
                    },
                }
            }

        def _get_registered(self, model_id):
            assert model_id == "model-1"
            return registered

        def train(self, request):
            return info

        def recompute_xai(self, model_id, request):
            return self.get_xai_summary(model_id)

        def get_xai_summary(self, model_id):
            target_state = self._xai_states[model_id]["targets"]["strength"]
            return SimpleNamespace(
                model_id=model_id,
                status=self._xai_states[model_id]["status"],
                targets={
                    "strength": SimpleNamespace(
                        status=target_state["status"],
                        importance_methods=[
                            method
                            for method in ("model", "pfi", "shap")
                            if child.importances is not None
                            and child.importances.get(method) is not None
                        ],
                    )
                },
            )

    install_xai_importance_fallback_service(FakeService)
    service = FakeService()
    request = SimpleNamespace(compute_xai=True)

    result = service.train(request)
    summary = service.get_xai_summary("model-1")

    assert result.xai_status == "ready"
    assert summary.status == "ready"
    assert summary.targets["strength"].status == "ready"
    assert summary.targets["strength"].importance_methods == ["pfi", "shap"]
    assert child.importances["shap"] is not None


def test_recompute_recovers_importance_after_full_xai_failure() -> None:
    """The explicit recompute action must also recover displayable importance."""

    child = RecoverableChild()
    info = SimpleNamespace(
        model_id="model-1",
        target_cols=["strength"],
        xai_status="failed",
    )
    registered = SimpleNamespace(model=child, info=info)

    class FakeService:
        def __init__(self) -> None:
            self._xai_states = {
                "model-1": {
                    "requested": True,
                    "status": "failed",
                    "targets": {
                        "strength": {
                            "target": "strength",
                            "status": "failed",
                            "computed_at": None,
                            "error": "RuntimeError: full XAI failed",
                        }
                    },
                }
            }

        def _get_registered(self, model_id):
            return registered

        def train(self, request):
            return info

        def recompute_xai(self, model_id, request):
            self._xai_states[model_id]["status"] = "failed"
            self._xai_states[model_id]["targets"]["strength"]["status"] = "failed"
            return None

        def get_xai_summary(self, model_id):
            state = self._xai_states[model_id]
            return SimpleNamespace(status=state["status"])

    install_xai_importance_fallback_service(FakeService)
    service = FakeService()

    summary = service.recompute_xai(
        "model-1",
        SimpleNamespace(targets=["strength"]),
    )

    assert summary.status == "ready"
    assert info.xai_status == "ready"
    assert service._xai_states["model-1"]["targets"]["strength"]["status"] == "ready"
    assert child.importances["pfi"] is not None
