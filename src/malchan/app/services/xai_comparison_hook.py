"""Refresh cached XAI after comparison activates a different fitted model."""

from typing import Any

from malchan.app.schemas import RecomputeXaiRequest


def install_xai_comparison_hooks(service_cls: type[Any]) -> None:
    """Wrap comparison operations so activated models receive fresh XAI caches."""

    if getattr(service_cls, "_xai_comparison_hooks_installed", False):
        return

    original_run_comparison = service_cls.run_comparison
    original_tune_best = service_cls.tune_best_comparison

    def recompute_activated_xai(self: Any, model_id: str) -> None:
        """Compute XAI for the activated model regardless of the seed-model setting."""

        self.recompute_xai(model_id, RecomputeXaiRequest())

    def run_comparison_with_xai(self: Any, model_id: str, request: Any) -> Any:
        response = original_run_comparison(self, model_id, request)
        if request.activate_best:
            recompute_activated_xai(self, model_id)
        return response

    def tune_best_with_xai(self: Any, model_id: str, request: Any) -> Any:
        response = original_tune_best(self, model_id, request)
        if request.activate_best:
            recompute_activated_xai(self, model_id)
        return response

    service_cls.run_comparison = run_comparison_with_xai
    service_cls.tune_best_comparison = tune_best_with_xai
    service_cls._xai_comparison_hooks_installed = True


__all__ = ["install_xai_comparison_hooks"]
