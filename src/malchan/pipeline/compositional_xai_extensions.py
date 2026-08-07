"""Compositional-aware XAI behavior for public model pipelines."""

from __future__ import annotations

from functools import wraps
from typing import Any


def _compositional_groups(model: Any) -> list[list[str]]:
    """Return normalized compositional groups stored on a fitted pipeline."""

    groups = getattr(model, "compositional_groups", None) or []
    return [list(group) for group in groups]


def _compositional_columns(model: Any) -> set[str]:
    """Return all raw feature columns participating in compositional groups."""

    return {
        column
        for group in _compositional_groups(model)
        for column in group
    }


def _safe_pdp_features(model: Any) -> list[str]:
    """Return raw features that can be perturbed independently for standard PDP."""

    compositional_columns = _compositional_columns(model)
    all_columns = list(model._shared_attr("all_cols"))
    return [
        column
        for column in all_columns
        if column not in compositional_columns
    ]


def _compute_compositional_xai(model: Any) -> dict[str, Any]:
    """Build XAI caches without independently perturbing simplex components.

    Standard one-dimensional PDP assumes that the selected feature can be varied
    independently while all other features remain fixed. That assumption is not
    valid for a compositional group because its components are coupled by the
    closure constraint. Attempting the legacy PDP on one component can also create
    an all-zero composition and make ILR/CLR/ALR preprocessing fail.

    SHAP is calculated in the transformed Euclidean feature space, so transformed
    log-ratio coordinates remain valid for importance and beeswarm plots.
    """

    importances = {
        "model": model.model_importance(),
        "pfi": model.pfi_importance(),
        "shap": model.shap_importance(),
        "shap_pd": {
            feature: model.get_shap_scatter_data(feature)
            for feature in model.feature_names
        },
        "pd": {
            feature: model.get_pd_and_ice(feature)
            for feature in _safe_pdp_features(model)
        },
    }
    importances["model_combine"] = model._combine_cat_importance(importances["model"])
    importances["pfi_combine"] = model._combine_cat_importance(importances["pfi"])
    importances["shap_combine"] = model._combine_cat_importance(importances["shap"])
    model.importances = importances
    return importances


def install_compositional_xai_extensions(single_output_cls: type[Any]) -> None:
    """Skip invalid raw-component PDP while retaining SHAP caches."""

    if getattr(single_output_cls, "_compositional_xai_extensions_installed", False):
        return

    original_get_xai = single_output_cls.get_xai

    @wraps(original_get_xai)
    def get_xai(self: Any) -> None:
        if not _compositional_groups(self):
            return original_get_xai(self)
        _compute_compositional_xai(self)
        return None

    single_output_cls.get_xai = get_xai
    single_output_cls._compositional_xai_extensions_installed = True


__all__ = [
    "install_compositional_xai_extensions",
]
