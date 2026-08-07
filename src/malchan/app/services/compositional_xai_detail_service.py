"""Compositional-aware SHAP detail and PDP availability for the Web API."""

from __future__ import annotations

import json
from collections.abc import Mapping
from functools import wraps
from typing import Any

import numpy as np
import pandas as pd

from malchan.app.schemas import XaiShapValuesResponse

from .xai_service import XaiNotReadyError


def _target_models(registered: Any) -> dict[str, Any]:
    """Return fitted child pipelines keyed by target."""

    model_map = getattr(registered.model, "models", None)
    if isinstance(model_map, Mapping):
        return {
            target: model_map[target]
            for target in registered.info.target_cols
            if target in model_map
        }
    return {
        target: registered.model
        for target in registered.info.target_cols
    }


def _shared_columns(model: Any, name: str) -> list[str]:
    """Read column metadata from a model or its shared context."""

    if hasattr(model, "_shared_attr"):
        try:
            value = model._shared_attr(name)
        except (AttributeError, TypeError):
            value = None
    else:
        value = getattr(model, name, None)
        context = getattr(model, "context", None)
        if value is None and context is not None:
            value = getattr(context, name, None)
    return [] if value is None else list(value)


def _compositional_groups(model: Any) -> list[list[str]]:
    """Return normalized compositional groups attached during fitting."""

    groups = getattr(model, "compositional_groups", None) or []
    return [list(group) for group in groups]


def _compositional_columns(model: Any) -> set[str]:
    return {
        column
        for group in _compositional_groups(model)
        for column in group
    }


def _transformed_frame(child: Any) -> pd.DataFrame:
    """Return SHAP sample values aligned with transformed feature coordinates."""

    sample = getattr(child, "X_sample", None)
    if isinstance(sample, pd.DataFrame):
        return sample.reset_index(drop=True).copy()

    array = np.asarray(sample)
    if array.ndim != 2 or array.size == 0:
        raise XaiNotReadyError("Transformed SHAP sample values are unavailable.")
    names = list(getattr(child, "feature_names", None) or [])
    if len(names) != array.shape[1]:
        names = [f"feature_{index}" for index in range(array.shape[1])]
    return pd.DataFrame(array, columns=names)


def _normalized_shap_values(child: Any, frame: pd.DataFrame) -> np.ndarray:
    """Return SHAP values in row x feature x output layout."""

    values = np.asarray(getattr(child, "shap_values", None))
    if values.size == 0:
        raise XaiNotReadyError("Cached SHAP values are unavailable.")
    if values.shape[0] != len(frame):
        raise ValueError(
            "SHAP values and transformed SHAP sample rows are not aligned."
        )

    if values.ndim == 2:
        if values.shape[1] != frame.shape[1]:
            raise ValueError(
                "SHAP values and transformed feature columns are not aligned."
            )
        return values[:, :, None]

    if values.ndim != 3:
        raise ValueError(f"Unsupported SHAP value shape: {values.shape}")

    if values.shape[1] == frame.shape[1]:
        return values
    if values.shape[2] == frame.shape[1]:
        return np.transpose(values, (0, 2, 1))
    raise ValueError(
        "SHAP values do not contain an axis matching transformed features."
    )


def _output_names(child: Any, output_count: int) -> list[str]:
    """Match the output naming convention used by the existing SHAP cache."""

    if output_count == 1:
        return ["shap"]

    labels = getattr(child, "target_items", None)
    if labels is not None:
        labels = [str(value) for value in list(labels)]
    else:
        labels = []
    if len(labels) != output_count:
        labels = [str(index) for index in range(output_count)]
    return [f"shap_{label}" for label in labels]


def _matrix_json(values: np.ndarray) -> list[list[float | None]]:
    return json.loads(pd.DataFrame(values).to_json(orient="values"))


def _fallback_shap_values_response(
    model_id: str,
    target: str,
    child: Any,
) -> XaiShapValuesResponse:
    """Rebuild Beeswarm input directly from transformed SHAP arrays."""

    frame = _transformed_frame(child)
    values = _normalized_shap_values(child, frame)
    output_names = _output_names(child, values.shape[2])
    records = json.loads(
        frame.to_json(orient="records", date_format="iso")
    )
    shap_values = {
        output_name: _matrix_json(values[:, :, output_index])
        for output_index, output_name in enumerate(output_names)
    }
    return XaiShapValuesResponse(
        model_id=model_id,
        target=target,
        features=[str(column) for column in frame.columns],
        cat_cols=[],
        output_names=output_names,
        records=records,
        shap_values=shap_values,
    )


def install_compositional_xai_detail_service(service_cls: type[Any]) -> None:
    """Keep Beeswarm available and advertise only valid standard-PDP features."""

    if getattr(service_cls, "_compositional_xai_detail_installed", False):
        return

    original_get_xai_summary = service_cls.get_xai_summary
    original_get_xai_shap_values = service_cls.get_xai_shap_values

    @wraps(original_get_xai_summary)
    def get_xai_summary(self: Any, model_id: str) -> Any:
        summary = original_get_xai_summary(self, model_id)
        registered = self._get_registered(model_id)
        target_models = _target_models(registered)

        for target, child in target_models.items():
            groups = _compositional_groups(child)
            target_summary = summary.targets.get(target)
            if not groups or target_summary is None:
                continue

            compositional_columns = _compositional_columns(child)
            safe_raw_features = [
                column
                for column in _shared_columns(child, "all_cols")
                if column not in compositional_columns
            ]
            # ``features`` is currently used by ExplainPage as a fallback source
            # for PDP selectors. Do not advertise simplex components as if they
            # could be independently perturbed.
            target_summary.features = safe_raw_features
            cached_pdp = [
                feature
                for feature in target_summary.pdp_features
                if feature not in compositional_columns
            ]
            target_summary.pdp_features = cached_pdp or safe_raw_features

            if not target_summary.shap_features:
                try:
                    target_summary.shap_features = [
                        str(column) for column in _transformed_frame(child).columns
                    ]
                except (XaiNotReadyError, ValueError):
                    pass

        return summary

    @wraps(original_get_xai_shap_values)
    def get_xai_shap_values(self: Any, model_id: str, target: str) -> Any:
        try:
            return original_get_xai_shap_values(self, model_id, target)
        except (XaiNotReadyError, ValueError) as original_error:
            registered = self._get_registered(model_id)
            child = _target_models(registered).get(target)
            if child is None or not _compositional_groups(child):
                raise
            try:
                return _fallback_shap_values_response(model_id, target, child)
            except (XaiNotReadyError, ValueError):
                raise original_error

    service_cls.get_xai_summary = get_xai_summary
    service_cls.get_xai_shap_values = get_xai_shap_values
    service_cls._compositional_xai_detail_installed = True


__all__ = ["install_compositional_xai_detail_service"]
