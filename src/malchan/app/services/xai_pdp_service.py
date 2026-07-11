"""Safe cached PDP serialization for regression and classification targets."""

from typing import Any

import numpy as np

from malchan.app.schemas import XaiPdpResponse, XaiPdpSeries

from .xai_service import XaiNotReadyError, _cached_target, _json_scalar


def get_xai_pdp(
    service: Any,
    model_id: str,
    target: str,
    feature: str,
    include_ice: bool = False,
    max_ice: int = 30,
) -> XaiPdpResponse:
    """Return cached PDP means without ambiguous NumPy truth-value checks."""

    child, cache = _cached_target(service, model_id, target)
    pdp_cache = cache.get("pd")
    if not isinstance(pdp_cache, dict) or feature not in pdp_cache:
        available = sorted(pdp_cache) if isinstance(pdp_cache, dict) else []
        raise ValueError(
            f"Unknown or unavailable PDP feature {feature!r}. Available: {available}"
        )
    cached = pdp_cache[feature]
    if cached is None or len(cached) != 2:
        raise XaiNotReadyError(
            f"Cached PDP data is unavailable for feature {feature!r}."
        )

    values, ticks = cached
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim not in {2, 3}:
        raise ValueError(f"Unsupported cached PDP shape: {array.shape}")

    if array.ndim == 2:
        mean_values = array.mean(axis=1, keepdims=True)
        labels = [target]
    else:
        mean_values = array.mean(axis=1)
        target_items = getattr(child, "target_items", None)
        labels = [] if target_items is None else [str(value) for value in list(target_items)]
        if len(labels) != mean_values.shape[1]:
            labels = [f"output_{index}" for index in range(mean_values.shape[1])]

    series: list[XaiPdpSeries] = []
    for output_index, label in enumerate(labels):
        ice_values = None
        if include_ice:
            if array.ndim == 2:
                ice_values = array[:, :max_ice].T.tolist()
            else:
                ice_values = array[:, :max_ice, output_index].T.tolist()
        series.append(
            XaiPdpSeries(
                name=label,
                pd_values=mean_values[:, output_index].tolist(),
                ice_values=ice_values,
            )
        )

    tick_values = np.asarray(ticks).tolist()
    return XaiPdpResponse(
        model_id=model_id,
        target=target,
        feature=feature,
        x_values=[_json_scalar(value) for value in tick_values],
        series=series,
    )


def install_xai_pdp_service(service_cls: type[Any]) -> None:
    """Install the safe PDP serializer on the application service."""

    service_cls.get_xai_pdp = get_xai_pdp


__all__ = ["install_xai_pdp_service"]
