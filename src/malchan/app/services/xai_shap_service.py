"""Service adapters for cached and request-scoped SHAP values."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from malchan.app.schemas import (
    LocalShapRequest,
    LocalShapResponse,
    LocalShapTargetResponse,
    XaiShapValuesResponse,
)

from .xai_service import (
    XaiNotReadyError,
    _cached_target,
    _shared_columns,
    _target_models,
)


def _ordered_shap_features(
    child: Any,
    shap_cache: Mapping[str, Any],
) -> list[str]:
    """Return cached SHAP features in the model's original column order."""

    cached_features = [str(feature) for feature in shap_cache]
    model_features = [
        feature
        for feature in _shared_columns(child, "all_cols")
        if feature in shap_cache
    ]
    return model_features + [
        feature for feature in cached_features if feature not in model_features
    ]


def _shap_value_columns(frame: pd.DataFrame) -> list[str]:
    """Return regression or class-specific SHAP columns from one cache frame."""

    return [
        str(column)
        for column in frame.columns
        if str(column) == "shap" or str(column).startswith("shap_")
    ]


def get_xai_shap_values(
    service: Any,
    model_id: str,
    target: str,
) -> XaiShapValuesResponse:
    """Return all raw feature values and aligned cached SHAP matrices for one target.

    The existing XAI cache stores one SHAP scatter frame per raw feature. This
    function reconstructs a row-by-feature matrix without recalculating SHAP.
    Regression normally returns one ``shap`` matrix. Classification may return
    one matrix per class, such as ``shap_OK`` and ``shap_NG``.
    """

    child, cache = _cached_target(service, model_id, target)
    shap_cache = cache.get("shap_pd")
    if not isinstance(shap_cache, Mapping) or not shap_cache:
        raise XaiNotReadyError(
            f"Cached SHAP values are unavailable for target {target!r}."
        )

    features = _ordered_shap_features(child, shap_cache)
    if not features:
        raise XaiNotReadyError(
            f"Cached SHAP values contain no features for target {target!r}."
        )

    frames: dict[str, pd.DataFrame] = {}
    row_count: int | None = None
    output_names: list[str] | None = None

    for feature in features:
        cached = shap_cache[feature]
        frame = cached if isinstance(cached, pd.DataFrame) else pd.DataFrame(cached)
        frame = frame.reset_index(drop=True)
        if feature not in frame.columns:
            raise ValueError(
                f"Cached SHAP records for feature {feature!r} do not contain "
                "the raw feature column."
            )

        current_outputs = _shap_value_columns(frame)
        if not current_outputs:
            raise XaiNotReadyError(
                f"Cached SHAP records for feature {feature!r} contain no SHAP values."
            )

        if row_count is None:
            row_count = len(frame)
            output_names = current_outputs
        elif len(frame) != row_count:
            raise ValueError(
                "Cached SHAP feature frames must contain the same number of rows."
            )

        if current_outputs != output_names:
            raise ValueError(
                "Cached SHAP feature frames must expose the same SHAP output columns."
            )
        frames[feature] = frame

    assert output_names is not None

    raw_frame = pd.DataFrame(
        {feature: frames[feature][feature] for feature in features}
    )
    records = json.loads(
        raw_frame.to_json(
            orient="records",
            date_format="iso",
            date_unit="ms",
        )
    )

    shap_values: dict[str, list[list[float | None]]] = {}
    for output_name in output_names:
        values_frame = pd.DataFrame(
            {
                feature: pd.to_numeric(
                    frames[feature][output_name],
                    errors="coerce",
                )
                for feature in features
            }
        )
        shap_values[output_name] = json.loads(
            values_frame.to_json(orient="values")
        )

    cat_cols = [
        feature
        for feature in _shared_columns(child, "cat_cols")
        if feature in features
    ]
    return XaiShapValuesResponse(
        model_id=model_id,
        target=target,
        features=features,
        cat_cols=cat_cols,
        output_names=output_names,
        records=records,
        shap_values=shap_values,
    )


def _transformed_feature_names(
    child: Any,
    preprocess: Any,
    width: int,
) -> list[str]:
    """Resolve names aligned with the fitted preprocessor output."""

    fitted_frame = getattr(child, "df_prerpocessed", None)
    if isinstance(fitted_frame, pd.DataFrame) and len(fitted_frame.columns) == width:
        return [str(column) for column in fitted_frame.columns]

    feature_names = getattr(child, "feature_names", None)
    if feature_names is not None and len(feature_names) == width:
        return [str(column) for column in feature_names]

    get_names = getattr(preprocess, "get_feature_names_out", None)
    if callable(get_names):
        try:
            names = [str(column) for column in get_names()]
        except (AttributeError, TypeError, ValueError):
            names = []
        if len(names) == width:
            return names

    return [f"feature_{index}" for index in range(width)]


def _preprocess_local_rows(child: Any, rows: pd.DataFrame) -> tuple[pd.DataFrame, Any]:
    """Apply the fitted preprocessing pipeline to request-scoped rows."""

    raw_features = _shared_columns(child, "all_cols")
    if not raw_features:
        raise ValueError("The fitted model does not expose its raw feature columns.")
    missing = sorted(set(raw_features).difference(rows.columns))
    if missing:
        raise ValueError(f"SHAP data is missing required columns: {missing}")

    model = getattr(child, "model", None)
    if model is None:
        raise ValueError("The fitted model pipeline is unavailable for SHAP calculation.")
    try:
        preprocess = model["preprocess"]
        predictor = model["predictor"]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "The fitted model must expose preprocess and predictor pipeline steps."
        ) from exc

    transformed = preprocess.transform(rows[raw_features])
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    array = np.asarray(transformed)
    if array.ndim != 2:
        raise ValueError("The fitted preprocessor must return a two-dimensional matrix.")
    names = _transformed_feature_names(child, preprocess, array.shape[1])
    return pd.DataFrame(array, columns=names, index=rows.index), predictor


def _training_background(child: Any, columns: list[str]) -> pd.DataFrame:
    """Return transformed training data used as the local-SHAP background."""

    background = getattr(child, "df_prerpocessed", None)
    if not isinstance(background, pd.DataFrame) or background.empty:
        raise ValueError(
            "Transformed training data is unavailable for local SHAP calculation."
        )
    if len(background.columns) != len(columns):
        raise ValueError(
            "Training and prediction preprocessing produced different feature counts."
        )
    frame = background.copy()
    frame.columns = columns
    return frame


def _local_explainer_store(service: Any) -> dict[tuple[str, str], dict[str, Any]]:
    """Return a process-local cache containing explainers but no SHAP values."""

    store = getattr(service, "_local_shap_explainers", None)
    if store is None:
        store = {}
        service._local_shap_explainers = store
    return store


def _resolve_local_explainer(
    service: Any,
    model_id: str,
    target: str,
    child: Any,
    predictor: Any,
    background: pd.DataFrame,
) -> Any:
    """Reuse a fitted explainer or build one from the training-data background."""

    store = _local_explainer_store(service)
    cache_key = (model_id, target)
    cached = store.get(cache_key)
    if cached is not None and cached.get("model_identity") == id(child):
        return cached["explainer"]

    explainer = getattr(child, "explainer", None)
    if not callable(explainer):
        from malchan.models.explainability import get_shap_values

        _, _, explainer, _ = get_shap_values(predictor, background)
    if not callable(explainer):
        raise ValueError(
            f"SHAP is unavailable for the fitted predictor of target {target!r}."
        )

    store[cache_key] = {
        "model_identity": id(child),
        "explainer": explainer,
    }
    return explainer


def _evaluate_local_explainer(explainer: Any, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate an existing SHAP explainer for the explicitly selected rows."""

    try:
        explanation = explainer(frame, check_additivity=False)
    except TypeError:
        max_evals = max(2 * frame.shape[1] + 1, 1000)
        explanation = explainer(frame, max_evals=max_evals)
    values = np.asarray(getattr(explanation, "values", None))
    base_values = np.asarray(getattr(explanation, "base_values", None))
    if values.size == 0:
        raise ValueError("The SHAP explainer returned no values.")
    return values, base_values


def _output_names(child: Any, target: str, count: int) -> list[str]:
    """Return class labels or stable output names for local SHAP matrices."""

    labels = getattr(child, "target_items", None)
    if labels is not None:
        values = [str(value) for value in list(labels)]
        if len(values) == count:
            return values
    if count == 1:
        return [target]
    return [f"{target}_{index}" for index in range(count)]


def _matrix_json(values: np.ndarray) -> list[list[float | None]]:
    """Serialize a numeric matrix while converting NaN and infinities to null."""

    return json.loads(pd.DataFrame(values).to_json(orient="values"))


def _base_json(values: np.ndarray, row_count: int) -> list[float | None]:
    """Normalize one output's base values to one scalar per request row."""

    flat = np.asarray(values).reshape(-1)
    if flat.size == 0:
        flat = np.full(row_count, np.nan)
    elif flat.size == 1 and row_count > 1:
        flat = np.repeat(flat, row_count)
    elif flat.size != row_count:
        flat = np.resize(flat, row_count)
    return json.loads(pd.Series(flat).to_json(orient="values"))


def _local_target_shap(
    service: Any,
    model_id: str,
    child: Any,
    target: str,
    rows: pd.DataFrame,
) -> LocalShapTargetResponse:
    """Calculate selected-row SHAP values against the training-data background."""

    transformed, predictor = _preprocess_local_rows(child, rows)
    background = _training_background(child, list(transformed.columns))
    explainer = _resolve_local_explainer(
        service,
        model_id,
        target,
        child,
        predictor,
        background,
    )
    shap_batches: dict[str, list[np.ndarray]] = {}
    base_batches: dict[str, list[np.ndarray]] = {}
    output_names: list[str] | None = None
    transformed_batches: list[pd.DataFrame] = []

    for start in range(0, len(transformed), 300):
        batch = transformed.iloc[start : start + 300]
        values, base_array = _evaluate_local_explainer(explainer, batch)
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        if values.ndim not in {2, 3}:
            raise ValueError(
                f"Unsupported SHAP value shape for target {target!r}: {values.shape}"
            )

        output_count = 1 if values.ndim == 2 else values.shape[2]
        names = _output_names(child, target, output_count)
        if output_names is None:
            output_names = names
            shap_batches = {name: [] for name in names}
            base_batches = {name: [] for name in names}
        elif names != output_names:
            raise ValueError("SHAP output names changed between request batches.")

        transformed_batches.append(batch.reset_index(drop=True))
        for output_index, output_name in enumerate(names):
            matrix = values if values.ndim == 2 else values[:, :, output_index]
            shap_batches[output_name].append(np.asarray(matrix, dtype=float))

            if output_count == 1:
                selected_base = base_array
            elif base_array.ndim >= 2 and base_array.shape[-1] == output_count:
                selected_base = base_array[..., output_index]
            elif base_array.ndim == 1 and base_array.size == output_count:
                selected_base = np.repeat(base_array[output_index], len(batch))
            else:
                selected_base = np.full(len(batch), np.nan)
            base_batches[output_name].append(np.asarray(selected_base))

    if output_names is None:
        raise ValueError("No rows were available for SHAP calculation.")

    transformed_frame = pd.concat(transformed_batches, ignore_index=True)
    records = json.loads(
        transformed_frame.to_json(orient="records", date_format="iso")
    )
    shap_json = {
        output_name: _matrix_json(np.concatenate(shap_batches[output_name], axis=0))
        for output_name in output_names
    }
    base_json = {
        output_name: _base_json(
            np.concatenate(base_batches[output_name], axis=0),
            len(transformed_frame),
        )
        for output_name in output_names
    }
    return LocalShapTargetResponse(
        target=target,
        features=[str(column) for column in transformed_frame.columns],
        output_names=output_names,
        records=records,
        shap_values=shap_json,
        base_values=base_json,
    )


def compute_local_shap(
    service: Any,
    model_id: str,
    request: LocalShapRequest,
) -> LocalShapResponse:
    """Calculate SHAP values only for rows explicitly submitted by the caller."""

    registered = service._get_registered(model_id)
    target_models = _target_models(registered)
    selected_targets = request.targets or list(registered.info.target_cols)
    unknown = sorted(set(selected_targets).difference(target_models))
    if unknown:
        raise ValueError(f"Local SHAP targets contain unknown targets: {unknown}")

    rows = pd.DataFrame.from_records(request.data)
    required = set(registered.info.feature_columns)
    missing = sorted(required.difference(rows.columns))
    if missing:
        raise ValueError(f"SHAP data is missing required columns: {missing}")
    rows = rows[registered.info.feature_columns]

    targets = {
        target: _local_target_shap(
            service,
            model_id,
            target_models[target],
            target,
            rows,
        )
        for target in selected_targets
    }
    return LocalShapResponse(
        model_id=model_id,
        row_count=len(rows),
        targets=targets,
    )


def install_xai_shap_service(service_cls: type[Any]) -> None:
    """Attach cached and request-scoped SHAP operations to a model service class."""

    service_cls.get_xai_shap_values = get_xai_shap_values
    service_cls.compute_local_shap = compute_local_shap


__all__ = ["install_xai_shap_service"]
