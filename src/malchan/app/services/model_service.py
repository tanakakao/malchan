"""In-memory model training and inference service."""

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from threading import RLock
from typing import Any, Callable
from uuid import uuid4

import pandas as pd

from malchan.app.schemas import ModelInfo, PredictRequest, TrainModelRequest


class ModelNotFoundError(KeyError):
    """Raised when a requested model identifier is not registered."""


@dataclass(slots=True)
class _RegisteredModel:
    """Internal model object and its API metadata."""

    model: Any
    info: ModelInfo


class InMemoryModelService:
    """Train and serve models for the lifetime of one Python process."""

    def __init__(
        self,
        model_factory: Callable[[], Any] | None = None,
        id_factory: Callable[[], str] | None = None,
    ) -> None:
        """Initialize an empty registry.

        Args:
            model_factory: Optional pipeline factory used mainly for tests.
            id_factory: Optional model identifier factory used mainly for tests.
        """

        self._model_factory = model_factory or self._default_model_factory
        self._id_factory = id_factory or (lambda: str(uuid4()))
        self._models: dict[str, _RegisteredModel] = {}
        self._lock = RLock()

    @staticmethod
    def _default_model_factory() -> Any:
        """Create the public single-output malchan pipeline lazily."""

        from malchan.models import SingleOutputMLModelPipeline

        return SingleOutputMLModelPipeline()

    def train(self, request: TrainModelRequest) -> ModelInfo:
        """Fit and register one model from tabular records."""

        model = self._model_factory()
        dataframe = pd.DataFrame.from_records(request.data)
        model.fit(
            df=dataframe,
            target_col=request.target_col,
            task=request.task,
            num_cols=request.num_cols,
            cat_cols=request.cat_cols,
            model_names=request.model_names,
            smiles_cols=request.smiles_cols,
            fingerprints=request.fingerprints,
            comp_cols=request.comp_cols,
            comp_method=request.comp_method,
            comp_feats=request.comp_feats,
            impute=request.impute,
            tuning=request.tuning,
            ensemble=request.ensemble,
            ens_type=request.ens_type,
            base_model=request.base_model,
            model_params=request.model_params,
            base_model_param=request.base_model_param,
            num_impute_type=request.num_impute_type,
            num_scale_type=request.num_scale_type,
            cat_impute=request.cat_impute,
            poly=request.poly,
            poly_degree=request.poly_degree,
            poly_interaction_only=request.poly_interaction_only,
            decomposition=request.decomposition,
            decomposition_method=request.decomposition_method,
            dec_n_components=request.dec_n_components,
            sampling_method=request.sampling_method,
        )

        info = ModelInfo(
            model_id=self._id_factory(),
            target_col=request.target_col,
            task=request.task,
            feature_columns=request.feature_columns,
            model_names=request.model_names,
            created_at=datetime.now(timezone.utc),
        )
        with self._lock:
            self._models[info.model_id] = _RegisteredModel(model=model, info=info)
        return info

    def list_models(self) -> list[ModelInfo]:
        """Return registered model metadata ordered by creation time."""

        with self._lock:
            return sorted(
                (entry.info for entry in self._models.values()),
                key=lambda item: item.created_at,
            )

    def get_model(self, model_id: str) -> ModelInfo:
        """Return metadata for one registered model."""

        return self._get_registered(model_id).info

    def predict(
        self,
        model_id: str,
        request: PredictRequest,
    ) -> list[dict[str, Any]]:
        """Run inference and return JSON-compatible records."""

        registered = self._get_registered(model_id)
        dataframe = pd.DataFrame.from_records(request.data)
        missing = sorted(
            set(registered.info.feature_columns).difference(dataframe.columns)
        )
        if missing:
            raise ValueError(
                f"Prediction data is missing required columns: {missing}"
            )

        predictions = registered.model.predict(
            X=dataframe[registered.info.feature_columns],
            proba=request.proba,
            idx2item=request.decode_labels,
        )
        if not isinstance(predictions, pd.DataFrame):
            predictions = pd.DataFrame(predictions)
        return json.loads(
            predictions.to_json(orient="records", date_format="iso")
        )

    def delete(self, model_id: str) -> None:
        """Remove one model from the process-local registry."""

        with self._lock:
            if model_id not in self._models:
                raise ModelNotFoundError(model_id)
            del self._models[model_id]

    def _get_registered(self, model_id: str) -> _RegisteredModel:
        """Resolve one internal registry entry."""

        with self._lock:
            try:
                return self._models[model_id]
            except KeyError as exc:
                raise ModelNotFoundError(model_id) from exc
