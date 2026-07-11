"""In-memory model training, prediction, and inverse-analysis service."""

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from threading import RLock
from typing import Any, Callable
from uuid import uuid4

import pandas as pd
from pandas.api.types import is_integer_dtype

from malchan.app.schemas import (
    InverseAnalysisRequest,
    InverseAnalysisResponse,
    ModelInfo,
    PredictRequest,
    TrainModelRequest,
)


class ModelNotFoundError(KeyError):
    """Raised when a requested model identifier is not registered."""


@dataclass(slots=True)
class _RegisteredModel:
    """Internal model object and its API metadata."""

    model: Any
    info: ModelInfo


class _InverseAnalysisModelAdapter:
    """Normalize single-output and shared-context models for inverse analysis."""

    def __init__(self, model: Any, info: ModelInfo) -> None:
        """Expose the attributes expected by ``malchan.inverse_analysis``."""

        self._model = model
        self.target_cols = list(info.target_cols)
        self.num_cols = self._resolve_columns("num_cols")
        self.cat_cols = self._resolve_columns("cat_cols")
        self.smiles_cols = self._resolve_columns("smiles_cols")
        self.comp_cols = self._resolve_columns("comp_cols")
        self.X = self._resolve_training_data()

    def _resolve_columns(self, name: str) -> list[str]:
        """Read feature metadata from a model or its shared context."""

        value = getattr(self._model, name, None)
        context = getattr(self._model, "context", None)
        if value is None and context is not None:
            value = getattr(context, name, None)
        return [] if value is None else list(value)

    def _resolve_training_data(self) -> pd.DataFrame:
        """Return training features from local or shared storage."""

        training_data = getattr(self._model, "X", None)
        context = getattr(self._model, "context", None)
        if training_data is None and context is not None:
            training_data = getattr(context, "X", None)
        if training_data is None and hasattr(self._model, "_get_X"):
            training_data = self._model._get_X()
        if training_data is None:
            raise ValueError(
                "Training features are unavailable for inverse analysis."
            )
        return training_data

    def predict(
        self,
        X: pd.DataFrame | None = None,
        proba: bool = False,
        idx2item: bool = False,
    ) -> pd.DataFrame:
        """Delegate prediction to the registered model."""

        return self._model.predict(
            X=X,
            proba=proba,
            idx2item=idx2item,
        )

    def predict_objective(
        self,
        X: pd.DataFrame | None = None,
        obj_values: list[Any] | None = None,
    ) -> pd.DataFrame:
        """Delegate objective prediction using the correct model signature."""

        resolved_values = [None] * len(self.target_cols)
        if obj_values is not None:
            resolved_values = list(obj_values)

        if getattr(self._model, "target_cols", None) is not None:
            return self._model.predict_objective(
                X=X,
                obj_values=resolved_values,
            )
        return self._model.predict_objective(
            X=X,
            obj_value=resolved_values[0],
        )


class InMemoryModelService:
    """Train and serve models for the lifetime of one Python process."""

    def __init__(
        self,
        model_factory: Callable[[], Any] | None = None,
        multi_model_factory: Callable[[], Any] | None = None,
        id_factory: Callable[[], str] | None = None,
        inverse_analysis_func: Callable[..., tuple[pd.DataFrame, Any]] | None = None,
    ) -> None:
        """Initialize an empty registry.

        Args:
            model_factory: Optional single-output pipeline factory used mainly
                for tests.
            multi_model_factory: Optional multi-output pipeline factory used
                mainly for tests.
            id_factory: Optional model identifier factory used mainly for tests.
            inverse_analysis_func: Optional inverse-analysis implementation used
                mainly for tests.
        """

        self._model_factory = model_factory or self._default_model_factory
        self._multi_model_factory = (
            multi_model_factory or self._default_multi_model_factory
        )
        self._id_factory = id_factory or (lambda: str(uuid4()))
        self._inverse_analysis_func = inverse_analysis_func
        self._models: dict[str, _RegisteredModel] = {}
        self._lock = RLock()

    @staticmethod
    def _default_model_factory() -> Any:
        """Create the public single-output malchan pipeline lazily."""

        from malchan.models import SingleOutputMLModelPipeline

        return SingleOutputMLModelPipeline()

    @staticmethod
    def _default_multi_model_factory() -> Any:
        """Create the public multi-output malchan pipeline lazily."""

        from malchan.models import MLModelPipeline

        return MLModelPipeline()

    def _get_inverse_analysis_func(
        self,
    ) -> Callable[..., tuple[pd.DataFrame, Any]]:
        """Resolve the optional inverse-analysis dependency lazily."""

        if self._inverse_analysis_func is None:
            from malchan.inverse_analysis import inverse_analysis

            self._inverse_analysis_func = inverse_analysis
        return self._inverse_analysis_func

    def train(self, request: TrainModelRequest) -> ModelInfo:
        """Fit and register one single-output or multi-output model."""

        dataframe = pd.DataFrame.from_records(request.data)
        target_cols = request.resolved_target_cols
        tasks = request.resolved_tasks
        model_names_by_target = request.model_names_for_targets
        aligned_model_names = [
            model_names_by_target[target] for target in target_cols
        ]

        if request.is_multi_output:
            model = self._multi_model_factory()
            model.fit(
                df=dataframe,
                target_cols=target_cols,
                tasks=tasks,
                num_cols=request.num_cols,
                cat_cols=request.cat_cols,
                model_names=aligned_model_names,
                smiles_cols=request.smiles_cols,
                fingerprints=request.fingerprints,
                comp_cols=request.comp_cols,
                comp_method=request.comp_method,
                comp_feats=request.comp_feats,
                impute=request.impute,
                tunings=request.tuning,
                ensembles=request.ensemble,
                ens_types=request.ens_type,
                base_models=request.base_model,
                model_params=request.resolved_model_params,
                base_model_params=request.resolved_base_model_params,
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
        else:
            model = self._model_factory()
            model.fit(
                df=dataframe,
                target_col=target_cols[0],
                task=tasks[0],
                num_cols=request.num_cols,
                cat_cols=request.cat_cols,
                model_names=aligned_model_names[0],
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
                model_params=request.resolved_model_params[0],
                base_model_param=request.resolved_base_model_params[0],
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

        is_single_output = len(target_cols) == 1
        info = ModelInfo(
            model_id=self._id_factory(),
            target_cols=target_cols,
            tasks=tasks,
            feature_columns=request.feature_columns,
            model_names_by_target=model_names_by_target,
            created_at=datetime.now(timezone.utc),
            target_col=target_cols[0] if is_single_output else None,
            task=tasks[0] if is_single_output else None,
            model_names=aligned_model_names[0] if is_single_output else [],
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
        """Run single-output or multi-output inference."""

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

    def run_inverse_analysis(
        self,
        model_id: str,
        request: InverseAnalysisRequest,
    ) -> InverseAnalysisResponse:
        """Search feature candidates against a registered fitted model."""

        registered = self._get_registered(model_id)
        adapter = _InverseAnalysisModelAdapter(
            model=registered.model,
            info=registered.info,
        )
        target_set = set(registered.info.target_cols)
        objective_targets = [item.target for item in request.objectives]
        unknown_targets = sorted(set(objective_targets).difference(target_set))
        if unknown_targets:
            raise ValueError(
                f"Inverse-analysis objectives contain unknown targets: {unknown_targets}"
            )

        task_by_target = dict(
            zip(registered.info.target_cols, registered.info.tasks)
        )
        for objective in request.objectives:
            if (
                task_by_target[objective.target] == "classification"
                and objective.target_value is None
            ):
                raise ValueError(
                    "Classification objectives require target_value with a class label."
                )
            self._validate_class_target(
                registered.model,
                objective.target,
                objective.target_value,
            )

        all_features = [
            *adapter.num_cols,
            *adapter.cat_cols,
            *adapter.smiles_cols,
            *adapter.comp_cols,
        ]
        self._validate_inverse_feature_settings(
            request=request,
            numeric_features=adapter.num_cols,
            categorical_features=[
                *adapter.cat_cols,
                *adapter.smiles_cols,
                *adapter.comp_cols,
            ],
            all_features=all_features,
        )

        bounds_min: list[float] = []
        bounds_max: list[float] = []
        dtypes: list[str] = []
        steps: list[float | None] = []
        for column in adapter.num_cols:
            series = adapter.X[column]
            setting = request.bounds.get(column)
            lower = float(series.min()) if setting is None else setting.min
            upper = float(series.max()) if setting is None else setting.max
            dtype = (
                "int" if is_integer_dtype(series.dtype) else "float"
            ) if setting is None or setting.dtype is None else setting.dtype
            step = setting.step if setting is not None else None
            if dtype == "int" and step is None:
                step = 1
            bounds_min.append(lower)
            bounds_max.append(upper)
            dtypes.append(dtype)
            steps.append(step)

        categorical_features = [
            *adapter.cat_cols,
            *adapter.smiles_cols,
            *adapter.comp_cols,
        ]
        cat_dict: dict[str, list[Any]] = {}
        for column in categorical_features:
            values = request.categories.get(column)
            if values is None:
                values = adapter.X[column].dropna().drop_duplicates().tolist()
            if column not in request.fixed_values and not values:
                raise ValueError(
                    f"No category candidates are available for {column!r}."
                )
            cat_dict[column] = list(values)

        fix_values = [request.fixed_values.get(column) for column in all_features]
        constraint_cols: list[str] = []
        constraint_value = 0.0
        if request.sum_constraint is not None:
            constraint_cols = list(request.sum_constraint.columns)
            constraint_value = request.sum_constraint.value

        objective_values = [
            item.target_value
            if item.target_value is not None
            else item.direction
            for item in request.objectives
        ]
        inverse_analysis_func = self._get_inverse_analysis_func()
        candidates, study = inverse_analysis_func(
            model=adapter,
            sampler_type=request.sampler_type,
            cat_dict=cat_dict,
            constraint_cols=constraint_cols,
            constraint_value=constraint_value,
            obj_directions=objective_values,
            bounds_min=bounds_min,
            bounds_max=bounds_max,
            dtypes=dtypes,
            steps=steps,
            fix_values=fix_values,
            target_cols=objective_targets,
            trials=request.trials,
            n_candidate=request.n_candidates,
        )
        if not isinstance(candidates, pd.DataFrame):
            candidates = pd.DataFrame(candidates)

        trial_items = list(getattr(study, "trials", []))
        n_completed = sum(
            getattr(getattr(trial, "state", None), "name", "") == "COMPLETE"
            for trial in trial_items
        )
        pareto_size = len(getattr(study, "best_trials", []))
        candidate_records = json.loads(
            candidates.to_json(orient="records", date_format="iso")
        )
        return InverseAnalysisResponse(
            model_id=model_id,
            objectives=request.objectives,
            candidates=candidate_records,
            n_trials=len(trial_items),
            n_completed_trials=n_completed,
            pareto_size=pareto_size,
        )

    @staticmethod
    def _validate_inverse_feature_settings(
        request: InverseAnalysisRequest,
        numeric_features: list[str],
        categorical_features: list[str],
        all_features: list[str],
    ) -> None:
        """Validate request feature names against registered model metadata."""

        unknown_bounds = sorted(set(request.bounds).difference(numeric_features))
        if unknown_bounds:
            raise ValueError(
                f"Numeric bounds contain unknown columns: {unknown_bounds}"
            )
        unknown_categories = sorted(
            set(request.categories).difference(categorical_features)
        )
        if unknown_categories:
            raise ValueError(
                f"Category settings contain unknown columns: {unknown_categories}"
            )
        unknown_fixed = sorted(set(request.fixed_values).difference(all_features))
        if unknown_fixed:
            raise ValueError(
                f"Fixed values contain unknown columns: {unknown_fixed}"
            )
        if request.sum_constraint is not None:
            invalid_constraint_columns = sorted(
                set(request.sum_constraint.columns).difference(numeric_features)
            )
            if invalid_constraint_columns:
                raise ValueError(
                    "Sum constraints support numeric columns only: "
                    f"{invalid_constraint_columns}"
                )

    @staticmethod
    def _validate_class_target(
        model: Any,
        target: str,
        target_value: Any,
    ) -> None:
        """Validate a requested class label when class metadata is available."""

        if target_value is None:
            return
        child_model = model
        models = getattr(model, "models", None)
        if models is not None and target in models:
            child_model = models[target]
        target_items = getattr(child_model, "target_items", None)
        if target_items is None:
            return
        values = list(target_items)
        if target_value not in values:
            raise ValueError(
                f"Unknown class label {target_value!r} for target {target!r}."
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
