"""Optuna-based inverse analysis entry point."""

from typing import Any

import optuna
import pandas as pd

from .utils import default_settings, get_result, get_sampler, objective, search_setting


class _InverseAnalysisModelView:
    """Expose one stable interface for single-output and multi-output models."""

    def __init__(self, model: Any) -> None:
        """Resolve training data and metadata from local or shared storage."""

        self._model = model
        target_cols = getattr(model, "target_cols", None)
        if target_cols is None:
            target_col = getattr(model, "target_col", None)
            target_cols = [] if target_col is None else [target_col]
        self.target_cols = list(target_cols)
        if not self.target_cols:
            raise ValueError("No target columns are available for inverse analysis.")

        self.active_target_cols = list(self.target_cols)
        self.num_cols = self._resolve_columns("num_cols")
        self.cat_cols = self._resolve_columns("cat_cols")
        self.smiles_cols = self._resolve_columns("smiles_cols")
        self.comp_cols = self._resolve_columns("comp_cols")
        self.X = self._resolve_training_data()

    def _resolve_columns(self, name: str) -> list[str]:
        """Resolve one feature-column group from a model or shared context."""

        value = getattr(self._model, name, None)
        context = getattr(self._model, "context", None)
        if value is None and context is not None:
            value = getattr(context, name, None)
        return [] if value is None else list(value)

    def _resolve_training_data(self) -> pd.DataFrame:
        """Return training features required for automatic search settings."""

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

    def _child_model(self, target: str) -> Any:
        """Return the fitted single-output model for one target."""

        models = getattr(self._model, "models", None)
        if models is None:
            return self._model
        try:
            return models[target]
        except KeyError as exc:
            raise ValueError(f"No fitted model exists for target {target!r}.") from exc

    def _normalize_class_objective(self, target: str, value: Any) -> Any:
        """Validate a class objective and convert it to a probability-column suffix."""

        child = self._child_model(target)
        if getattr(child, "task", None) != "classification" or value is None:
            return value

        target_items = getattr(child, "target_items", None)
        if target_items is not None:
            matches = [
                item
                for item in list(target_items)
                if item == value or str(item) == str(value)
            ]
            if not matches:
                raise ValueError(
                    f"Unknown class label {value!r} for target {target!r}."
                )
        return str(value)

    def set_active_targets(self, target_cols: list[str]) -> None:
        """Set the target subset and order used during one inverse-analysis run."""

        if not target_cols:
            raise ValueError("At least one inverse-analysis target is required.")
        if len(target_cols) != len(set(target_cols)):
            raise ValueError("Inverse-analysis targets must not contain duplicates.")
        unknown = sorted(set(target_cols).difference(self.target_cols))
        if unknown:
            raise ValueError(f"Unknown inverse-analysis targets: {unknown}")
        self.active_target_cols = list(target_cols)

    def validate_objectives(
        self,
        target_cols: list[str],
        objectives: list[Any],
    ) -> None:
        """Validate target alignment and classification objectives."""

        self.set_active_targets(target_cols)
        if len(target_cols) != len(objectives):
            raise ValueError("target_cols and obj_directions must have the same length.")
        for target, value in zip(target_cols, objectives):
            child = self._child_model(target)
            if (
                getattr(child, "task", None) == "classification"
                and isinstance(value, str)
                and value in {"min", "max"}
            ):
                raise ValueError(
                    f"Classification target {target!r} requires a fitted class "
                    "label rather than 'min' or 'max'."
                )
            self._normalize_class_objective(target, value)

    def predict(
        self,
        X: pd.DataFrame | None = None,
        proba: bool = False,
        idx2item: bool = False,
    ) -> pd.DataFrame:
        """Predict only the active inverse-analysis targets in their chosen order."""

        predictions = []
        for target in self.active_target_cols:
            child = self._child_model(target)
            predictions.append(
                child.predict(
                    X=X,
                    proba=proba and getattr(child, "task", None) == "classification",
                    idx2item=idx2item,
                )
            )
        return pd.concat(predictions, axis=1)

    def predict_objective(
        self,
        X: pd.DataFrame | None = None,
        obj_values: list[Any] | None = None,
    ) -> pd.DataFrame:
        """Evaluate objective values for the active target subset."""

        resolved_values = [None] * len(self.active_target_cols)
        if obj_values is not None:
            resolved_values = list(obj_values)
        if len(resolved_values) != len(self.active_target_cols):
            raise ValueError(
                "Objective values must align with the active inverse-analysis targets."
            )

        objectives = []
        for target, value in zip(self.active_target_cols, resolved_values):
            child = self._child_model(target)
            normalized_value = self._normalize_class_objective(target, value)
            objectives.append(
                child.predict_objective(
                    X=X,
                    obj_value=normalized_value,
                )
            )
        return pd.concat(objectives, axis=1)


def _normalize_integer_search_settings(
    bounds_min: list[float],
    bounds_max: list[float],
    steps: list[float | None],
    dtypes: list[str],
) -> tuple[list[float | int], list[float | int], list[float | int | None]]:
    """Validate and cast integer search settings for Optuna.

    Args:
        bounds_min: Lower numeric bounds.
        bounds_max: Upper numeric bounds.
        steps: Optional numeric search steps.
        dtypes: Optuna search types aligned with the numeric columns.

    Returns:
        Bounds and steps with integer-search entries converted to ``int``.

    Raises:
        ValueError: If an integer search setting contains a fractional value.
    """

    normalized_min: list[float | int] = []
    normalized_max: list[float | int] = []
    normalized_steps: list[float | int | None] = []
    for index, dtype in enumerate(dtypes):
        lower = bounds_min[index]
        upper = bounds_max[index]
        step = steps[index]
        if dtype == "int":
            values = [lower, upper, step]
            if any(
                value is not None and not float(value).is_integer()
                for value in values
            ):
                raise ValueError(
                    "Integer inverse-analysis bounds and steps must be integral."
                )
            lower = int(lower)
            upper = int(upper)
            step = 1 if step is None else int(step)
        normalized_min.append(lower)
        normalized_max.append(upper)
        normalized_steps.append(step)
    return normalized_min, normalized_max, normalized_steps


def inverse_analysis(
    model: Any,
    sampler_type: str = "TPE",
    cat_dict: dict[str, list[Any]] | None = None,
    constraint_cols: list[str] | None = None,
    constraint_value: float = 0.0,
    obj_directions: list[Any] | None = None,
    bounds_min: list[float] | None = None,
    bounds_max: list[float] | None = None,
    dtypes: list[str] | None = None,
    steps: list[float | None] | None = None,
    fix_values: list[Any] | None = None,
    target_cols: list[str] | None = None,
    trials: int = 250,
    n_candidate: int = 10,
) -> tuple[pd.DataFrame, optuna.Study]:
    """Run inverse analysis and return ranked candidates and the Optuna study.

    Args:
        model: Fitted single-output or multi-output malchan model.
        sampler_type: Optuna sampler name.
        cat_dict: Candidate values keyed by categorical feature name.
        constraint_cols: Numeric feature columns whose sum is constrained.
        constraint_value: Required sum for ``constraint_cols``.
        obj_directions: Per-target objective definitions. Use ``"min"`` or
            ``"max"`` for directional optimization, or a target value/class
            label to minimize distance from that target.
        bounds_min: Numeric lower bounds aligned with ``model.num_cols``.
        bounds_max: Numeric upper bounds aligned with ``model.num_cols``.
        dtypes: Numeric search types (``"float"`` or ``"int"``).
        steps: Optional numeric search steps aligned with ``model.num_cols``.
        fix_values: Fixed feature values aligned with all input columns.
        target_cols: Target columns included in the inverse analysis.
        trials: Number of Optuna trials.
        n_candidate: Maximum number of ranked candidates to return.

    Returns:
        Ranked candidate dataframe and the completed Optuna study.
    """

    normalized_model = _InverseAnalysisModelView(model)
    cat_dict = {} if cat_dict is None else dict(cat_dict)
    constraint_cols = [] if constraint_cols is None else list(constraint_cols)
    obj_directions = [] if obj_directions is None else list(obj_directions)
    bounds_min = [] if bounds_min is None else list(bounds_min)
    bounds_max = [] if bounds_max is None else list(bounds_max)
    dtypes = [] if dtypes is None else list(dtypes)
    steps = [] if steps is None else list(steps)
    fix_values = [] if fix_values is None else list(fix_values)
    target_cols = [] if target_cols is None else list(target_cols)

    resolved_targets = normalized_model.target_cols if not target_cols else target_cols
    normalized_model.validate_objectives(resolved_targets, obj_directions)

    bounds_min, bounds_max, steps, dtypes, fix_values = default_settings(
        normalized_model,
        bounds_min,
        bounds_max,
        steps,
        dtypes,
        fix_values,
    )
    bounds_min, bounds_max, steps = _normalize_integer_search_settings(
        bounds_min,
        bounds_max,
        steps,
        dtypes,
    )

    df_range, y_cols, x_cols, obj_cols, obj_values, directions = search_setting(
        normalized_model,
        obj_directions,
        bounds_min,
        bounds_max,
        dtypes,
        steps,
        fix_values,
        resolved_targets,
    )
    normalized_model.set_active_targets(y_cols)

    sampler = get_sampler(sampler_type)
    if sampler is None:
        raise ValueError(f"Unsupported sampler_type: {sampler_type!r}")

    study = optuna.create_study(directions=directions, sampler=sampler)
    study.optimize(
        lambda trial: objective(
            trial,
            model=normalized_model,
            cat_dict=cat_dict,
            y_cols=y_cols,
            obj_values=obj_values,
            df_range=df_range,
            x_cols=x_cols,
            constraint_cols=constraint_cols,
            constraint_value=constraint_value,
        ),
        n_trials=trials,
    )

    _, df_trials = get_result(
        normalized_model,
        study,
        df_range,
        constraint_cols,
        constraint_value,
        x_cols,
        y_cols,
        obj_cols,
        fix_values,
    )

    objective_frame = df_trials[y_cols].copy()
    for index, column in enumerate(y_cols):
        if directions[index] == "maximize":
            objective_frame[column] = -objective_frame[column]
        value_range = objective_frame[column].max() - objective_frame[column].min()
        if value_range == 0 or pd.isna(value_range):
            objective_frame[column] = 0.0
        else:
            objective_frame[column] = (
                objective_frame[column] - objective_frame[column].min()
            ) / value_range

    df_trials = df_trials.copy()
    df_trials["obj_score"] = objective_frame[y_cols].sum(axis=1).values
    candidates = (
        df_trials.sort_values("obj_score", ascending=True)
        .reset_index(drop=True)
        .head(n_candidate)
        .drop(columns=y_cols + ["obj_score"])
    )
    return candidates, study
