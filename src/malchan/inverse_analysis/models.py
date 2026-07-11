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

    def predict(
        self,
        X: pd.DataFrame | None = None,
        proba: bool = False,
        idx2item: bool = False,
    ) -> pd.DataFrame:
        """Delegate model prediction through the normalized interface."""

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
        """Call the appropriate objective-prediction signature."""

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

    bounds_min, bounds_max, steps, dtypes, fix_values = default_settings(
        normalized_model,
        bounds_min,
        bounds_max,
        steps,
        dtypes,
        fix_values,
    )

    df_range, y_cols, x_cols, obj_cols, obj_values, directions = search_setting(
        normalized_model,
        obj_directions,
        bounds_min,
        bounds_max,
        dtypes,
        steps,
        fix_values,
        target_cols,
    )

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
