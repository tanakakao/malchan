"""Runtime extensions adding analysis methods to public pipeline classes.

The core pipeline implementations remain focused on fitting, prediction, and
serialization. This module attaches higher-level analysis entry points to those
same class objects when :mod:`malchan.pipeline` is imported, so existing import
paths and checkpoint behavior remain unchanged.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def _is_direction(value: Any) -> bool:
    """Return whether an inverse-analysis objective is ``min`` or ``max``."""

    return isinstance(value, str) and value in {"min", "max"}


def single_output_inverse_analysis(
    self: Any,
    objective: Any,
    **kwargs: Any,
) -> tuple[Any, Any]:
    """Run inverse analysis for a fitted single-output pipeline."""

    if getattr(self, "target_col", None) in (None, "AD"):
        raise ValueError("Fit a supervised model before calling inverse_analysis().")
    if getattr(self, "task", None) == "classification" and _is_direction(objective):
        raise ValueError(
            "Classification inverse analysis requires a fitted class label as "
            "the objective."
        )
    if "obj_directions" in kwargs or "target_cols" in kwargs:
        raise ValueError(
            "Use the objective argument for a single-output model; "
            "obj_directions and target_cols are managed automatically."
        )

    from ..inverse_analysis import inverse_analysis

    candidates, study = inverse_analysis(
        self,
        obj_directions=[objective],
        target_cols=[self.target_col],
        **kwargs,
    )
    self.inverse_candidates = candidates
    self.inverse_study = study
    return candidates, study


def multi_output_inverse_analysis(
    self: Any,
    objectives: Mapping[str, Any] | Sequence[Any],
    *,
    target_cols: Sequence[str] | None = None,
    **kwargs: Any,
) -> tuple[Any, Any]:
    """Run inverse analysis for selected outputs of a fitted multi-output model."""

    fitted_targets = list(getattr(self, "target_cols", None) or [])
    if not fitted_targets:
        raise ValueError("Fit a multi-output model before calling inverse_analysis().")
    if "obj_directions" in kwargs:
        raise ValueError(
            "Use the objectives argument; obj_directions is managed automatically."
        )

    if isinstance(objectives, Mapping):
        if target_cols is not None:
            raise ValueError(
                "target_cols must be omitted when objectives is a mapping."
            )
        resolved_targets = list(objectives)
        resolved_objectives = [objectives[target] for target in resolved_targets]
    else:
        resolved_targets = fitted_targets if target_cols is None else list(target_cols)
        resolved_objectives = list(objectives)

    if not resolved_targets:
        raise ValueError("At least one inverse-analysis target is required.")
    if len(resolved_targets) != len(resolved_objectives):
        raise ValueError("target_cols and objectives must have the same length.")
    if len(resolved_targets) != len(set(resolved_targets)):
        raise ValueError("Inverse-analysis target columns must not contain duplicates.")

    unknown_targets = sorted(set(resolved_targets).difference(fitted_targets))
    if unknown_targets:
        raise ValueError(f"Unknown inverse-analysis targets: {unknown_targets}")

    task_by_target = dict(zip(fitted_targets, self.tasks, strict=True))
    for target, objective in zip(resolved_targets, resolved_objectives, strict=True):
        if task_by_target[target] == "classification" and _is_direction(objective):
            raise ValueError(
                f"Classification target {target!r} requires a fitted class label "
                "as its objective."
            )

    from ..inverse_analysis import inverse_analysis

    candidates, study = inverse_analysis(
        self,
        obj_directions=resolved_objectives,
        target_cols=resolved_targets,
        **kwargs,
    )
    self.inverse_candidates = candidates
    self.inverse_study = study
    return candidates, study


def single_output_compare(
    self: Any,
    model_names: Sequence[str] | None = None,
    *,
    df: Any | None = None,
    model_params: Mapping[str, Mapping[str, Any] | None] | None = None,
    method: str = "kfold",
    n_splits: int = 5,
    metric: str | None = None,
    tuning: bool = False,
    tune_best: bool = False,
    tuning_trials: int = 30,
    tuning_verbose: int = 0,
    continue_on_error: bool = True,
    **fit_kwargs: Any,
) -> Any:
    """Fit and rank single-output candidates, optionally from raw data.

    Calling this method on an unfitted pipeline requires ``df`` and the
    remaining arguments accepted by :meth:`SingleOutputMLModelPipeline.fit`,
    such as ``target_col``, ``task``, ``num_cols``, and ``cat_cols``.  The
    comparison candidates in ``model_names`` are also used to initialize the
    pipeline.  On a fitted pipeline, data and fitting options must be omitted
    and the existing fitted configuration is used unchanged.

    Args:
        model_names: Candidate estimator names. ``None`` compares all models.
        df: Raw training dataframe when the pipeline has not been fitted.
        model_params: Optional estimator parameters keyed by model name.
        method: Cross-validation method, ``"kfold"`` or ``"loo"``.
        n_splits: Number of K-fold splits.
        metric: Ranking metric. Defaults to RMSE or F1.
        tuning: Whether to tune every candidate before ranking.
        tune_best: Whether to tune only the selected best candidate.
        tuning_trials: Number of tuning trials.
        tuning_verbose: OptunaSearchCV verbosity.
        continue_on_error: Whether to retain failures and compare remaining candidates.
        **fit_kwargs: Remaining arguments accepted by :meth:`fit` when ``df``
            is supplied, including columns and preprocessing settings.

    Returns:
        A single-output model comparison result.

    Raises:
        ValueError: If raw data is absent before fitting or supplied after fitting.
    """

    fitted = getattr(self, "target_col", None) not in (None, "AD")
    if not fitted:
        if df is None:
            raise ValueError("df is required when calling compare() before fit().")
        task = fit_kwargs.get("task")
        if task is None:
            raise ValueError("task is required when calling compare() before fit().")
        if model_names is None:
            from ..models.compare import _available_model_names

            initial_model_names = [_available_model_names(task)[0]]
        else:
            initial_model_names = list(model_names)
        self.fit(df=df, model_names=initial_model_names, **fit_kwargs)
    elif df is not None or fit_kwargs:
        raise ValueError("Data and fitting options can only be supplied before fit().")

    from ..models.compare import compare_single_output_model

    result = compare_single_output_model(
        self,
        model_names=model_names,
        model_params=model_params,
        method=method,
        n_splits=n_splits,
        metric=metric,
        tuning=tuning,
        tune_best=tune_best,
        tuning_trials=tuning_trials,
        tuning_verbose=tuning_verbose,
        continue_on_error=continue_on_error,
    )
    self.comparison_result = result
    return result


def multi_output_compare(
    self: Any,
    model_names: Sequence[str] | Mapping[str, Sequence[str]] | None = None,
    *,
    df: Any | None = None,
    model_params: Mapping[str, Any] | None = None,
    method: str = "kfold",
    n_splits: int = 5,
    metric: str | Mapping[str, str] | None = None,
    tuning: bool = False,
    tune_best: bool = False,
    tuning_trials: int | Mapping[str, int] = 30,
    tuning_verbose: int = 0,
    continue_on_error: bool = True,
    **fit_kwargs: Any,
) -> Any:
    """Compare multi-output candidates, optionally fitting from raw data first.

    On an unfitted pipeline, provide ``df`` plus the remaining arguments of
    :meth:`MLModelPipeline.fit`, including ``target_cols``, ``tasks``, feature
    columns, and preprocessing settings.  ``model_names`` selects comparison
    candidates and initializes each target with its first candidate.  Fitted
    pipelines retain their existing behavior and do not accept fitting inputs.

    Args:
        model_names: Shared or per-target candidate estimator names.
        df: Raw training dataframe when the pipeline has not been fitted.
        model_params: Shared or per-target estimator parameters.
        method: Cross-validation method, ``"kfold"`` or ``"loo"``.
        n_splits: Number of K-fold splits.
        metric: Shared or per-target ranking metric.
        tuning: Whether to tune every candidate before ranking.
        tune_best: Whether to tune only each selected best candidate.
        tuning_trials: Shared or per-target number of tuning trials.
        tuning_verbose: OptunaSearchCV verbosity.
        continue_on_error: Whether to retain failures and compare remaining candidates.
        **fit_kwargs: Remaining arguments accepted by :meth:`fit` when ``df``
            is supplied, including columns and preprocessing settings.

    Returns:
        A multi-output model comparison result.

    Raises:
        ValueError: If raw data is absent before fitting or supplied after fitting.
    """

    fitted = bool(getattr(self, "target_cols", None))
    if not fitted:
        if df is None:
            raise ValueError("df is required when calling compare() before fit().")
        target_cols = fit_kwargs.get("target_cols")
        tasks = fit_kwargs.get("tasks")
        if target_cols is None or tasks is None:
            raise ValueError(
                "target_cols and tasks are required when calling compare() before fit()."
            )
        target_cols = list(target_cols)
        task_list = [tasks] * len(target_cols) if isinstance(tasks, str) else list(tasks)
        if len(target_cols) != len(task_list):
            raise ValueError("target_cols and tasks must have the same length.")

        from ..models.compare import _available_model_names

        initial_model_names = []
        for target, task in zip(target_cols, task_list, strict=True):
            candidates = (
                model_names.get(target)
                if isinstance(model_names, Mapping)
                else model_names
            )
            initial_model_names.append(
                [_available_model_names(task)[0]] if candidates is None else list(candidates)
            )
        self.fit(df=df, model_names=initial_model_names, **fit_kwargs)
    elif df is not None or fit_kwargs:
        raise ValueError("Data and fitting options can only be supplied before fit().")

    from ..models.compare import compare_multi_output_model

    result = compare_multi_output_model(
        self,
        model_names=model_names,
        model_params=model_params,
        method=method,
        n_splits=n_splits,
        metric=metric,
        tuning=tuning,
        tune_best=tune_best,
        tuning_trials=tuning_trials,
        tuning_verbose=tuning_verbose,
        continue_on_error=continue_on_error,
    )
    self.comparison_result = result
    return result


def install_analysis_extensions(
    single_output_cls: type[Any],
    multi_output_cls: type[Any],
) -> None:
    """Attach model-native analysis methods without replacing class identity."""

    single_output_cls.inverse_analysis = single_output_inverse_analysis
    single_output_cls.compare = single_output_compare
    multi_output_cls.inverse_analysis = multi_output_inverse_analysis
    multi_output_cls.compare = multi_output_compare


__all__ = [
    "install_analysis_extensions",
    "multi_output_compare",
    "multi_output_inverse_analysis",
    "single_output_compare",
    "single_output_inverse_analysis",
]
