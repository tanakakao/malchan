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
    """Run inverse analysis for a fitted single-output pipeline.

    Args:
        objective: ``"min"`` or ``"max"`` for directional regression
            optimization, a numeric target value for regression target matching,
            or a fitted class label for classification.
        **kwargs: Remaining arguments accepted by
            :func:`malchan.inverse_analysis.inverse_analysis`, such as bounds,
            fixed values, constraints, sampler type, and trial count.

    Returns:
        Candidate dataframe and completed Optuna study.
    """

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
    """Run inverse analysis for selected outputs of a fitted multi-output model.

    Args:
        objectives: Prefer a mapping from target column to objective, for example
            ``{"strength": "max", "cost": "min"}``. A sequence is also
            accepted when ``target_cols`` supplies the aligned target order.
        target_cols: Target order used when ``objectives`` is a sequence. When
            omitted, all fitted target columns are used.
        **kwargs: Remaining inverse-analysis settings.

    Returns:
        Candidate dataframe and completed Optuna study.
    """

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

    task_by_target = dict(zip(fitted_targets, self.tasks))
    for target, objective in zip(resolved_targets, resolved_objectives):
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
    model_params: Mapping[str, Mapping[str, Any] | None] | None = None,
    method: str = "kfold",
    n_splits: int = 5,
    metric: str | None = None,
    tuning: bool = False,
    continue_on_error: bool = True,
) -> Any:
    """Fit and rank candidate estimators using one common CV configuration."""

    from ..models.compare import compare_single_output_model

    result = compare_single_output_model(
        self,
        model_names=model_names,
        model_params=model_params,
        method=method,
        n_splits=n_splits,
        metric=metric,
        tuning=tuning,
        continue_on_error=continue_on_error,
    )
    self.comparison_result = result
    return result


def multi_output_compare(
    self: Any,
    model_names: Sequence[str] | Mapping[str, Sequence[str]] | None = None,
    *,
    model_params: Mapping[str, Any] | None = None,
    method: str = "kfold",
    n_splits: int = 5,
    metric: str | Mapping[str, str] | None = None,
    tuning: bool = False,
    continue_on_error: bool = True,
) -> Any:
    """Compare candidate estimators independently for every fitted target."""

    from ..models.compare import compare_multi_output_model

    result = compare_multi_output_model(
        self,
        model_names=model_names,
        model_params=model_params,
        method=method,
        n_splits=n_splits,
        metric=metric,
        tuning=tuning,
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
