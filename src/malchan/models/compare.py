"""Model comparison utilities used by pipeline instance methods."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


@dataclass(slots=True)
class ModelComparisonResult:
    """Cross-validation ranking and fitted candidate models for one target."""

    ranking: pd.DataFrame
    models: dict[str, Any]
    failures: dict[str, str]
    metric: str
    higher_is_better: bool
    _source_model: Any = field(repr=False)
    method: str = "kfold"
    n_splits: int = 5

    @property
    def best_model_name(self) -> str | None:
        """Return the highest-ranked successful model name."""

        if self.ranking.empty:
            return None
        return str(self.ranking.iloc[0]["model_name"])

    @property
    def best_model(self) -> Any | None:
        """Return the fitted model corresponding to :attr:`best_model_name`."""

        name = self.best_model_name
        return None if name is None else self.models[name]

    @property
    def best_params(self) -> Any:
        """Return the current best model's fitted or tuned parameters."""

        model = self.best_model
        return None if model is None else getattr(model, "model_params", None)

    @property
    def best_is_tuned(self) -> bool:
        """Return whether the current best model has been parameter-tuned."""

        model = self.best_model
        return bool(model is not None and getattr(model, "tuning", False))

    @property
    def best_cv_scores(self) -> dict[str, pd.DataFrame] | None:
        """Return CV scores for the current best model."""

        model = self.best_model
        return None if model is None else getattr(model, "cv_scores", None)

    def tune_best(
        self,
        *,
        n_trials: int = 30,
        verbose: int = 0,
        evaluate: bool = True,
    ) -> Any:
        """Tune and replace the highest-ranked candidate model.

        The original ranking remains unchanged because it represents the fair,
        common-configuration comparison used to select the candidate family.
        When ``evaluate`` is true, the tuned model is evaluated with the same CV
        method and split count and its scores are available through
        :attr:`best_cv_scores`.

        Args:
            n_trials: Number of OptunaSearchCV trials.
            verbose: OptunaSearchCV verbosity.
            evaluate: Whether to run the original comparison CV after tuning.

        Returns:
            The tuned, fully fitted ``SingleOutputMLModelPipeline``.
        """

        if n_trials < 1:
            raise ValueError("n_trials must be at least 1.")
        name = self.best_model_name
        if name is None:
            raise RuntimeError("No successful comparison model is available to tune.")

        tuned_model = _fit_candidate(
            source_model=self._source_model,
            context=_comparison_context(self._source_model),
            model_name=name,
            model_params=None,
            tuning=True,
            tuning_trials=n_trials,
            tuning_verbose=verbose,
        )
        if evaluate:
            tuned_model.cv_score(method=self.method, n_splits=self.n_splits)
        self.models[name] = tuned_model
        return tuned_model


@dataclass(slots=True)
class MultiOutputModelComparisonResult:
    """Per-target comparison results for a multi-output pipeline."""

    results: dict[str, ModelComparisonResult]

    @property
    def ranking(self) -> pd.DataFrame:
        """Return all target rankings as one dataframe."""

        frames = [result.ranking for result in self.results.values()]
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    @property
    def best_model_names(self) -> dict[str, str | None]:
        """Return the highest-ranked model name for every target."""

        return {
            target: result.best_model_name
            for target, result in self.results.items()
        }

    @property
    def best_models(self) -> dict[str, Any | None]:
        """Return the fitted highest-ranked model for every target."""

        return {
            target: result.best_model
            for target, result in self.results.items()
        }

    @property
    def best_params(self) -> dict[str, Any]:
        """Return current best-model parameters for every target."""

        return {
            target: result.best_params
            for target, result in self.results.items()
        }

    @property
    def best_are_tuned(self) -> dict[str, bool]:
        """Return per-target tuning state for the selected models."""

        return {
            target: result.best_is_tuned
            for target, result in self.results.items()
        }

    def tune_best(
        self,
        *,
        targets: Sequence[str] | None = None,
        n_trials: int | Mapping[str, int] = 30,
        verbose: int = 0,
        evaluate: bool = True,
    ) -> dict[str, Any]:
        """Tune the selected best model for each requested output target.

        Args:
            targets: Targets to tune. ``None`` tunes every target.
            n_trials: Shared trial count or per-target trial-count mapping.
            verbose: OptunaSearchCV verbosity.
            evaluate: Whether to run each target's original comparison CV.

        Returns:
            Mapping from target name to tuned fitted model.
        """

        resolved_targets = list(self.results) if targets is None else list(targets)
        if len(resolved_targets) != len(set(resolved_targets)):
            raise ValueError("targets must not contain duplicates.")
        unknown_targets = sorted(set(resolved_targets).difference(self.results))
        if unknown_targets:
            raise ValueError(f"Unknown comparison targets: {unknown_targets}")
        if isinstance(n_trials, Mapping):
            unknown_trial_targets = sorted(set(n_trials).difference(self.results))
            if unknown_trial_targets:
                raise ValueError(
                    f"n_trials contains unknown targets: {unknown_trial_targets}"
                )

        tuned_models: dict[str, Any] = {}
        for target in resolved_targets:
            target_trials = (
                n_trials.get(target, 30)
                if isinstance(n_trials, Mapping)
                else n_trials
            )
            tuned_models[target] = self.results[target].tune_best(
                n_trials=target_trials,
                verbose=verbose,
                evaluate=evaluate,
            )
        return tuned_models


def _available_model_names(task: str) -> list[str]:
    """Return registered model names for one supervised-learning task."""

    from .utils import CLS_MODEL_DICT, REG_MODEL_DICT

    if task == "regression":
        return list(REG_MODEL_DICT)
    if task == "classification":
        return list(CLS_MODEL_DICT)
    raise ValueError("Model comparison supports regression and classification only.")


def _normalize_candidate_names(
    task: str,
    model_names: Sequence[str] | None,
) -> list[str]:
    """Validate and normalize requested comparison candidates."""

    available = _available_model_names(task)
    candidates = available if model_names is None else list(model_names)
    if not candidates:
        raise ValueError("At least one comparison model is required.")
    if len(candidates) != len(set(candidates)):
        raise ValueError("Comparison model names must not contain duplicates.")
    unknown = sorted(set(candidates).difference(available))
    if unknown:
        raise ValueError(f"Unknown model names for {task}: {unknown}")
    return candidates


def _original_target_frame(model: Any) -> pd.DataFrame:
    """Return original, non-encoded target values for candidate refits."""

    target = model.target_col
    context = getattr(model, "context", None)
    if context is not None and getattr(context, "Y", None) is not None:
        return context.Y[[target]].copy()

    target_data = getattr(model, "y", None)
    if target_data is None:
        raise ValueError("Training targets are unavailable for model comparison.")
    if isinstance(target_data, pd.Series):
        target_data = target_data.to_frame(name=target)
    else:
        target_data = target_data.copy()

    if model.task == "classification" and getattr(model, "le", None) is not None:
        encoded = np.asarray(target_data).reshape(-1)
        decoded = model.le.inverse_transform(encoded.astype(int))
        return pd.DataFrame(
            {target: decoded},
            index=model._get_X().index,
        )
    target_data.columns = [target]
    return target_data


def _comparison_context(model: Any) -> Any:
    """Build an immutable shared context reused by all candidate models."""

    from ..pipeline.single_output import PipelineSharedContext

    return PipelineSharedContext(
        X=model._get_X().copy(),
        Y=_original_target_frame(model),
        num_cols=list(model._shared_attr("num_cols")),
        cat_cols=list(model._shared_attr("cat_cols")),
        smiles_cols=list(model._shared_attr("smiles_cols")),
        comp_cols=list(model._shared_attr("comp_cols")),
        all_cols=list(model._shared_attr("all_cols")),
        unique_cols=dict(model._shared_attr("unique_cols") or {}),
    )


def _tune_fitted_candidate(
    candidate: Any,
    *,
    n_trials: int,
    verbose: int,
) -> Any:
    """Tune an initialized candidate and refresh its fitted model metadata."""

    from .training import tune_model
    from .utils import feature_names_from_pipeline

    if n_trials < 1:
        raise ValueError("tuning_trials must be at least 1.")

    X_train = candidate._get_X()
    y_train = candidate._get_y()
    tuned_model, best_params, best_base_params = tune_model(
        X=X_train,
        y=y_train,
        model_pipeline=candidate._make_pipeline(),
        model_names=candidate.model_names,
        base_model=candidate.base_model,
        ens_type=candidate.ens_type,
        sampling_method=candidate.sampling_method,
        cat_index=candidate.cat_index,
        cat_index_fit=candidate.cat_index_fit,
        task=candidate.task,
        n_trials=n_trials,
        verbose=verbose,
    )
    candidate.model = tuned_model
    candidate.model_params = best_params
    candidate.base_model_param = best_base_params
    candidate.tuning = True
    candidate.tuning_trials = n_trials
    candidate.feature_names = feature_names_from_pipeline(candidate.model)
    candidate.df_prerpocessed = pd.DataFrame(
        candidate.model["preprocess"].transform(X_train),
        columns=candidate.feature_names,
    )
    return candidate


def _fit_candidate(
    source_model: Any,
    context: Any,
    model_name: str,
    model_params: Mapping[str, Any] | None,
    tuning: bool,
    tuning_trials: int = 30,
    tuning_verbose: int = 0,
) -> Any:
    """Fit one candidate using the source model's preprocessing settings."""

    candidate = type(source_model)()
    candidate.fit_from_context(
        context=context,
        target_col=source_model.target_col,
        task=source_model.task,
        model_names=[model_name],
        fingerprints=source_model.fingerprints,
        comp_method=source_model.comp_method,
        comp_feats=source_model.comp_feats,
        ad=False,
        tuning=False,
        ensemble=False,
        ens_type=None,
        base_model=None,
        model_params=None if model_params is None else dict(model_params),
        base_model_param=None,
        num_impute_type=source_model.num_impute_type,
        num_scale_type=source_model.num_scale_type,
        cat_impute=source_model.cat_impute,
        poly=source_model.poly,
        poly_degree=source_model.poly_degree,
        poly_interaction_only=source_model.poly_interaction_only,
        decomposition=source_model.decomposition,
        decomposition_method=source_model.decomposition_method,
        dec_n_components=source_model.dec_n_components,
        sampling_method=source_model.sampling_method,
    )
    if tuning:
        candidate = _tune_fitted_candidate(
            candidate,
            n_trials=tuning_trials,
            verbose=tuning_verbose,
        )
    return candidate


def compare_single_output_model(
    model: Any,
    model_names: Sequence[str] | None = None,
    *,
    model_params: Mapping[str, Mapping[str, Any] | None] | None = None,
    method: str = "kfold",
    n_splits: int = 5,
    metric: str | None = None,
    tuning: bool = False,
    tune_best: bool = False,
    tuning_trials: int = 30,
    tuning_verbose: int = 0,
    continue_on_error: bool = True,
) -> ModelComparisonResult:
    """Fit and rank candidate estimators for one fitted model pipeline.

    Args:
        model: Fitted ``SingleOutputMLModelPipeline``-compatible object.
        model_names: Candidate estimator names. ``None`` compares all models.
        model_params: Optional estimator parameters keyed by model name.
        method: ``"kfold"`` or ``"loo"``.
        n_splits: Number of K-fold splits.
        metric: Ranking metric. Defaults to RMSE or F1.
        tuning: Tune every candidate before ranking. This is expensive.
        tune_best: Tune only the highest-ranked candidate after comparison.
        tuning_trials: Number of Optuna trials for candidate or best tuning.
        tuning_verbose: OptunaSearchCV verbosity.
        continue_on_error: Keep comparing when one candidate fails.

    Returns:
        Comparison result containing ranking and fitted candidate models.
    """

    if getattr(model, "target_col", None) in (None, "AD"):
        raise ValueError("Fit a supervised model before calling compare().")
    if method not in {"kfold", "loo"}:
        raise ValueError("method must be 'kfold' or 'loo'.")
    if tuning_trials < 1:
        raise ValueError("tuning_trials must be at least 1.")
    if tuning and tune_best:
        raise ValueError(
            "Use either tuning=True for every candidate or tune_best=True for "
            "two-stage comparison, not both."
        )

    context = _comparison_context(model)
    if method == "kfold" and not 2 <= n_splits <= len(context.X):
        raise ValueError(
            "n_splits must be between 2 and the number of training rows."
        )

    candidates = _normalize_candidate_names(model.task, model_names)
    default_metric = "RMSE" if model.task == "regression" else "F1"
    selected_metric = default_metric if metric is None else metric.upper()
    higher_is_better = selected_metric in {
        "R2",
        "ACCURACY",
        "PRECISION",
        "RECALL",
        "F1",
    }

    parameter_map = {} if model_params is None else dict(model_params)
    unknown_parameters = sorted(set(parameter_map).difference(candidates))
    if unknown_parameters:
        raise ValueError(
            "model_params contains models not selected for comparison: "
            f"{unknown_parameters}"
        )

    rows: list[dict[str, Any]] = []
    fitted_models: dict[str, Any] = {}
    failures: dict[str, str] = {}

    for model_name in candidates:
        try:
            candidate = _fit_candidate(
                source_model=model,
                context=context,
                model_name=model_name,
                model_params=parameter_map.get(model_name),
                tuning=tuning,
                tuning_trials=tuning_trials,
                tuning_verbose=tuning_verbose,
            )
            candidate.cv_score(method=method, n_splits=n_splits)
            train_scores = candidate.cv_scores["train"].iloc[0].to_dict()
            test_scores = candidate.cv_scores["test"].iloc[0].to_dict()
            row: dict[str, Any] = {
                "model_name": model_name,
                "target": model.target_col,
                "task": model.task,
            }
            row.update({f"train_{key}": value for key, value in train_scores.items()})
            row.update({f"test_{key}": value for key, value in test_scores.items()})
            rows.append(row)
            fitted_models[model_name] = candidate
        except Exception as exc:  # noqa: BLE001 - candidate failures are reported
            failures[model_name] = f"{type(exc).__name__}: {exc}"
            if not continue_on_error:
                raise

    if not rows:
        details = "; ".join(f"{name}: {error}" for name, error in failures.items())
        raise RuntimeError(f"Every comparison model failed. {details}")

    ranking = pd.DataFrame(rows)
    score_column = f"test_{selected_metric}"
    if score_column not in ranking.columns:
        available_metrics = sorted(
            column.removeprefix("test_")
            for column in ranking.columns
            if column.startswith("test_")
        )
        raise ValueError(
            f"Metric {selected_metric!r} is unavailable. "
            f"Choose from {available_metrics}."
        )
    ranking = ranking.sort_values(
        score_column,
        ascending=not higher_is_better,
        kind="stable",
    ).reset_index(drop=True)
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))

    result = ModelComparisonResult(
        ranking=ranking,
        models=fitted_models,
        failures=failures,
        metric=selected_metric,
        higher_is_better=higher_is_better,
        _source_model=model,
        method=method,
        n_splits=n_splits,
    )
    if tune_best:
        result.tune_best(
            n_trials=tuning_trials,
            verbose=tuning_verbose,
            evaluate=True,
        )
    return result


def compare_multi_output_model(
    model: Any,
    model_names: Sequence[str] | Mapping[str, Sequence[str]] | None = None,
    *,
    model_params: Mapping[str, Any] | None = None,
    method: str = "kfold",
    n_splits: int = 5,
    metric: str | Mapping[str, str] | None = None,
    tuning: bool = False,
    tune_best: bool = False,
    tuning_trials: int | Mapping[str, int] = 30,
    tuning_verbose: int = 0,
    continue_on_error: bool = True,
) -> MultiOutputModelComparisonResult:
    """Compare candidate estimators independently for every output target."""

    if not getattr(model, "target_cols", None):
        raise ValueError("Fit a multi-output model before calling compare().")

    target_set = set(model.target_cols)
    if isinstance(model_names, Mapping):
        unknown_targets = sorted(set(model_names).difference(target_set))
        if unknown_targets:
            raise ValueError(
                f"model_names contains unknown targets: {unknown_targets}"
            )
    if isinstance(metric, Mapping):
        unknown_targets = sorted(set(metric).difference(target_set))
        if unknown_targets:
            raise ValueError(f"metric contains unknown targets: {unknown_targets}")
    if isinstance(tuning_trials, Mapping):
        unknown_targets = sorted(set(tuning_trials).difference(target_set))
        if unknown_targets:
            raise ValueError(
                f"tuning_trials contains unknown targets: {unknown_targets}"
            )

    nested_params = (
        isinstance(model_params, Mapping)
        and bool(set(model_params).intersection(target_set))
    )
    results: dict[str, ModelComparisonResult] = {}
    for target in model.target_cols:
        target_names = (
            model_names.get(target)
            if isinstance(model_names, Mapping)
            else model_names
        )
        target_metric = metric.get(target) if isinstance(metric, Mapping) else metric
        target_params = (
            model_params.get(target, {})
            if nested_params and model_params is not None
            else model_params
        )
        target_trials = (
            tuning_trials.get(target, 30)
            if isinstance(tuning_trials, Mapping)
            else tuning_trials
        )
        results[target] = compare_single_output_model(
            model.models[target],
            model_names=target_names,
            model_params=target_params,
            method=method,
            n_splits=n_splits,
            metric=target_metric,
            tuning=tuning,
            tune_best=tune_best,
            tuning_trials=target_trials,
            tuning_verbose=tuning_verbose,
            continue_on_error=continue_on_error,
        )
    return MultiOutputModelComparisonResult(results=results)


def _optional_columns(value: Any) -> list[str]:
    """Normalize one optional legacy column argument to a list."""

    if value is None or value == [None]:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


def train_and_evaluate_models(
    df: pd.DataFrame,
    target_cols: Sequence[str],
    tasks: Sequence[str],
    num_cols: Sequence[str],
    cat_cols: Sequence[str],
    smiles_col: str | Sequence[str] | None = None,
    comp_col: str | Sequence[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Backward-compatible wrapper that compares all models for each target."""

    from ..pipeline import SingleOutputMLModelPipeline

    target_cols = list(target_cols)
    tasks = list(tasks)
    if len(target_cols) != len(tasks):
        raise ValueError("target_cols and tasks must have the same length.")

    smiles_cols = _optional_columns(smiles_col)
    comp_cols = _optional_columns(comp_col)
    comparison_results: dict[str, pd.DataFrame] = {}

    for target, task in zip(target_cols, tasks):
        seed_name = _available_model_names(task)[0]
        seed = SingleOutputMLModelPipeline()
        seed.fit(
            df=df,
            target_col=target,
            task=task,
            num_cols=list(num_cols),
            cat_cols=list(cat_cols),
            smiles_cols=smiles_cols,
            fingerprints=["RDKit"] if smiles_cols else [],
            comp_cols=comp_cols,
            comp_method="xenonpy" if comp_cols else None,
            comp_feats=None,
            model_names=[seed_name],
            tuning=False,
            ensemble=False,
            ens_type=None,
            base_model=None,
            model_params=None,
            base_model_param=None,
            num_impute_type=None,
            num_scale_type=None,
            cat_impute=False,
            poly=False,
            poly_degree=2,
            poly_interaction_only=False,
            decomposition=False,
            decomposition_method="PCA",
            dec_n_components=2,
        )
        result = compare_single_output_model(seed)
        ranking = result.ranking.set_index("model_name")
        metric_names = (
            ["RMSE", "MAE", "MAPE", "R2"]
            if task == "regression"
            else ["ACCURACY", "PRECISION", "RECALL", "F1"]
        )
        train = ranking[[f"train_{name}" for name in metric_names]].copy()
        test = ranking[[f"test_{name}" for name in metric_names]].copy()
        train.columns = pd.MultiIndex.from_product([["train"], metric_names])
        test.columns = pd.MultiIndex.from_product([["test"], metric_names])
        legacy_result = pd.concat([train, test], axis=1)
        legacy_result.attrs["failures"] = result.failures
        comparison_results[target] = legacy_result

    return comparison_results


__all__ = [
    "ModelComparisonResult",
    "MultiOutputModelComparisonResult",
    "compare_multi_output_model",
    "compare_single_output_model",
    "train_and_evaluate_models",
]
