"""Model explainability helpers.

This module contains only model/XAI dependencies. Chemistry and materials
featurizers are intentionally isolated under :mod:`malchan.features` so that
numeric and categorical XAI does not require optional packages such as
XenonPy, Matminer, RDKit, or scikit-fingerprints.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
import shap
from sklearn.base import RegressorMixin
from sklearn.covariance import EllipticEnvelope
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    IsolationForest,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import (
    ARDRegression,
    BayesianRidge,
    ElasticNet,
    HuberRegressor,
    Lasso,
    LassoLars,
    LinearRegression,
    LogisticRegression,
    OrthogonalMatchingPursuit,
    PassiveAggressiveClassifier,
    PassiveAggressiveRegressor,
    Ridge,
    RidgeClassifier,
    TweedieRegressor,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import OneClassSVM, SVC, SVR
from sklearn.tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
)


_SKLEARN_TREE_REGRESSORS = (
    DecisionTreeRegressor,
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
)
_SKLEARN_TREE_CLASSIFIERS = (
    DecisionTreeClassifier,
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
)
_LINEAR_MODELS = (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet,
    TweedieRegressor,
    HuberRegressor,
    PLSRegression,
    BayesianRidge,
    ARDRegression,
    PassiveAggressiveRegressor,
    LassoLars,
    OrthogonalMatchingPursuit,
)
_PROBABILITY_CLASSIFIERS = (
    GaussianProcessClassifier,
    SVC,
    MLPClassifier,
    LogisticRegression,
    RidgeClassifier,
    PassiveAggressiveClassifier,
    GaussianNB,
)


def _external_model_family(model: Any) -> str:
    """Return the top-level package name for an optional estimator."""
    return type(model).__module__.split(".", maxsplit=1)[0]


def _is_external_tree_regressor(model: Any) -> bool:
    """Return whether an optional estimator is a supported tree regressor."""
    family = _external_model_family(model)
    name = type(model).__name__
    return family in {"xgboost", "lightgbm"} and name.endswith("Regressor")


def _is_external_tree_classifier(model: Any) -> bool:
    """Return whether an optional estimator is a supported tree classifier."""
    family = _external_model_family(model)
    name = type(model).__name__
    return family in {"xgboost", "lightgbm"} and name.endswith("Classifier")


def _is_catboost(model: Any) -> bool:
    """Return whether the estimator comes from CatBoost."""
    return _external_model_family(model) == "catboost"


def get_pd_and_ice(
    X: pd.DataFrame,
    _model: RegressorMixin,
    target: str,
    unique_dict: dict[str, Any] | None = None,
    bounds: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate partial-dependence and ICE values for one raw feature."""
    unique_dict = {} if unique_dict is None else unique_dict
    if target in unique_dict:
        column_names = [column for column in X.columns if target in column]
        target_idx = np.where(np.isin(X.columns.to_numpy(), column_names))[0]
        xticks = np.asarray(unique_dict[target])
    else:
        target_idx = np.where(np.asarray(X.columns == target))[0][0]
        if bounds is not None:
            xticks = np.linspace(bounds[0], bounds[1], 200)
        else:
            xticks = np.linspace(X[target].min(), X[target].max(), 200)

    X_sample = X.sample(300) if X.shape[0] > 300 else X
    partial_dependence = []
    for row_index in range(len(X_sample)):
        values = X_sample.values[[row_index], :]
        values = np.repeat(values, len(xticks), axis=0)
        values[:, target_idx] = (
            np.asarray(xticks).reshape(-1, 1)
            if target in unique_dict
            else np.asarray(xticks)
        )
        values_frame = pd.DataFrame(values, columns=X.columns)
        prediction = _model.predict(
            values_frame,
            proba=getattr(_model, "task", "") == "classification",
        ).values
        if prediction.shape[-1] == 1:
            prediction = prediction.reshape(-1, 1)
        else:
            prediction = prediction.reshape(-1, 1, prediction.shape[-1])
        partial_dependence.append(prediction)

    return np.concatenate(partial_dependence, axis=1), xticks


def get_pd_and_ice_2d(
    X: pd.DataFrame,
    _model: RegressorMixin,
    targets: list[str],
    unique_dict: dict[str, Any] | None = None,
    bounds: list[tuple[float, float]] | None = None,
    n_grid: int = 75,
    max_samples: int = 300,
    random_state: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate two-dimensional partial-dependence and ICE values."""
    unique_dict = {} if unique_dict is None else unique_dict
    if len(targets) != 2:
        raise ValueError(f"targets must be length 2, got {len(targets)}: {targets}")

    metadata: list[dict[str, Any]] = []
    for index, target in enumerate(targets):
        if target in unique_dict:
            onehot_cols = [
                column
                for column in X.columns
                if str(column).startswith(f"{target}_")
            ]
            if len(onehot_cols) >= 2:
                column_idx = np.where(
                    np.isin(X.columns.to_numpy(), onehot_cols)
                )[0]
                metadata.append(
                    {
                        "name": target,
                        "kind": "onehot",
                        "idx": column_idx,
                        "cols": onehot_cols,
                        "ticks": list(unique_dict[target]),
                    }
                )
            else:
                if target not in X.columns:
                    raise ValueError(
                        f"unique_dict has key {target!r}, but {target!r} not in X.columns."
                    )
                column_idx = int(
                    np.where(X.columns.to_numpy() == target)[0][0]
                )
                metadata.append(
                    {
                        "name": target,
                        "kind": "categorical",
                        "idx": column_idx,
                        "ticks": list(unique_dict[target]),
                    }
                )
        else:
            if target not in X.columns:
                raise ValueError(f"{target!r} not in X.columns.")
            column_idx = int(np.where(X.columns.to_numpy() == target)[0][0])
            if bounds is not None:
                lower, upper = bounds[index]
            else:
                lower, upper = float(X[target].min()), float(X[target].max())
            metadata.append(
                {
                    "name": target,
                    "kind": "numeric",
                    "idx": column_idx,
                    "ticks": np.linspace(lower, upper, n_grid),
                }
            )

    ticks0 = metadata[0]["ticks"]
    ticks1 = metadata[1]["ticks"]
    grid0, grid1 = np.meshgrid(ticks0, ticks1, indexing="xy")
    grid0 = grid0.ravel()
    grid1 = grid1.ravel()
    grid_size = grid0.shape[0]

    X_sample = (
        X.sample(max_samples, random_state=random_state)
        if X.shape[0] > max_samples
        else X
    )

    def apply_onehot(
        values: np.ndarray,
        base_name: str,
        columns: list[str],
        categories: list[Any],
        chosen: np.ndarray,
    ) -> None:
        absolute_columns = np.where(np.isin(X.columns.to_numpy(), columns))[0]
        values[:, absolute_columns] = 0.0
        column_set = set(columns)
        category_to_column: dict[Any, int] = {}
        for category in categories:
            column = f"{base_name}_{category}"
            if column in column_set:
                category_to_column[category] = int(
                    np.where(X.columns.to_numpy() == column)[0][0]
                )

        if not category_to_column and len(categories) == len(columns):
            for category, column in zip(categories, columns, strict=True):
                category_to_column[category] = int(
                    np.where(X.columns.to_numpy() == column)[0][0]
                )

        if not category_to_column:
            raise ValueError(
                f"Cannot map categories to one-hot columns for {base_name!r}. "
                f"categories={categories}, onehot_cols={columns[:5]}..."
            )

        for category, absolute_column in category_to_column.items():
            mask = chosen == category
            if np.any(mask):
                values[mask, absolute_column] = 1.0

    partial_dependence = []
    for row_index in range(len(X_sample)):
        values = np.repeat(
            X_sample.iloc[[row_index]].to_numpy(),
            grid_size,
            axis=0,
        )
        for meta, chosen in zip(metadata, (grid0, grid1), strict=True):
            if meta["kind"] == "numeric":
                values[:, meta["idx"]] = chosen.astype(float)
            elif meta["kind"] == "categorical":
                values[:, meta["idx"]] = chosen
            else:
                apply_onehot(
                    values,
                    meta["name"],
                    meta["cols"],
                    meta["ticks"],
                    chosen,
                )

        values_frame = pd.DataFrame(values, columns=X.columns)
        prediction = _model.predict(
            values_frame,
            proba=getattr(_model, "task", "") == "classification",
        )
        prediction = np.asarray(getattr(prediction, "values", prediction))
        if prediction.ndim == 1:
            prediction = prediction.reshape(-1, 1)
        elif prediction.ndim == 2 and prediction.shape[1] >= 2:
            prediction = prediction.reshape(-1, 1, prediction.shape[1])
        else:
            prediction = prediction.reshape(-1, 1)
        partial_dependence.append(prediction)

    return np.concatenate(partial_dependence, axis=1), grid0, grid1


def get_shap_values(
    model: RegressorMixin,
    X: pd.DataFrame,
) -> tuple[np.ndarray | None, np.ndarray | None, Any | None, pd.DataFrame | None]:
    """Calculate SHAP values with an explainer suitable for the estimator."""
    X_sample = shap.utils.sample(X, 300) if len(X) > 300 else X

    if isinstance(model, _SKLEARN_TREE_REGRESSORS) or _is_external_tree_regressor(model):
        explainer = shap.TreeExplainer(
            model,
            X_sample,
            feature_names=X.columns,
        )
    elif isinstance(model, _SKLEARN_TREE_CLASSIFIERS) or _is_external_tree_classifier(model):
        explainer = shap.Explainer(
            model.predict_proba,
            X_sample,
            feature_names=X.columns,
        )
    elif _is_catboost(model):
        explainer = shap.TreeExplainer(
            model,
            feature_perturbation="tree_path_dependent",
        )
    elif isinstance(model, _LINEAR_MODELS):
        explainer = shap.LinearExplainer(
            model,
            X_sample,
            feature_names=X.columns,
        )
    elif isinstance(
        model,
        (GaussianProcessRegressor, SVR, MLPRegressor, OneClassSVM, IsolationForest),
    ):
        explainer = shap.Explainer(model.predict, X_sample)
    elif isinstance(model, _PROBABILITY_CLASSIFIERS):
        explainer = shap.Explainer(model.predict_proba, X_sample)
    elif isinstance(model, EllipticEnvelope):
        explainer = shap.Explainer(model.decision_function, X_sample)
    else:
        return None, None, None, None

    num_features = X_sample.shape[1]
    if isinstance(explainer, shap.explainers._permutation.PermutationExplainer):
        max_evals = max(2 * num_features + 1, 1000)
    elif isinstance(explainer, shap.explainers._exact.ExactExplainer):
        if num_features > 15:
            warnings.warn(
                f"ExactExplainer with {num_features} features may be extremely slow.",
                RuntimeWarning,
                stacklevel=2,
            )
        max_evals = 2**num_features
    elif isinstance(explainer, shap.explainers._kernel.KernelExplainer):
        max_evals = max(2 * num_features + 1, 1000)
    else:
        max_evals = max(2 * num_features + 1, 1000)

    try:
        shap_result = explainer(X_sample, check_additivity=False)
    except TypeError:
        shap_result = explainer(X_sample, max_evals=max_evals)

    return (
        shap_result.values,
        shap_result.base_values,
        explainer,
        X_sample,
    )


def get_shap_scatter(
    X: pd.DataFrame,
    shap_values: np.ndarray,
    target_col: str,
    modelname: str | None = None,
    unique_dict: dict[str, list[Any]] | None = None,
    smiles_cols: list[str] | None = None,
    comp_cols: list[str] | None = None,
    le: Any | None = None,
) -> pd.DataFrame:
    """Build plotting data for SHAP values of one raw feature."""
    unique_dict = {} if unique_dict is None else unique_dict
    smiles_cols = [] if smiles_cols is None else smiles_cols
    comp_cols = [] if comp_cols is None else comp_cols

    if target_col in unique_dict:
        column_names = [column for column in X.columns if target_col in column]
        column_idx = np.where(np.isin(X.columns.to_numpy(), column_names))[0]
        X_plot = X[column_names].copy()
        combined_columns = [
            f"{target_col}_{category}"
            for category in unique_dict[target_col]
        ]
        X_plot["cat"] = combined_columns[0]

        if modelname == "LightGBM":
            for index in range(1, len(combined_columns)):
                X_plot.loc[
                    X_plot[target_col] == index,
                    "cat",
                ] = combined_columns[index]
        elif (
            modelname == "CatBoost"
            or target_col in smiles_cols
            or target_col in comp_cols
        ):
            for index in range(1, len(combined_columns)):
                X_plot.loc[
                    X_plot[target_col] == unique_dict[target_col][index],
                    "cat",
                ] = combined_columns[index]
        else:
            for index in range(1, len(combined_columns)):
                X_plot.loc[
                    X_plot[combined_columns[index]] == 1,
                    "cat",
                ] = combined_columns[index]

        X_plot = X_plot[["cat"]].rename(columns={"cat": target_col})
        if shap_values.ndim == 2:
            X_plot["shap"] = shap_values[:, column_idx].sum(axis=1)
        else:
            if le is None:
                raise ValueError("le is required for multi-class SHAP values.")
            output_columns = [
                f"shap_{category}"
                for category in le.inverse_transform(
                    np.arange(shap_values.shape[-1])
                )
            ]
            X_plot[output_columns] = shap_values[:, column_idx].sum(axis=1)
        return X_plot.sort_values(target_col)

    shap_column_idx = np.where(np.asarray(X.columns) == target_col)[0][0]
    X_plot = X[[target_col]].copy()
    if shap_values.ndim == 2:
        X_plot["shap"] = shap_values[:, shap_column_idx]
    else:
        if le is None:
            raise ValueError("le is required for multi-class SHAP values.")
        output_columns = [
            f"shap_{category}"
            for category in le.inverse_transform(
                np.arange(shap_values.shape[-1])
            )
        ]
        X_plot[output_columns] = shap_values[:, shap_column_idx]
    return X_plot


def get_importances(model: RegressorMixin) -> np.ndarray | None:
    """Extract native feature importance values from a fitted estimator."""
    if isinstance(model, GaussianNB):
        feature_importances = model.theta_.mean(axis=0)
    elif isinstance(model, (PLSRegression, LogisticRegression)):
        feature_importances = model.coef_[0]
    elif hasattr(model, "coef_"):
        feature_importances = np.asarray(model.coef_)
        if feature_importances.ndim == 2:
            feature_importances = feature_importances[0]
    elif hasattr(model, "feature_importances_"):
        feature_importances = model.feature_importances_
    else:
        warnings.warn(
            f"Model type {type(model).__name__} is not supported for feature "
            "importance extraction.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    return np.asarray(feature_importances)


def get_pfi_values(
    model: RegressorMixin,
    X: pd.DataFrame | np.ndarray,
    y: pd.DataFrame | np.ndarray,
) -> np.ndarray:
    """Calculate permutation feature importance values."""
    X_values = X.values if isinstance(X, pd.DataFrame) else X
    y_values = y.values.ravel() if isinstance(y, pd.DataFrame) else y
    max_samples = 300 / X_values.shape[0] if X_values.shape[0] > 300 else 1.0
    result = permutation_importance(
        model,
        X_values,
        y_values,
        max_samples=max_samples,
        n_jobs=-1,
        random_state=0,
    )
    return result.importances_mean


def get_feature_importances(
    model: RegressorMixin,
    importance_type: str = "model",
    n_features: int = 15,
    shap_values: np.ndarray | None = None,
    X: pd.DataFrame | np.ndarray | None = None,
    y: pd.DataFrame | np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate importance values and return indices of the top features."""
    if importance_type == "model":
        feature_importances = get_importances(model)
        if feature_importances is None:
            raise ValueError(
                f"Model type {type(model).__name__} does not expose importances."
            )
    elif importance_type == "shap":
        if shap_values is None:
            raise ValueError("shap_values must be provided for SHAP importance.")
        feature_importances = np.sqrt((shap_values**2).sum(axis=0))
    elif importance_type == "pfi":
        if X is None or y is None:
            raise ValueError("X and y must be provided for permutation importance.")
        feature_importances = get_pfi_values(model, X, y)
    else:
        raise ValueError(f"Unknown importance_type: {importance_type}")

    importance_idx = np.argsort(-np.abs(feature_importances))[:n_features]
    return feature_importances, importance_idx


__all__ = [
    "get_feature_importances",
    "get_importances",
    "get_pd_and_ice",
    "get_pd_and_ice_2d",
    "get_pfi_values",
    "get_shap_scatter",
    "get_shap_values",
]
