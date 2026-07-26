"""Tests for inverse-analysis step normalization before Optuna suggestions."""

from types import SimpleNamespace

import numpy as np
import optuna
import pandas as pd
import pytest

from malchan.inverse_analysis import utils
from malchan.inverse_analysis._step_normalization import (
    normalize_optional_step,
    safe_suggest_parameter,
)


def test_search_setting_nan_step_is_safe_for_float_suggestion() -> None:
    """Treat a pandas-converted NaN float step as an Optuna continuous search."""

    model = SimpleNamespace(
        target_cols=["property"],
        num_cols=["continuous", "integer"],
        cat_cols=[],
        smiles_cols=[],
        comp_cols=[],
        X=pd.DataFrame(
            {
                "continuous": [0.1, 0.2, 0.3],
                "integer": pd.Series([1, 2, 3], dtype="int64"),
            }
        ),
    )
    df_range, *_ = utils.search_setting(
        model=model,
        obj_directions=["max"],
        bounds_min=[0.1, 1],
        bounds_max=[0.3, 3],
        dtypes=["float", "int"],
        steps=[None, 1],
        fix_values=[None, None],
        target_cols=["property"],
    )

    assert pd.isna(df_range.loc["continuous", "step"])

    study = optuna.create_study(direction="maximize")
    trial = study.ask()
    value = utils.suggest_parameter(
        trial,
        "continuous",
        "float",
        df_range.loc["continuous", "min"],
        df_range.loc["continuous", "max"],
        step=df_range.loc["continuous", "step"],
    )

    assert 0.1 <= value <= 0.3
    assert utils.suggest_parameter is safe_suggest_parameter


def test_normalize_optional_step_uses_defaults_for_missing_values() -> None:
    """Map pandas missing steps to continuous or integer Optuna defaults."""

    assert normalize_optional_step(np.nan, dtype="float", param_name="x") is None
    assert normalize_optional_step(pd.NA, dtype="float", param_name="x") is None
    assert normalize_optional_step(np.nan, dtype="int", param_name="n") == 1


def test_normalize_optional_step_rejects_invalid_values() -> None:
    """Reject non-finite, non-positive, and fractional integer steps clearly."""

    with pytest.raises(ValueError, match="finite and positive"):
        normalize_optional_step(np.inf, dtype="float", param_name="x")
    with pytest.raises(ValueError, match="finite and positive"):
        normalize_optional_step(0, dtype="float", param_name="x")
    with pytest.raises(ValueError, match="must be integral"):
        normalize_optional_step(0.5, dtype="int", param_name="n")


def test_safe_suggest_parameter_rejects_non_finite_bounds() -> None:
    """Raise an actionable error for all-missing or non-finite training bounds."""

    study = optuna.create_study(direction="maximize")
    trial = study.ask()

    with pytest.raises(ValueError, match="must be finite"):
        safe_suggest_parameter(
            trial,
            "all_missing",
            "float",
            np.nan,
            np.nan,
        )
