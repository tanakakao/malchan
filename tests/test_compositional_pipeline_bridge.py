"""Regression tests for compositional settings on public pipeline classes."""

import importlib.util

import pandas as pd
import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("imblearn") is None,
    reason="Pipeline bridge tests require the models extra.",
)


def _noop_fit_prepared(self, **kwargs) -> None:
    """Stop before estimator fitting while retaining prepared pipeline state."""

    self._captured_fit_prepared = kwargs


def test_single_output_fit_accepts_compositional_preprocessing_kwargs(monkeypatch) -> None:
    """New settings should be accepted without changing the legacy fit body."""

    from malchan.pipeline import SingleOutputMLModelPipeline

    monkeypatch.setattr(
        SingleOutputMLModelPipeline,
        "_fit_prepared",
        _noop_fit_prepared,
    )
    frame = pd.DataFrame(
        {
            "a": [0.2, 0.3, 0.4],
            "b": [0.8, 0.7, 0.6],
            "temperature": [100.0, 200.0, 300.0],
            "y": [1.0, 2.0, 3.0],
        }
    )
    pipeline = SingleOutputMLModelPipeline()

    pipeline.fit(
        df=frame,
        target_col="y",
        task="regression",
        num_cols=["a", "b", "temperature"],
        cat_cols=[],
        model_names=["Ridge"],
        compositional_groups=[["a", "b"]],
        compositional_method="CLR",
        compositional_zero_replacement=1e-5,
        compositional_closure=True,
        compositional_scale_type="StandardScaler",
    )

    assert pipeline.compositional_groups == [["a", "b"]]
    assert pipeline.compositional_method == "CLR"
    assert pipeline.compositional_zero_replacement == pytest.approx(1e-5)
    assert pipeline.compositional_closure is True
    assert pipeline.compositional_scale_type == "StandardScaler"
    assert pipeline.all_cols == ["a", "b", "temperature"]


def test_multi_output_fit_propagates_compositional_settings_to_children(monkeypatch) -> None:
    """Shared-context child pipelines should inherit the same simplex settings."""

    from malchan.pipeline import MLModelPipeline, SingleOutputMLModelPipeline

    monkeypatch.setattr(
        SingleOutputMLModelPipeline,
        "_fit_prepared",
        _noop_fit_prepared,
    )
    frame = pd.DataFrame(
        {
            "a": [0.2, 0.3, 0.4],
            "b": [0.8, 0.7, 0.6],
            "temperature": [100.0, 200.0, 300.0],
            "strength": [1.0, 2.0, 3.0],
            "cost": [3.0, 2.0, 1.0],
        }
    )
    pipeline = MLModelPipeline()

    pipeline.fit(
        df=frame,
        target_cols=["strength", "cost"],
        tasks=["regression", "regression"],
        num_cols=["a", "b", "temperature"],
        cat_cols=[],
        model_names=[["Ridge"], ["Ridge"]],
        compositional_groups=[["a", "b"]],
        compositional_method="ILR",
        compositional_zero_replacement=1e-6,
        compositional_closure=True,
    )

    assert pipeline.compositional_groups == [["a", "b"]]
    assert pipeline.models["strength"].compositional_groups == [["a", "b"]]
    assert pipeline.models["cost"].compositional_groups == [["a", "b"]]
    assert pipeline.models["strength"].compositional_method == "ILR"
    assert pipeline.models["cost"].compositional_method == "ILR"
