import sys
import types

import pandas as pd

from malchan.pipeline.single_output import PipelineSharedContext, SingleOutputMLModelPipeline


class DummyPreprocess:
    """Minimal preprocess step used to isolate shared-context target handling."""

    def transform(self, X):
        """Return a single feature column for the supplied rows.

        Args:
            X: Feature matrix supplied to the dummy pipeline.

        Returns:
            list: One zero-valued feature per row.
        """
        return [[0] for _ in range(len(X))]


class DummyPipeline:
    """Minimal sklearn-like pipeline with a preprocess step."""

    def __init__(self):
        """Initialize the dummy preprocess step."""
        self.preprocess = DummyPreprocess()

    def __getitem__(self, key):
        """Return a named pipeline step.

        Args:
            key: Pipeline step name.

        Returns:
            DummyPreprocess: The preprocess step.
        """
        if key != "preprocess":
            raise KeyError(key)
        return self.preprocess


def test_fit_from_context_passes_target_y_to_regressor(monkeypatch):
    """Regression fitting from a shared context must pass the selected target y."""

    captured = {}

    def fit_model(**kwargs):
        """Capture the y value passed from the single-output pipeline.

        Args:
            **kwargs: Keyword arguments passed by ``_fit_prepared``.

        Returns:
            DummyPipeline: The unchanged dummy model pipeline.
        """
        captured["y"] = kwargs["y"]
        return kwargs["model_pipeline"]

    fake_training = types.ModuleType("malchan.models.training")
    fake_training.fit_model = fit_model
    fake_training.tune_model = None

    fake_utils = types.ModuleType("malchan.models.utils")
    fake_utils.feature_names_from_pipeline = lambda model: ["feature"]
    fake_utils.label_encode = lambda y: (y, None)

    monkeypatch.setitem(sys.modules, "malchan.models.training", fake_training)
    monkeypatch.setitem(sys.modules, "malchan.models.utils", fake_utils)
    monkeypatch.setattr(SingleOutputMLModelPipeline, "_make_pipeline", lambda self: DummyPipeline())

    context = PipelineSharedContext(
        X=pd.DataFrame({"x": [1.0, 2.0, 3.0]}),
        Y=pd.DataFrame({"property": [10.0, 20.0, 30.0]}),
        num_cols=["x"],
        cat_cols=[],
        smiles_cols=[],
        comp_cols=[],
        all_cols=["x"],
        unique_cols={},
    )
    pipeline = SingleOutputMLModelPipeline()

    pipeline.fit_from_context(
        context=context,
        target_col="property",
        task="regression",
        model_names=["ランダムフォレスト回帰"],
    )

    pd.testing.assert_frame_equal(captured["y"], context.Y[["property"]])
