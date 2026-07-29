"""Tests for sklearn-style diagrams when direct estimator rendering fails."""

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.utils import estimator_html_repr as sklearn_estimator_html_repr


def _sample_pipeline() -> Pipeline:
    """Return a representative preprocessing and predictor pipeline."""

    return Pipeline(
        steps=[
            (
                "preprocess",
                ColumnTransformer(
                    transformers=[
                        ("numeric", SimpleImputer(strategy="mean"), ["x"]),
                    ]
                ),
            ),
            ("predictor", RandomForestRegressor(n_estimators=5, random_state=0)),
        ]
    )


def test_diagram_fallback_preserves_sklearn_block_structure(monkeypatch) -> None:
    """A rendering error should retry with a safe diagram instead of plain repr text."""

    from malchan.app.services import model_visualization_service as service

    calls = []

    def fail_direct_render_once(estimator):
        calls.append(estimator)
        if len(calls) == 1:
            raise ValueError("simulated estimator incompatibility")
        return sklearn_estimator_html_repr(estimator)

    monkeypatch.setattr(service, "estimator_html_repr", fail_direct_render_once)

    html, renderer = service._diagram_html(_sample_pipeline())

    assert len(calls) == 2
    assert renderer == "sklearn"
    assert "sk-container" in html
    assert "preprocess" in html
    assert "ColumnTransformer" in html
    assert "RandomForestRegressor" in html
    assert "malchan-estimator-fallback" not in html


def test_display_estimator_handles_unhashable_transformers() -> None:
    """Estimator objects must not be compared through string-only hash sets."""

    from malchan.app.services import model_visualization_service as service

    display_estimator = service._display_estimator(_sample_pipeline())
    html = sklearn_estimator_html_repr(display_estimator)

    assert "sk-container" in html
    assert "numeric" in html
    assert "predictor" in html
