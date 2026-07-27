from pathlib import Path


CONTROL = Path("frontend/src/components/YyDiagnosticsControl.jsx")
APP = Path("frontend/src/App.jsx")


def test_explain_yy_plot_supports_residual_and_cv_controls() -> None:
    """Explain should expose residual and CV options through yy_plot_ml API parameters."""

    source = CONTROL.read_text(encoding="utf-8")
    app_source = APP.read_text(encoding="utf-8")

    assert "差分プロット" in source
    assert 'cv: source === "cv"' in source
    assert 'residual: plotType === "residual"' in source
    assert "CV Train / Validation" in source
    assert 'train_test="${split}"' in source
    assert 'disabled={!cvAvailable}' in source
    assert "YyDiagnosticsControl" in app_source
