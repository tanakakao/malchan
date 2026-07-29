"""Source regression tests for trained-model diagrams and validation summaries."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / "frontend" / "src" / "components" / "ModelResultVisualizationControl.jsx"
API = ROOT / "frontend" / "src" / "api.js"
APP = ROOT / "frontend" / "src" / "App.jsx"
MAIN = ROOT / "frontend" / "src" / "main.jsx"
CSS = ROOT / "frontend" / "src" / "model-result-visualization.css"


def test_registered_model_uses_sandboxed_sklearn_diagram() -> None:
    """The raw model JSON should be replaced by a target-specific diagram panel."""

    source = CONTROL.read_text(encoding="utf-8")
    api = API.read_text(encoding="utf-8")
    app = APP.read_text(encoding="utf-8")

    assert "api.modelVisualization(modelInfo.model_id)" in source
    assert 'className="sklearn-diagram-frame"' in source
    assert 'sandbox=""' in source
    assert "srcDoc={diagramDocument(targetDiagram.html)}" in source
    assert "ModelTargetTabs" in source
    assert 'request(`/models/${encodeURIComponent(modelId)}/visualization`)' in api
    assert "ModelResultVisualizationControl" in app
    assert "<ModelResultVisualizationControl />" in app


def test_model_html_is_fixed_directly_below_registered_model_heading() -> None:
    """Estimator HTML should be mounted directly below the registered-model title."""

    source = CONTROL.read_text(encoding="utf-8")

    assert 'contentRoot.querySelector(".model-registration-panel")' in source
    assert 'panel.querySelector(":scope > .panel-title")' in source
    assert 'nextHost.dataset.location = "registered-model"' in source
    assert 'nextHost.setAttribute("aria-label", "登録モデルの構成と精度検証")' in source
    assert 'panelTitle.insertAdjacentElement("afterend", nextHost)' in source
    assert 'panel.insertBefore(nextHost, codebox)' not in source


def test_validation_result_is_grouped_by_metric_and_fold_statistics() -> None:
    """Validation metrics should show Train and Validation means and standard deviations."""

    source = CONTROL.read_text(encoding="utf-8")
    api = API.read_text(encoding="utf-8")

    assert "function summarizeMetric" in source
    assert "Math.sqrt(variance)" in source
    assert "平均 ± 標準偏差" in source
    assert "Train" in source
    assert "Validation" in source
    assert 'new CustomEvent("malchan:model-evaluated"' in api
    assert 'window.addEventListener("malchan:model-evaluated"' in source


def test_model_result_layout_hides_duplicate_json_and_evaluation_panels() -> None:
    """The enhanced result card should replace duplicate legacy result blocks."""

    css = CSS.read_text(encoding="utf-8")
    main = MAIN.read_text(encoding="utf-8")

    assert '.model-registration-panel:has(> .model-result-visualization-host) > .codebox' in css
    assert ".evaluation-result-panel" in css
    assert ".model-result-summary" in css
    assert ".model-result-metrics" in css
    assert "height: 460px" in css
    assert 'import "./model-result-visualization.css";' in main
