"""Source regression tests for trained-model structures and validation summaries."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / "frontend" / "src" / "components" / "ModelResultVisualizationControl.jsx"
API = ROOT / "frontend" / "src" / "api.js"
APP = ROOT / "frontend" / "src" / "App.jsx"
MAIN = ROOT / "frontend" / "src" / "main.jsx"
CSS = ROOT / "frontend" / "src" / "model-result-visualization.css"


def test_registered_model_uses_native_react_structure_diagram() -> None:
    """The result card should render API structure nodes without an iframe."""

    source = CONTROL.read_text(encoding="utf-8")
    api = API.read_text(encoding="utf-8")
    app = APP.read_text(encoding="utf-8")

    assert "api.modelVisualization(modelInfo.model_id)" in source
    assert "function ModelStructureDiagram" in source
    assert "function StructureNode" in source
    assert "function StructureCard" in source
    assert "targetDiagram.structure" in source
    assert 'className="model-structure-canvas"' in source
    assert "<iframe" not in source
    assert "sandbox=" not in source
    assert "srcDoc=" not in source
    assert "diagramDocument" not in source
    assert "ModelTargetTabs" in source
    assert 'request(`/models/${encodeURIComponent(modelId)}/visualization`)' in api
    assert "ModelResultVisualizationControl" in app
    assert "<ModelResultVisualizationControl />" in app


def test_native_structure_shows_columns_parameters_and_flow_labels() -> None:
    """Cards should expose the important parts of the fitted processing flow."""

    source = CONTROL.read_text(encoding="utf-8")

    assert "NODE_KIND_LABELS" in source
    assert "NODE_NAME_LABELS" in source
    assert 'className="model-structure-columns"' in source
    assert 'className="model-structure-parameters"' in source
    assert "入力データ" in source
    assert "予測結果" in source
    assert 'node.kind === "pipeline"' in source
    assert '["branch", "ensemble"].includes(node.kind)' in source


def test_model_structure_is_fixed_below_registered_model_heading() -> None:
    """The native diagram should remain directly below the registered-model title."""

    source = CONTROL.read_text(encoding="utf-8")

    assert 'contentRoot.querySelector(".model-registration-panel")' in source
    assert 'panel.querySelector(":scope > .panel-title")' in source
    assert 'nextHost.dataset.location = "registered-model"' in source
    assert 'nextHost.setAttribute("aria-label", "登録モデルの構成と精度検証")' in source
    assert 'panelTitle.insertAdjacentElement("afterend", nextHost)' in source
    assert "panel.insertBefore(nextHost, codebox)" not in source


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


def test_model_result_layout_uses_cards_connections_and_no_iframe() -> None:
    """The native tree should use cards and sequence or branch connectors."""

    css = CSS.read_text(encoding="utf-8")
    main = MAIN.read_text(encoding="utf-8")

    assert '.model-registration-panel:has(> .model-result-visualization-host) > .codebox' in css
    assert ".evaluation-result-panel" in css
    assert ".model-result-summary" in css
    assert ".model-result-metrics" in css
    assert ".model-structure-card" in css
    assert ".layout-sequence" in css
    assert ".layout-branches" in css
    assert ".model-structure-columns" in css
    assert ".sklearn-diagram-frame" not in css
    assert "iframe" not in css
    assert 'import "./model-result-visualization.css";' in main
