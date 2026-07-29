"""Source regression tests for trained-model summaries and validation results."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / "frontend" / "src" / "components" / "ModelResultVisualizationControl.jsx"
SUMMARY = ROOT / "frontend" / "src" / "components" / "ModelStructureSummaryTable.jsx"
API = ROOT / "frontend" / "src" / "api.js"
APP = ROOT / "frontend" / "src" / "App.jsx"
MAIN = ROOT / "frontend" / "src" / "main.jsx"
RESULT_CSS = ROOT / "frontend" / "src" / "model-result-visualization.css"
SUMMARY_CSS = ROOT / "frontend" / "src" / "model-structure-summary.css"


def test_registered_model_uses_summary_table_without_iframe() -> None:
    """The result card should render a concise four-column structure table."""

    source = CONTROL.read_text(encoding="utf-8")
    summary = SUMMARY.read_text(encoding="utf-8")
    api = API.read_text(encoding="utf-8")
    app = APP.read_text(encoding="utf-8")

    assert "api.modelVisualization(modelInfo.model_id)" in source
    assert "ModelStructureSummaryTable" in source
    assert "targetDiagram.structure" in source
    assert 'className="model-structure-summary-table"' in summary
    assert "変数名" in summary
    assert "共通前処理" in summary
    assert "種別別の前処理" in summary
    assert "モデル" in summary
    assert "<iframe" not in source
    assert "srcDoc=" not in source
    assert "ModelTargetTabs" in source
    assert 'request(`/models/${encodeURIComponent(modelId)}/visualization`)' in api
    assert "ModelResultVisualizationControl" in app
    assert "<ModelResultVisualizationControl />" in app


def test_summary_table_groups_feature_types_and_preprocessing() -> None:
    """Columns should be grouped into numeric, category, composition, and SMILES rows."""

    summary = SUMMARY.read_text(encoding="utf-8")

    assert 'label: "連続値"' in summary
    assert 'label: "通常カテゴリ"' in summary
    assert 'label: "組成式"' in summary
    assert 'label: "分子表記（SMILES）"' in summary
    assert 'name === "num"' in summary
    assert 'name === "cat"' in summary
    assert 'name.startsWith("comp_")' in summary
    assert 'name.startsWith("smiles_")' in summary
    assert "COMMON_NAMES" in summary
    assert "rowSpan={summary.rows.length}" in summary
    assert "featureColumns" in summary


def test_model_summary_is_fixed_below_registered_model_heading() -> None:
    """The summary should remain directly below the registered-model title."""

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


def test_model_summary_table_has_fixed_columns_and_responsive_scroll() -> None:
    """The four-column table should stay readable without a connection graph."""

    result_css = RESULT_CSS.read_text(encoding="utf-8")
    summary_css = SUMMARY_CSS.read_text(encoding="utf-8")
    main = MAIN.read_text(encoding="utf-8")

    assert '.model-registration-panel:has(> .model-result-visualization-host) > .codebox' in result_css
    assert ".evaluation-result-panel" in result_css
    assert ".model-result-summary" in result_css
    assert ".model-result-metrics" in result_css
    assert ".model-structure-summary-scroll" in summary_css
    assert ".model-structure-summary-table" in summary_css
    assert "table-layout: fixed" in summary_css
    assert "overflow-x: auto" in summary_css
    assert ".model-summary-columns" in summary_css
    assert 'import "./model-structure-summary.css";' in main
