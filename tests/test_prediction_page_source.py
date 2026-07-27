from pathlib import Path


APP = Path("frontend/src/App.jsx")
PREDICTION_PAGE = Path("frontend/src/pages/PredictionPage.jsx")
OPTIMIZE_PAGE = Path("frontend/src/pages/OptimizePage.jsx")
API = Path("frontend/src/api.js")


def test_prediction_and_inverse_analysis_are_separate_pages() -> None:
    """Predict must appear before the inverse-analysis-only Optimize page."""

    app_source = APP.read_text(encoding="utf-8")
    optimize_source = OPTIMIZE_PAGE.read_text(encoding="utf-8")

    assert '["predict", "Predict", "予測・ローカルSHAP"]' in app_source
    assert '["optimize", "Optimize", "逆解析"]' in app_source
    assert app_source.index('["predict", "Predict"') < app_source.index('["optimize", "Optimize"')
    assert "PredictionPage" in app_source
    assert "任意条件で予測" not in optimize_source
    assert "predictOne" not in optimize_source
    assert "逆解析を実行" in optimize_source


def test_prediction_page_supports_files_and_selected_row_shap() -> None:
    """File prediction should explain only rows selected by the user."""

    source = PREDICTION_PAGE.read_text(encoding="utf-8")
    api_source = API.read_text(encoding="utf-8")

    assert "parseTabularFile" in source
    assert 'accept=".csv,.xlsx,.xls' in source
    assert "selectedRows" in source
    assert "requiredRecords(selected, features)" in source
    assert "api.predict" in source
    assert "api.localShap" in source
    assert "全行予測とSHAP計算を分離" in source
    assert "/xai/local" in api_source


def test_custom_prediction_always_requests_local_shap() -> None:
    """Every custom prediction action should calculate SHAP for that one row."""

    source = PREDICTION_PAGE.read_text(encoding="utf-8")

    assert "Promise.all" in source
    assert "api.predict(modelInfo.model_id, { data })" in source
    assert "api.localShap(modelInfo.model_id, { data })" in source
    assert "予測ボタンを押すたびに" in source
