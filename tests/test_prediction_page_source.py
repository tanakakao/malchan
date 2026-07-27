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


def test_file_input_predicts_automatically_and_appends_prediction_columns() -> None:
    """Loading a file should predict immediately and show outputs at the table right edge."""

    source = PREDICTION_PAGE.read_text(encoding="utf-8")

    load_start = source.index("async function loadPredictionFile")
    shap_start = source.index("async function runSelectedShap")
    load_source = source[load_start:shap_start]

    assert "parseTabularFile" in load_source
    assert "await api.predict" in load_source
    assert "setFilePredictions(predictionResponse.predictions || [])" in load_source
    assert "runFilePrediction" not in source
    assert "fileDisplayRows" in source
    assert "fileDisplayColumns" in source
    assert "`予測_${source}`" in source
    assert "予測列を入力データフレームの右端へ追加" in source


def test_file_shap_uses_only_selected_rows() -> None:
    """File prediction should explain only rows selected by the user."""

    source = PREDICTION_PAGE.read_text(encoding="utf-8")
    api_source = API.read_text(encoding="utf-8")

    assert 'accept=".csv,.xlsx' in source
    assert "selectedRows" in source
    assert "const selected = indexes.map((index) => fileRows[index])" in source
    assert "requiredRecords(selected, features)" in source
    assert "api.localShap" in source
    assert "SHAPはチェックした行だけ計算" in source
    assert "/xai/local" in api_source


def test_custom_prediction_always_shows_prediction_and_shap_regions() -> None:
    """Custom mode should keep both prediction and SHAP output panels mounted."""

    source = PREDICTION_PAGE.read_text(encoding="utf-8")

    assert "Promise.all" in source
    assert "api.predict(modelInfo.model_id, { data })" in source
    assert "api.localShap(modelInfo.model_id, { data })" in source
    assert "<PredictionSummary prediction={customPrediction} alwaysVisible />" in source
    assert 'title="カスタム入力のSHAP"' in source
    assert "alwaysVisible" in source
