"""Source regression tests for browser-managed model files."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_BUNDLES = ROOT / "frontend" / "src" / "modelBundles.js"
CONTROL = ROOT / "frontend" / "src" / "components" / "ModelBundleControl.jsx"
DEFAULTS = ROOT / "frontend" / "src" / "components" / "ImportedModelDefaultsControl.jsx"
CONTEXT = ROOT / "frontend" / "src" / "context" / "WorkbenchContext.jsx"
DATA = ROOT / "frontend" / "src" / "data.js"
APP = ROOT / "frontend" / "src" / "App.jsx"
MAIN = ROOT / "frontend" / "src" / "main.jsx"
CSS = ROOT / "frontend" / "src" / "model-bundle.css"
ROUTES = ROOT / "src" / "malchan" / "app" / "api" / "routes.py"
SETTINGS = ROOT / "src" / "malchan" / "app" / "core" / "settings.py"


def test_browser_downloads_and_uploads_raw_model_file_bytes() -> None:
    """The browser should use Blob download and raw request bodies without server paths."""

    source = MODEL_BUNDLES.read_text(encoding="utf-8")

    assert '/models/${encodeURIComponent(modelId)}/export' in source
    assert '`${API_BASE}/model-bundles/import`' in source
    assert 'body: file' in source
    assert 'await response.blob()' in source
    assert 'cache: "no-store"' in source
    assert 'application/vnd.malchan.model' in source
    assert "FormData" not in source


def test_model_file_card_is_available_on_model_page() -> None:
    """Model download and load controls should be mounted after model settings."""

    control = CONTROL.read_text(encoding="utf-8")
    app = APP.read_text(encoding="utf-8")
    main = MAIN.read_text(encoding="utf-8")
    css = CSS.read_text(encoding="utf-8")

    assert 'step !== "model"' in control
    assert 'contentRoot.querySelector(".model-settings-columns")' in control
    assert 'settings.insertAdjacentElement("afterend", nextHost)' in control
    assert "モデルをダウンロード" in control
    assert "モデルファイルを読み込む" in control
    assert "Server storage off" in control
    assert "作成元を信頼できるファイルだけ" in control
    assert "署名用の秘密値は不要" in control
    assert 'accept=".malchan,application/vnd.malchan.model"' in control
    assert "<ModelBundleControl />" in app
    assert 'import "./model-bundle.css";' in main
    assert ".model-bundle-actions" in css


def test_loaded_model_restores_metadata_without_displaying_training_rows() -> None:
    """Import should activate the model while keeping retained rows out of the data table."""

    context = CONTEXT.read_text(encoding="utf-8")

    assert "async function loadModelBundle(file)" in context
    assert "requestModelBundleImport(file)" in context
    assert "setRows([]);" in context
    assert "setNumFeatures(restoredNum);" in context
    assert "setCatFeatures(restoredCat);" in context
    assert "setTargets(restoredTargets);" in context
    assert "setModelInfo(info);" in context
    assert "downloadActiveModel" in context
    assert "URL.createObjectURL(result.blob)" in context


def test_loaded_model_training_rows_initialize_prediction_and_optimization_defaults() -> None:
    """Retained rows should drive defaults and categorical choices without table restore."""

    bundles = MODEL_BUNDLES.read_text(encoding="utf-8")
    defaults = DEFAULTS.read_text(encoding="utf-8")
    data = DATA.read_text(encoding="utf-8")
    app = APP.read_text(encoding="utf-8")

    assert "payload?.training_rows" in bundles
    assert "importedRowsByModel.set(modelId, trainingRows)" in bundles
    assert "export function importedModelRows(modelId)" in bundles
    assert "function medianValue(rows, column)" in defaults
    assert "function modeValue(rows, column)" in defaults
    assert "setPredictValues" in defaults
    assert "setBounds" in defaults
    assert "setObjectives" in defaults
    assert "numericSummary(rows, column)" in defaults
    assert "activeImportedModelRows()" in data
    assert "const resolvedLimit = usingImportedRows" in data
    assert "<ImportedModelDefaultsControl />" in app


def test_fastapi_bundle_routes_use_bounded_in_memory_artifacts() -> None:
    """The API should expose bounded transfer without signing configuration."""

    routes = ROUTES.read_text(encoding="utf-8")
    settings = SETTINGS.read_text(encoding="utf-8")

    assert '"/models/{model_id}/export"' in routes
    assert '"/model-bundles/import"' in routes
    assert "async for chunk in request.stream()" in routes
    assert "bundle = await _read_limited_body(request, configured_limit)" in routes
    assert '"Cache-Control": "no-store"' in routes
    assert "MALCHAN_MODEL_BUNDLE_SECRET" not in settings
    assert "MALCHAN_MODEL_BUNDLE_MAX_MB" in settings
