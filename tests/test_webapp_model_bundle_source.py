"""Source regression tests for browser-managed model files."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_BUNDLES = ROOT / "frontend" / "src" / "modelBundles.js"
CONTROL = ROOT / "frontend" / "src" / "components" / "ModelBundleControl.jsx"
CONTEXT = ROOT / "frontend" / "src" / "context" / "WorkbenchContext.jsx"
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
    assert 'accept=".malchan,application/vnd.malchan.model"' in control
    assert "<ModelBundleControl />" in app
    assert 'import "./model-bundle.css";' in main
    assert ".model-bundle-actions" in css


def test_loaded_model_restores_workbench_metadata_without_training_rows() -> None:
    """Import should activate the model and restore input groups without raw data."""

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


def test_fastapi_bundle_routes_require_signature_configuration() -> None:
    """The API should expose in-memory transfer routes and environment-backed signing."""

    routes = ROUTES.read_text(encoding="utf-8")
    settings = SETTINGS.read_text(encoding="utf-8")

    assert '"/models/{model_id}/export"' in routes
    assert '"/model-bundles/import"' in routes
    assert "bundle = await request.body()" in routes
    assert '"Cache-Control": "no-store"' in routes
    assert "MALCHAN_MODEL_BUNDLE_SECRET" in settings
    assert "MALCHAN_MODEL_BUNDLE_MAX_MB" in settings
