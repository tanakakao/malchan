"""Source regression tests for model defaults and active preprocessing cards."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / "frontend" / "src" / "components" / "ModelSettingDefaultsControl.jsx"
APP = ROOT / "frontend" / "src" / "App.jsx"
MAIN = ROOT / "frontend" / "src" / "main.jsx"
ACTIVE_CSS = ROOT / "frontend" / "src" / "model-active-settings.css"


def test_parameter_setting_defaults_to_manual_once_per_model_entry() -> None:
    """Model and ensemble parameter controls should initialize to manual settings once."""

    source = CONTROL.read_text(encoding="utf-8")
    app = APP.read_text(encoding="utf-8")

    assert "useLayoutEffect" in source
    assert 'modelDefaultApplied = useRef(false)' in source
    assert 'ensembleDefaultApplied = useRef(false)' in source
    assert '".parameter-mode-switch button"' in source
    assert '"個別設定"' in source
    assert 'manualButton.click()' in source
    assert 'option[value=\'manual\']' in source
    assert 'setNativeSelectValue(ensembleParameterSelect, "manual")' in source
    assert 'if (!ensembleEnabled)' in source
    assert "MutationObserver" not in source
    assert "ModelSettingDefaultsControl" in app
    assert "<ModelSettingDefaultsControl />" in app


def test_active_preprocessing_cards_use_subtle_state_tint() -> None:
    """Enabled preprocessing cards should be highlighted without JavaScript observers."""

    css = ACTIVE_CSS.read_text(encoding="utf-8")
    main = MAIN.read_text(encoding="utf-8")

    assert '.preprocessing-panel .model-setting-section:has(input[type="checkbox"]:checked)' in css
    assert 'select > option:checked:not([value=""])' in css
    assert "color-mix(in srgb, var(--primary-soft) 30%, var(--surface-subtle))" in css
    assert "inset 3px 0 0" in css
    assert 'import "./model-active-settings.css";' in main
