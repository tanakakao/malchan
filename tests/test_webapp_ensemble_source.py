"""Source-level regression tests for the Web ensemble configuration."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "frontend" / "src" / "App.jsx"
API = ROOT / "frontend" / "src" / "api.js"
CONTROL = ROOT / "frontend" / "src" / "components" / "EnsembleModelSettingsControl.jsx"
STYLE = ROOT / "frontend" / "src" / "model-ensemble.css"


def test_app_mounts_ensemble_model_settings_control() -> None:
    """The workbench should mount the Model-page ensemble control."""

    source = APP.read_text(encoding="utf-8")

    assert 'import EnsembleModelSettingsControl from "./components/EnsembleModelSettingsControl"' in source
    assert "<EnsembleModelSettingsControl />" in source


def test_ensemble_control_exposes_all_supported_methods() -> None:
    """Voting, stacking, bagging, and boosting should be selectable."""

    source = CONTROL.read_text(encoding="utf-8")

    assert '["アンサンブル", "Voting"' in source
    assert '["スタッキング", "Stacking"' in source
    assert '["バギング", "Bagging"' in source
    assert '["ブースティング", "Boosting"' in source
    assert "<CheckboxList" in source
    assert "membersByTarget" in source
    assert "stackingBaseModel" in source
    assert 'disabled={value === "スタッキング" && mixedTasks}' in source


def test_ensemble_control_uses_event_driven_portal_without_dom_observer() -> None:
    """The control must not recreate the Optimize MutationObserver loop."""

    source = CONTROL.read_text(encoding="utf-8")

    assert "MutationObserver" not in source
    assert "requestAnimationFrame" in source
    assert "cancelAnimationFrame" in source
    assert 'contentRoot.addEventListener("click", scheduleResolve)' in source
    assert 'contentRoot.removeEventListener("click", scheduleResolve)' in source
    assert 'step !== "model" || !host' in source


def test_training_api_converts_web_ensemble_configuration() -> None:
    """The selected Web configuration should override only training payloads."""

    source = API.read_text(encoding="utf-8")

    assert "setEnsembleTrainingOptions" in source
    assert "function ensembleTrainingPayload(payload)" in source
    assert 'ensemble: true' in source
    assert 'ens_type: ensembleType' in source
    assert 'model_names_by_target: normalizedMembers' in source
    assert 'model_names: normalizedMembers[target] || []' in source
    assert 'model_params_by_target: {}' in source
    assert 'model_params: null' in source
    assert "JSON.stringify(ensembleTrainingPayload(payload))" in source
    assert "構成モデルを2件以上選択してください" in source
    assert "Stackingの最終モデルを選択してください" in source


def test_ensemble_styles_replace_single_model_controls_only_when_enabled() -> None:
    """Single-model controls should remain visible until ensemble is enabled."""

    source = STYLE.read_text(encoding="utf-8")

    assert ".model-selection-panel.ensemble-active" in source
    assert ".ensemble-model-settings-host + .model-setting-section" in source
    assert ".model-workflow-content > .model-target-tabs" in source
    assert ".model-workflow-content > .model-target-tab-panel" in source
    assert 'content: attr(data-ensemble-summary)' in source
