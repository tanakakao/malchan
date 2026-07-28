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


def test_ensemble_control_supports_per_model_manual_parameters() -> None:
    """Each ensemble member should be editable through a compact model tab."""

    source = CONTROL.read_text(encoding="utf-8")

    assert '["manual", "各モデルで設定"]' in source
    assert "api.modelParameters(entry.task, entry.model)" in source
    assert "memberParamsByTarget" in source
    assert "baseParamsByTarget" in source
    assert 'role: "final"' in source
    assert 'className="ensemble-parameter-tabs"' in source
    assert "setActiveParameterTab(tab.key)" in source
    assert "<EnsembleParameterEditor" in source
    assert "既定値へ戻す" in source
    assert "選択中のモデルタブだけを表示しています" in source


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
    assert "manualParameters = parameterMode === \"manual\"" in source
    assert "normalizedModelParams" in source
    assert "normalizedBaseParams" in source
    assert "memberParamsByTarget[target]?.[model]" in source
    assert "baseParamsByTarget[target]" in source
    assert "model_params_by_target: manualParameters ? normalizedModelParams : {}" in source
    assert "base_model_params_by_target: manualParameters && usesBaseParameters" in source
    assert "model_params: manualParameters ? normalizedModelParams[target] : null" in source
    assert "base_model_param: manualParameters && usesBaseParameters" in source
    assert "JSON.stringify(ensembleTrainingPayload(payload))" in source
    assert "構成モデルを2件以上選択してください" in source
    assert "Stackingの最終モデルを選択してください" in source


def test_ensemble_styles_keep_manual_editor_compact() -> None:
    """Model tabs and a bounded scroll area should avoid a vertically long panel."""

    source = STYLE.read_text(encoding="utf-8")

    assert ".ensemble-parameter-tabs" in source
    assert "overflow-x: auto" in source
    assert ".ensemble-parameter-scroll" in source
    assert "max-height: 280px" in source
    assert "overflow-y: auto" in source
    assert ".ensemble-parameter-grid" in source


def test_ensemble_styles_replace_single_model_controls_only_when_enabled() -> None:
    """Single-model controls should remain visible until ensemble is enabled."""

    source = STYLE.read_text(encoding="utf-8")

    assert ".model-selection-panel.ensemble-active" in source
    assert ".ensemble-model-settings-host + .model-setting-section" in source
    assert ".model-workflow-content > .model-target-tabs" in source
    assert ".model-workflow-content > .model-target-tab-panel" in source
    assert 'content: attr(data-ensemble-summary)' in source
