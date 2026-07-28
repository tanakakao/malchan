"""Source regression tests for Web SMILES and composition settings."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STORE = ROOT / "frontend" / "src" / "materialFeatures.js"
KIND_CONTROL = ROOT / "frontend" / "src" / "components" / "MaterialFeatureKindControl.jsx"
DESCRIPTOR_CONTROL = (
    ROOT / "frontend" / "src" / "components" / "MaterialDescriptorSettingsControl.jsx"
)
API = ROOT / "frontend" / "src" / "api.js"
APP = ROOT / "frontend" / "src" / "App.jsx"
CSS = ROOT / "frontend" / "src" / "material-features.css"
KIND_CSS = ROOT / "frontend" / "src" / "material-feature-kind.css"


def test_selected_categorical_columns_can_choose_material_representation() -> None:
    """Selected category cards should expose inline normal, composition, and SMILES buttons."""

    source = KIND_CONTROL.read_text(encoding="utf-8")
    store = STORE.read_text(encoding="utf-8")

    assert 'step !== "prepare"' in source
    assert 'document.querySelectorAll(".feature-variable-choice")' in source
    assert 'host.className = "material-feature-kind-host"' in source
    assert 'className="material-feature-kind-options"' in source
    assert 'role="radiogroup"' in source
    assert 'role="radio"' in source
    assert "setMaterialFeatureKind(column, value)" in source
    assert 'categorical: "通常"' in source
    assert 'composition: "組成式"' in source
    assert 'smiles: "SMILES"' in source
    assert "<select" not in source
    assert '["categorical", "通常カテゴリ"]' in store
    assert '["composition", "組成式"]' in store
    assert '["smiles", "分子表記（SMILES）"]' in store


def test_material_kind_control_is_event_driven_without_dom_observer() -> None:
    """Prepare integration must not introduce a self-triggering DOM observer."""

    source = KIND_CONTROL.read_text(encoding="utf-8")

    assert "MutationObserver" not in source
    assert "requestAnimationFrame" in source
    assert "cancelAnimationFrame" in source
    assert 'contentRoot.addEventListener("click", scheduleConnect)' in source
    assert 'contentRoot.addEventListener("change", scheduleConnect)' in source
    assert 'contentRoot.removeEventListener("click", scheduleConnect)' in source
    assert 'contentRoot.removeEventListener("change", scheduleConnect)' in source


def test_preprocessing_shows_descriptor_settings_only_for_material_columns() -> None:
    """Model preprocessing should conditionally show compact descriptor tabs."""

    source = DESCRIPTOR_CONTROL.read_text(encoding="utf-8")

    assert 'settings.kinds[column] === "smiles"' in source
    assert 'settings.kinds[column] === "composition"' in source
    assert "hasMaterialFeatures" in source
    assert 'step !== "model" || !hasMaterialFeatures' in source
    assert 'document.querySelector(".preprocessing-panel .model-settings-stack")' in source
    assert 'className="material-descriptor-tabs"' in source
    assert "SmilesDescriptorPanel" in source
    assert "CompositionDescriptorPanel" in source
    assert "SMILES_FINGERPRINTS" in source
    assert "PYMATGEN_PROPERTIES" in source
    assert "MATMINER_DESCRIPTORS" in source
    assert "MENDELEEV_PROPERTIES" in source
    assert "MutationObserver" not in source


def test_web_composition_methods_default_to_pymatgen_and_hide_xenonpy() -> None:
    """Pymatgen should be the default app method and XenonPy should not be selectable."""

    source = DESCRIPTOR_CONTROL.read_text(encoding="utf-8")
    store = STORE.read_text(encoding="utf-8")

    assert 'compMethod: "pymatgen"' in store
    assert "pymatgenProperties" in store
    assert '<option value="pymatgen">Pymatgen基本統計</option>' in source
    assert '<option value="matminer">Matminer</option>' in source
    assert '<option value="mendeleev">Mendeleev元素プロパティ</option>' in source
    assert 'option value="xenonpy"' not in source
    assert "組成加重平均・標準偏差・最小・最大・範囲" in source


def test_training_payload_separates_material_columns_from_categories() -> None:
    """Training should map represented category columns to FastAPI material fields."""

    store = STORE.read_text(encoding="utf-8")
    api = API.read_text(encoding="utf-8")

    assert "applyMaterialFeatureTrainingPayload" in store
    assert "normalCategoricalColumns" in store
    assert "smiles_cols: smilesColumns" in store
    assert "fingerprints: smilesColumns.length ? settings.fingerprints : []" in store
    assert "comp_cols: compositionColumns" in store
    assert "comp_method: compositionColumns.length ? settings.compMethod : null" in store
    assert "comp_feats: compositionColumns.length ? compositionDescriptors : []" in store
    assert 'settings.compMethod === "pymatgen"' in store
    assert "compositionDescriptors = settings.pymatgenProperties" in store
    assert "SMILES列に使用する分子記述子を1件以上選択してください" in store
    assert "組成式列に使用する記述子を1件以上選択してください" in store
    assert "ensembleTrainingPayload(applyMaterialFeatureTrainingPayload(payload))" in api


def test_material_descriptor_layout_stays_compact() -> None:
    """Descriptor and material-kind choices should stay within their cards."""

    css = CSS.read_text(encoding="utf-8")
    kind_css = KIND_CSS.read_text(encoding="utf-8")

    assert ".material-descriptor-tabs" in css
    assert "grid-template-columns: repeat(2, minmax(0, 1fr))" in css
    assert ".material-descriptor-checklist" in css
    assert "max-height: 210px" in css
    assert "overflow-y: auto" in css
    assert ".material-feature-kind-host" in kind_css
    assert "grid-column: 1 / -1" in kind_css
    assert "grid-row: 2" in kind_css
    assert ".material-feature-kind-options" in kind_css
    assert "grid-template-columns: repeat(3, minmax(0, 1fr))" in kind_css
    assert ".material-feature-kind-option.composition.active" in kind_css
    assert ".material-feature-kind-option.smiles.active" in kind_css


def test_material_controls_are_mounted_in_the_workbench() -> None:
    """Both Prepare and Model material controls should be globally mounted."""

    source = APP.read_text(encoding="utf-8")

    assert "MaterialFeatureKindControl" in source
    assert "MaterialDescriptorSettingsControl" in source
    assert "<MaterialFeatureKindControl />" in source
    assert "<MaterialDescriptorSettingsControl />" in source
