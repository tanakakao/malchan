from pathlib import Path


MODEL_PAGE = Path("frontend/src/pages/ModelPage.jsx")


def test_multi_target_model_settings_use_tabs() -> None:
    """Model selection and comparison should render one active target panel."""

    source = MODEL_PAGE.read_text(encoding="utf-8")

    assert "function TargetTabs" in source
    assert 'if (targets.length <= 1) return null;' in source
    assert 'role="tablist"' in source
    assert 'role="tabpanel"' in source
    assert source.count("<TargetTabs") == 2
    assert "targets.map((target) => (\n                  <section className=\"target-model-card\"" not in source
