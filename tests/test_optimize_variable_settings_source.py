from pathlib import Path


OPTIMIZE_PAGE = Path("frontend/src/pages/OptimizePage.jsx")
API = Path("frontend/src/api.js")


def test_objective_control_uses_max_min_and_target_value() -> None:
    """Target value input should be visible only for the target-value mode."""

    source = OPTIMIZE_PAGE.read_text(encoding="utf-8")

    assert '<option value="max"' in source
    assert '<option value="min"' in source
    assert '<option value="target">目標値</option>' in source
    assert 'mode === "target" ? (' in source
    assert '<span className="muted-cell">入力不要</span>' in source
    assert "target_value" in source
    assert "direction: mode" in source


def test_search_variable_table_matches_bochan_style_settings() -> None:
    """Optimize should expose range, step, fixed flag, and fixed value together."""

    source = OPTIMIZE_PAGE.read_text(encoding="utf-8")

    for heading in ("変数", "型", "下限", "上限", "刻み", "固定", "固定値"):
        assert f"<th>{heading}</th>" in source

    assert "fixed-variable-row" in source
    assert "fixedValue" in source
    assert "fixed_values: fixedValues" in source
    assert "variableSettings[column]?.fixed" in source
    assert "setInverseAnalysisPayloadOverride(inversePayloadOverride())" in source


def test_inverse_api_merges_page_specific_payload_override() -> None:
    """The context-owned execution should receive fixed values and custom steps."""

    source = API.read_text(encoding="utf-8")

    assert "setInverseAnalysisPayloadOverride" in source
    assert "inverseAnalysisPayloadOverride" in source
    assert "JSON.stringify(inverseAnalysisPayload(payload))" in source
