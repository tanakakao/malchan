from pathlib import Path


OPTIMIZE_PAGE = Path("frontend/src/pages/OptimizePage.jsx")
API = Path("frontend/src/api.js")
INVERSE_SCHEMA = Path("src/malchan/app/schemas/inverse_analysis.py")
INVERSE_SERVICE = Path("src/malchan/app/services/model_service.py")


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


def test_sum_constraint_is_configurable_and_sent_to_inverse_api() -> None:
    """The UI should expose the inverse function's numeric sum-equality constraint."""

    source = OPTIMIZE_PAGE.read_text(encoding="utf-8")
    schema_source = INVERSE_SCHEMA.read_text(encoding="utf-8")
    service_source = INVERSE_SERVICE.read_text(encoding="utf-8")

    assert "説明変数の合計制約" in source
    assert "合計制約を使用する" in source
    assert "sumConstraint.columns" in source
    assert "constraintRange.minimum" in source
    assert "constraintRange.maximum" in source
    assert "sum_constraint: sumConstraint.enabled" in source
    assert "value: Number(sumConstraint.value)" in source

    assert "sum_constraint: SumConstraint | None" in schema_source
    assert "request.sum_constraint.columns" in service_source
    assert "constraint_cols=constraint_cols" in service_source
    assert "constraint_value=constraint_value" in service_source


def test_sampler_options_change_for_single_and_multi_objective_search() -> None:
    """Single- and multi-objective searches should present compatible sampler lists."""

    source = OPTIMIZE_PAGE.read_text(encoding="utf-8")
    single_section = source.split("const SINGLE_OBJECTIVE_SAMPLERS = [", 1)[1].split(
        "const MULTI_OBJECTIVE_SAMPLERS = [",
        1,
    )[0]
    multi_section = source.split("const MULTI_OBJECTIVE_SAMPLERS = [", 1)[1].split(
        "const variableSettingsByModel",
        1,
    )[0]

    for sampler in ('value: "TPE"', 'value: "CmaEs"', 'value: "GP"', 'value: "QMS"'):
        assert sampler in single_section
    for sampler in ('value: "MOTPE"', 'value: "NSGAII"', 'value: "NSGAIII"'):
        assert sampler in multi_section

    assert 'value: "CmaEs"' not in multi_section
    assert 'value: "GP"' not in multi_section
    assert 'value: "QMS"' not in multi_section
    assert "const multiObjective = targets.length > 1" in source
    assert "const samplerOptions = multiObjective" in source
    assert "setSampler(samplerOptions[0].value)" in source
    assert 'multiObjective ? "多目的" : "単目的"' in source


def test_inverse_api_merges_page_specific_payload_override() -> None:
    """The context-owned execution should receive fixed values and custom steps."""

    source = API.read_text(encoding="utf-8")

    assert "setInverseAnalysisPayloadOverride" in source
    assert "inverseAnalysisPayloadOverride" in source
    assert "JSON.stringify(inverseAnalysisPayload(payload))" in source
