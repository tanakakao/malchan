from pathlib import Path


OPTIMIZE_PAGE = Path("frontend/src/pages/OptimizePage.jsx")
OPTIMIZE_STYLE = Path("frontend/src/optimize-variable-settings.css")
API = Path("frontend/src/api.js")
INVERSE_SCHEMA = Path("src/malchan/app/schemas/inverse_analysis.py")
INVERSE_SERVICE = Path("src/malchan/app/services/model_service.py")
INVERSE_FUNCTION = Path("src/malchan/inverse_analysis/models.py")


def test_objective_control_uses_direction_constraint_and_target_value() -> None:
    """The bochan-style objective table should separate direction and constraint."""

    source = OPTIMIZE_PAGE.read_text(encoding="utf-8")

    for heading in (
        "目的変数",
        "最適化対象",
        "方向",
        "制約",
        "しきい値 / 目標値",
        "対象クラス",
    ):
        assert f"<th>{heading}</th>" in source

    assert '<option value="max">最大化</option>' in source
    assert '<option value="min">最小化</option>' in source
    assert '<option value="none">なし</option>' in source
    assert '<option value="target">目標値</option>' in source
    assert 'mode === "target" && !classification' in source
    assert "changeObjectiveConstraint" in source
    assert "target_value" in source
    assert "direction: mode" in source


def test_inverse_objective_targets_are_selectable_and_filtered() -> None:
    """Only checked targets should be sent to the existing target-cols path."""

    source = OPTIMIZE_PAGE.read_text(encoding="utf-8")
    service_source = INVERSE_SERVICE.read_text(encoding="utf-8")
    function_source = INVERSE_FUNCTION.read_text(encoding="utf-8")

    assert "objectiveSelectionByModel" in source
    assert "selectedTargets" in source
    assert "<th>最適化対象</th>" in source
    assert "逆解析に使用" in source
    assert "selectedTargets.map((target) =>" in source
    assert "selectedTargets.length > 1" in source
    assert "逆解析に使用する目的変数を1つ以上選択してください" in source
    assert "busy || selectedTargets.length === 0" in source

    assert "objective_targets = [item.target for item in request.objectives]" in service_source
    assert "target_cols=objective_targets" in service_source
    assert "target_cols: list[str] | None = None" in function_source
    assert "resolved_targets = normalized_model.target_cols if not target_cols else target_cols" in function_source
    assert "normalized_model.validate_objectives(resolved_targets, obj_directions)" in function_source


def test_search_variable_table_matches_bochan_style_settings() -> None:
    """Optimize should expose range, step, fixed flag, and fixed value together."""

    source = OPTIMIZE_PAGE.read_text(encoding="utf-8")

    assert "探索変数（検索空間）" in source
    for heading in ("変数", "型", "下限", "上限", "刻み", "固定", "固定値"):
        assert f"<th>{heading}</th>" in source

    assert "fixed-variable-row" in source
    assert "fixedValue" in source
    assert "fixed_values: fixedValues" in source
    assert "variableSettings[column]?.fixed" in source
    assert "setInverseAnalysisPayloadOverride(inversePayloadOverride())" in source


def test_optimize_page_uses_bochan_style_header_cards_and_primary_action() -> None:
    """The page should expose the requested summary, cards, and prominent action."""

    source = OPTIMIZE_PAGE.read_text(encoding="utf-8")
    style = OPTIMIZE_STYLE.read_text(encoding="utf-8")

    for class_name in (
        "optimize-hero",
        "optimize-summary-strip",
        "optimize-section-card",
        "optimize-section-title",
        "optimize-primary-run",
        "optimize-bottom-action",
    ):
        assert class_name in source
        assert f".{class_name}" in style

    assert "Optimize" in source
    assert "目的変数" in source
    assert "探索変数" in source
    assert "設定ステータス" in source
    assert "逆解析を実行" in source
    assert "詳細設定" in source
    assert "準備完了" in source


def test_sum_constraint_is_configurable_and_sent_to_inverse_api() -> None:
    """The UI should expose the inverse function's numeric sum-equality constraint."""

    source = OPTIMIZE_PAGE.read_text(encoding="utf-8")
    schema_source = INVERSE_SCHEMA.read_text(encoding="utf-8")
    service_source = INVERSE_SERVICE.read_text(encoding="utf-8")

    assert 'title="合計制約"' in source
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


def test_sampler_options_change_for_selected_objective_count() -> None:
    """Sampler choices should follow the number of checked objective targets."""

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
    assert "const multiObjective = selectedTargets.length > 1" in source
    assert "const samplerOptions = multiObjective" in source
    assert "setSampler(samplerOptions[0].value)" in source
    assert 'multiObjective ? "多目的" : "単目的"' in source


def test_inverse_api_merges_page_specific_payload_override() -> None:
    """The context-owned execution should receive page-specific inverse settings."""

    source = API.read_text(encoding="utf-8")

    assert "setInverseAnalysisPayloadOverride" in source
    assert "inverseAnalysisPayloadOverride" in source
    assert "JSON.stringify(inverseAnalysisPayload(payload))" in source
