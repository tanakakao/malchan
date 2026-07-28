from pathlib import Path


API = Path("frontend/src/api.js")
APP = Path("frontend/src/App.jsx")
CATEGORY_CONTROL = Path(
    "frontend/src/components/OptimizeCategoryCandidatesControl.jsx"
)
CATEGORY_STYLE = Path("frontend/src/optimize-category-candidates.css")
SHAP_CONTROL = Path("frontend/src/components/ShapScatterControl.jsx")
SHAP_STYLE = Path("frontend/src/shap-scatter-control.css")
VISUALIZATION_ROUTES = Path("src/malchan/app/api/visualization_routes.py")
WEB_PLOTS = Path("src/malchan/visualization/web_api_plots.py")
MACHINE_LEARNING_PLOTS = Path(
    "src/malchan/visualization/machine_learning_plots.py"
)


def test_optimize_categorical_candidates_use_dropdown_like_multi_select() -> None:
    """Categorical search values should be selectable as a non-empty subset."""

    source = CATEGORY_CONTROL.read_text(encoding="utf-8")
    style = CATEGORY_STYLE.read_text(encoding="utf-8")

    assert "カテゴリ探索候補" in source
    assert "CategoryMultiSelect" in source
    assert "category-candidate-trigger" in source
    assert "category-candidate-menu" in source
    assert 'type="checkbox"' in source
    assert 'type="search"' in source
    assert "候補を検索" in source
    assert "全選択" in source
    assert "探索候補は1つ以上必要です" in source
    assert "if (selected.length <= 1) return" in source
    assert "categoryValues(rows, column)" in source
    assert "Math.max(rows.length, 1)" in source
    assert "candidateSelectionsByModel" in source
    assert "fixed-variable-row" in source

    for class_name in (
        "optimize-category-candidates",
        "category-candidate-grid",
        "category-candidate-card",
        "category-candidate-trigger",
        "category-candidate-menu",
        "category-candidate-filter",
    ):
        assert f".{class_name}" in style


def test_selected_category_candidates_override_only_searchable_categories() -> None:
    """Selected candidates should not re-add fixed categorical variables."""

    api_source = API.read_text(encoding="utf-8")
    control_source = CATEGORY_CONTROL.read_text(encoding="utf-8")

    assert "setInverseCategoryCandidatesOverride" in api_source
    assert "inverseCategoryCandidatesOverride" in api_source
    assert "Object.prototype.hasOwnProperty.call(existingCategories, column)" in api_source
    assert "categories: {" in api_source
    assert "...existingCategories" in api_source
    assert "...selectableOverrides" in api_source

    assert "setInverseCategoryCandidatesOverride(selections)" in control_source
    assert "setInverseCategoryCandidatesOverride(null)" in control_source
    assert "固定中の変数は固定値が優先されます" in control_source


def test_explain_adds_shap_scatter_with_interactive_column_selection() -> None:
    """The PD panel should expose SHAP scatter and optional interaction colouring."""

    source = SHAP_CONTROL.read_text(encoding="utf-8")
    style = SHAP_STYLE.read_text(encoding="utf-8")
    api_source = API.read_text(encoding="utf-8")

    assert "PD 1D / 2D / SHAP散布図" in source
    assert "SHAP散布図" in source
    assert "SHAP特徴量" in source
    assert "interactive col" in source
    assert 'option value="">なし</option>' in source
    assert "interactive_col: interactiveColumn" in source
    assert "api.visualizationShapScatter" in source
    assert "SHAP出力" in source
    assert "show_shap_scatter" in source

    assert "visualizationShapScatter" in api_source
    assert 'visualizationPath(modelId, target, "shap-scatter", options)' in api_source

    assert ".shap-scatter-controlled" in style
    assert ".shap-scatter-toolbar" in style
    assert ".shap-scatter-result-card" in style


def test_new_controls_are_mounted_inside_workbench_provider() -> None:
    """Portal-based controls should have access to WorkbenchContext."""

    source = APP.read_text(encoding="utf-8")

    assert 'import OptimizeCategoryCandidatesControl from "./components/OptimizeCategoryCandidatesControl"' in source
    assert 'import ShapScatterControl from "./components/ShapScatterControl"' in source
    assert "<OptimizeCategoryCandidatesControl />" in source
    assert "<ShapScatterControl />" in source
    assert source.index("<OptimizeCategoryCandidatesControl />") < source.index("{toast &&")


def test_shap_scatter_route_reuses_existing_visualization_function() -> None:
    """FastAPI should delegate SHAP scatter geometry to show_shap_scatter."""

    route_source = VISUALIZATION_ROUTES.read_text(encoding="utf-8")
    adapter_source = WEB_PLOTS.read_text(encoding="utf-8")
    plot_source = MACHINE_LEARNING_PLOTS.read_text(encoding="utf-8")

    assert '"/models/{model_id}/visualizations/{target}/shap-scatter"' in route_source
    assert "interactive_col: str | None = Query(default=None)" in route_source
    assert "show_model_shap_scatter" in route_source
    assert '"visualization_function": "show_shap_scatter"' in route_source
    assert "target_item=selected_output" in route_source

    assert "def show_model_shap_scatter(" in adapter_source
    assert "show_shap_scatter(" in adapter_source
    assert "interactive_col=interactive_col" in adapter_source
    assert "target_item=target_item" in adapter_source

    assert "def show_shap_scatter(" in plot_source
    assert "interactive_col: Optional[str] = None" in plot_source
