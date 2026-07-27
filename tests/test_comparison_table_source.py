from pathlib import Path


def test_comparison_table_shows_best_model_evaluation_and_plot_data() -> None:
    """The comparison result should expose best CV metrics and plot-ready rows."""

    source = Path("frontend/src/components/ComparisonTable.jsx").read_text(encoding="utf-8")

    assert "result.best_cv_scores?.train" in source
    assert "result.best_cv_scores?.test" in source
    assert "result.best_cv_predictions?.train" in source
    assert "result.best_cv_predictions?.test" in source
    assert "ベストモデル精度評価" in source
    assert "Y-Y／残差" in source
