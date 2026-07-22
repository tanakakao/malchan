import pandas as pd

from malchan.llm import DatasetSummary


def test_dataset_summary_contains_statistics_without_raw_rows():
    df = pd.DataFrame(
        {
            "x1": [1.0, 2.0, 3.0, 4.0],
            "x2": [2.0, 4.0, 6.0, 8.0],
            "category": ["a", "a", "b", None],
            "y": [10.0, 12.0, 14.0, 16.0],
        }
    )

    summary = DatasetSummary.from_dataframe(
        df,
        target_cols=["y"],
        tasks=["regression"],
        num_cols=["x1", "x2"],
        cat_cols=["category"],
    )

    assert summary.n_rows == 4
    assert summary.n_features == 3
    assert summary.column("category").missing_count == 1
    assert summary.column("x1").mean == 2.5
    assert summary.strong_correlations[0]["correlation"] == 1.0
    assert "data" not in summary.to_dict()


def test_dataset_summary_tracks_class_counts():
    df = pd.DataFrame(
        {
            "x": [0.0, 1.0, 2.0, 3.0],
            "label": ["ok", "ok", "ng", "ok"],
        }
    )

    summary = DatasetSummary.from_dataframe(
        df,
        target_cols=["label"],
        tasks=["classification"],
        num_cols=["x"],
        cat_cols=[],
    )

    assert summary.column("label").class_counts == {"ok": 3, "ng": 1}
    assert summary.minimum_class_count("label") == 1
