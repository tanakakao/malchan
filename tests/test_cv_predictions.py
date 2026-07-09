import pandas as pd

from malchan.pipeline.single_output import SingleOutputMLModelPipeline


def test_aggregate_cv_train_predictions_uses_mode_for_classification_strings():
    """分類CVの学習予測は文字列ラベルでも最頻値で集約できる。"""
    pipeline = SingleOutputMLModelPipeline()
    pipeline.task = "classification"
    predicts = [
        pd.DataFrame({"y_cat_str": ["a", "b"], "index": [0, 1]}),
        pd.DataFrame({"y_cat_str": ["a", "c"], "index": [0, 1]}),
        pd.DataFrame({"y_cat_str": ["b", "c"], "index": [0, 1]}),
    ]

    aggregated = pipeline._aggregate_cv_train_predictions(predicts)

    expected = pd.DataFrame({"y_cat_str": ["a", "c"]}, index=pd.Index([0, 1], name="index"))
    pd.testing.assert_frame_equal(aggregated, expected)


def test_aggregate_cv_train_predictions_averages_regression_values():
    """回帰CVの学習予測は従来通り平均値で集約する。"""
    pipeline = SingleOutputMLModelPipeline()
    pipeline.task = "regression"
    predicts = [
        pd.DataFrame({"y": [1.0, 3.0], "index": [0, 1]}),
        pd.DataFrame({"y": [2.0, 5.0], "index": [0, 1]}),
    ]

    aggregated = pipeline._aggregate_cv_train_predictions(predicts)

    expected = pd.DataFrame({"y": [1.5, 4.0]}, index=pd.Index([0, 1], name="index"))
    pd.testing.assert_frame_equal(aggregated, expected)
