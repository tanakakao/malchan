"""Tests for OOF metrics exposed by the model evaluation API."""

import pytest

from malchan.app.services.model_configuration_service import _oof_metrics


def test_classification_oof_metrics_use_weighted_average() -> None:
    """Multiclass OOF metrics should include bochan-compatible summary keys."""

    records = [
        {"actual": "A", "predicted": "A"},
        {"actual": "A", "predicted": "B"},
        {"actual": "B", "predicted": "B"},
        {"actual": "C", "predicted": "C"},
    ]

    metrics = _oof_metrics("classification", records)

    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["precision"] == pytest.approx(0.875)
    assert metrics["recall"] == pytest.approx(0.75)
    assert metrics["f1"] == pytest.approx(0.75)


def test_regression_oof_metrics_include_common_errors() -> None:
    """Regression OOF metrics should match the values shown in Explain."""

    records = [
        {"actual": 1.0, "predicted": 1.0},
        {"actual": 2.0, "predicted": 3.0},
        {"actual": 3.0, "predicted": 2.0},
    ]

    metrics = _oof_metrics("regression", records)

    assert metrics["mae"] == pytest.approx(2 / 3)
    assert metrics["mse"] == pytest.approx(2 / 3)
    assert metrics["rmse"] == pytest.approx((2 / 3) ** 0.5)
    assert metrics["r2"] == pytest.approx(0.0)
    assert metrics["mape"] == pytest.approx((0 + 0.5 + 1 / 3) / 3)
