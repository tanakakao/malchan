"""Tests for creating the FastAPI application from Python and notebooks."""

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from malchan import __version__
from malchan.app import AppSettings, create_app, dataframe_to_records


def test_create_app_supports_in_process_notebook_usage() -> None:
    """Create a customized app and call it without starting a network server."""

    settings = AppSettings(
        api_prefix="/notebook-api",
        cors_origins=(),
        serve_frontend=False,
    )
    app = create_app(
        settings=settings,
        title="malchan Notebook API",
        version="0.2.0-notebook",
    )

    with TestClient(app) as client:
        response = client.get("/notebook-api/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "malchan Notebook API",
        "version": "0.2.0-notebook",
    }
    assert app.title == "malchan Notebook API"
    assert app.version == "0.2.0-notebook"
    assert app.state.settings is settings
    assert app.state.frontend_dist is None


def test_create_app_uses_package_version_by_default() -> None:
    """Use the installed package version when no override is supplied."""

    app = create_app(
        settings=AppSettings(cors_origins=(), serve_frontend=False),
    )

    with TestClient(app) as client:
        response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json()["version"] == __version__


def test_dataframe_to_records_normalizes_pandas_values() -> None:
    """Convert missing values and datetimes to JSON-compatible records."""

    df = pd.DataFrame(
        {
            "measured_at": pd.to_datetime(["2026-07-25 12:34:56", None]),
            "x1": [1.5, float("nan")],
            "category": pd.Series(["A", pd.NA], dtype="string"),
        }
    )

    records = dataframe_to_records(df)

    assert records == [
        {
            "measured_at": "2026-07-25T12:34:56.000",
            "x1": 1.5,
            "category": "A",
        },
        {
            "measured_at": None,
            "x1": None,
            "category": None,
        },
    ]


def test_dataframe_to_records_rejects_ambiguous_columns() -> None:
    """Reject columns that cannot be represented reliably as API fields."""

    duplicate_columns = pd.DataFrame([[1, 2]], columns=["x", "x"])
    with pytest.raises(ValueError, match="must be unique"):
        dataframe_to_records(duplicate_columns)

    integer_column = pd.DataFrame([[1]], columns=[0])
    with pytest.raises(TypeError, match="must be strings"):
        dataframe_to_records(integer_column)
