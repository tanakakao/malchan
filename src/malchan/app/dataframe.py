"""DataFrame conversion helpers for the FastAPI transport layer."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd


def dataframe_to_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a pandas DataFrame to JSON-compatible API records.

    FastAPI request bodies are JSON, so a DataFrame cannot be passed directly
    to ``TestClient.post(..., json=...)``. This helper preserves the familiar
    DataFrame-first workflow while normalizing pandas and NumPy scalar values,
    missing values, and datetime columns to values accepted by a JSON client.

    Args:
        df: DataFrame whose rows will be sent in a FastAPI request body.

    Returns:
        Row-oriented dictionaries suitable for a request ``data`` field.

    Raises:
        TypeError: If ``df`` is not a DataFrame or its column names are not
            strings.
        ValueError: If the DataFrame contains duplicate column names.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas.DataFrame.")
    if not df.columns.is_unique:
        duplicates = df.columns[df.columns.duplicated()].tolist()
        raise ValueError(f"DataFrame columns must be unique: {duplicates}")

    non_string_columns = [column for column in df.columns if not isinstance(column, str)]
    if non_string_columns:
        raise TypeError(
            "DataFrame column names must be strings before sending them to the API: "
            f"{non_string_columns}"
        )

    records = json.loads(
        df.to_json(
            orient="records",
            date_format="iso",
            date_unit="ms",
        )
    )
    return records
