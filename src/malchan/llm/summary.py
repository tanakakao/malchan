"""Local dataset summaries for LLM-assisted configuration planning."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

ColumnRole = Literal["feature", "target"]
ColumnKind = Literal["numeric", "categorical", "smiles", "composition", "other"]
TaskType = Literal["regression", "classification"]


def _optional_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _json_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if pd.isna(value):
        return None
    return value


@dataclass(frozen=True, slots=True)
class ColumnSummary:
    """Statistical summary of one feature or target column."""

    name: str
    role: ColumnRole
    kind: ColumnKind
    dtype: str
    missing_count: int
    missing_rate: float
    unique_count: int
    is_constant: bool
    high_cardinality: bool = False
    numeric_min: float | None = None
    numeric_max: float | None = None
    mean: float | None = None
    std: float | None = None
    skewness: float | None = None
    class_counts: dict[str, int] = field(default_factory=dict)
    top_values: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly dictionary."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class DatasetSummary:
    """Compact metadata sent to a future LLM planner instead of raw rows."""

    n_rows: int
    n_columns: int
    feature_columns: tuple[str, ...]
    target_columns: tuple[str, ...]
    tasks_by_target: dict[str, TaskType]
    numeric_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    smiles_columns: tuple[str, ...]
    composition_columns: tuple[str, ...]
    columns: tuple[ColumnSummary, ...]
    strong_correlations: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def n_features(self) -> int:
        """Return the number of raw feature columns."""

        return len(self.feature_columns)

    def column(self, name: str) -> ColumnSummary:
        """Return a column summary by name."""

        for column in self.columns:
            if column.name == name:
                return column
        raise KeyError(f"Unknown summarized column: {name!r}.")

    def minimum_class_count(self, target: str) -> int | None:
        """Return the smallest observed class count for a classification target."""

        counts = self.column(target).class_counts
        return min(counts.values()) if counts else None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""

        return {
            "n_rows": self.n_rows,
            "n_columns": self.n_columns,
            "n_features": self.n_features,
            "feature_columns": list(self.feature_columns),
            "target_columns": list(self.target_columns),
            "tasks_by_target": dict(self.tasks_by_target),
            "numeric_columns": list(self.numeric_columns),
            "categorical_columns": list(self.categorical_columns),
            "smiles_columns": list(self.smiles_columns),
            "composition_columns": list(self.composition_columns),
            "columns": [column.to_dict() for column in self.columns],
            "strong_correlations": [
                dict(item) for item in self.strong_correlations
            ],
            "warnings": list(self.warnings),
        }

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        *,
        target_cols: list[str],
        tasks: list[TaskType],
        num_cols: list[str],
        cat_cols: list[str],
        smiles_cols: list[str] | None = None,
        comp_cols: list[str] | None = None,
        correlation_threshold: float = 0.9,
        high_cardinality_threshold: float = 0.5,
        max_top_values: int = 10,
    ) -> "DatasetSummary":
        """Summarize a dataframe without retaining raw data rows."""

        smiles_cols = list(smiles_cols or [])
        comp_cols = list(comp_cols or [])
        target_cols = list(target_cols)
        tasks = list(tasks)
        num_cols = list(num_cols)
        cat_cols = list(cat_cols)

        if len(target_cols) != len(tasks):
            raise ValueError("target_cols and tasks must have the same length.")
        if len(target_cols) != len(set(target_cols)):
            raise ValueError("target_cols must not contain duplicates.")

        feature_columns = num_cols + cat_cols + smiles_cols + comp_cols
        declared_columns = feature_columns + target_cols
        if len(declared_columns) != len(set(declared_columns)):
            raise ValueError(
                "Feature and target column declarations must not overlap or repeat."
            )
        missing_columns = sorted(set(declared_columns).difference(df.columns))
        if missing_columns:
            raise ValueError(
                f"Dataframe is missing declared columns: {missing_columns}"
            )
        if not 0.0 <= correlation_threshold <= 1.0:
            raise ValueError("correlation_threshold must be between 0 and 1.")
        if high_cardinality_threshold < 0.0:
            raise ValueError(
                "high_cardinality_threshold must be non-negative."
            )

        tasks_by_target = dict(zip(target_cols, tasks, strict=True))
        kind_by_column: dict[str, ColumnKind] = {
            **{name: "numeric" for name in num_cols},
            **{name: "categorical" for name in cat_cols},
            **{name: "smiles" for name in smiles_cols},
            **{name: "composition" for name in comp_cols},
        }
        for target, task in tasks_by_target.items():
            kind_by_column[target] = (
                "numeric" if task == "regression" else "categorical"
            )

        column_summaries: list[ColumnSummary] = []
        warnings: list[str] = []
        for name in declared_columns:
            series = df[name]
            role: ColumnRole = (
                "target" if name in tasks_by_target else "feature"
            )
            kind = kind_by_column.get(name, "other")
            missing_count = int(series.isna().sum())
            non_missing = series.dropna()
            unique_count = int(non_missing.nunique(dropna=True))
            high_cardinality = (
                role == "feature"
                and kind in {"categorical", "smiles", "composition"}
                and len(non_missing) > 0
                and unique_count / len(non_missing)
                >= high_cardinality_threshold
            )

            numeric_min = numeric_max = mean = std = skewness = None
            if kind == "numeric" and len(non_missing) > 0:
                numeric = pd.to_numeric(
                    non_missing, errors="coerce"
                ).dropna()
                if len(numeric) > 0:
                    numeric_min = _optional_float(numeric.min())
                    numeric_max = _optional_float(numeric.max())
                    mean = _optional_float(numeric.mean())
                    std = _optional_float(numeric.std(ddof=1))
                    skewness = _optional_float(numeric.skew())

            class_counts: dict[str, int] = {}
            if (
                role == "target"
                and tasks_by_target[name] == "classification"
            ):
                class_counts = {
                    str(_json_scalar(key)): int(value)
                    for key, value in non_missing.value_counts(
                        dropna=False
                    ).items()
                }

            top_values = {
                str(_json_scalar(key)): int(value)
                for key, value in non_missing.value_counts(dropna=False)
                .head(max_top_values)
                .items()
            }
            column_summaries.append(
                ColumnSummary(
                    name=name,
                    role=role,
                    kind=kind,
                    dtype=str(series.dtype),
                    missing_count=missing_count,
                    missing_rate=(
                        missing_count / len(df) if len(df) else 0.0
                    ),
                    unique_count=unique_count,
                    is_constant=unique_count <= 1,
                    high_cardinality=high_cardinality,
                    numeric_min=numeric_min,
                    numeric_max=numeric_max,
                    mean=mean,
                    std=std,
                    skewness=skewness,
                    class_counts=class_counts,
                    top_values=top_values,
                )
            )

            if missing_count:
                warnings.append(
                    f"Column {name!r} contains {missing_count} missing values."
                )
            if unique_count <= 1:
                warnings.append(
                    f"Column {name!r} is constant or entirely missing."
                )
            if high_cardinality:
                warnings.append(f"Column {name!r} has high cardinality.")

        strong_correlations: list[dict[str, Any]] = []
        numeric_features = [
            name
            for name in num_cols
            if pd.api.types.is_numeric_dtype(df[name])
        ]
        if len(numeric_features) >= 2:
            correlations = df[numeric_features].corr(numeric_only=True)
            for left_index, left in enumerate(numeric_features):
                for right in numeric_features[left_index + 1 :]:
                    value = correlations.loc[left, right]
                    if (
                        pd.notna(value)
                        and abs(float(value)) >= correlation_threshold
                    ):
                        strong_correlations.append(
                            {
                                "left": left,
                                "right": right,
                                "correlation": float(value),
                            }
                        )

        if len(df) < 20:
            warnings.append(
                "The dataset has fewer than 20 rows; prefer simple models "
                "and conservative validation."
            )
        if len(feature_columns) >= max(len(df), 1):
            warnings.append(
                "The number of features is greater than or equal to the "
                "number of rows."
            )

        return cls(
            n_rows=int(len(df)),
            n_columns=int(df.shape[1]),
            feature_columns=tuple(feature_columns),
            target_columns=tuple(target_cols),
            tasks_by_target=tasks_by_target,
            numeric_columns=tuple(num_cols),
            categorical_columns=tuple(cat_cols),
            smiles_columns=tuple(smiles_cols),
            composition_columns=tuple(comp_cols),
            columns=tuple(column_summaries),
            strong_correlations=tuple(strong_correlations),
            warnings=tuple(dict.fromkeys(warnings)),
        )
