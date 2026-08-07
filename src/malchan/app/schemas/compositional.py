"""FastAPI schema extensions for compositional preprocessing."""

from typing import Literal

from pydantic import Field, field_validator, model_validator

from .models import TrainModelRequest as BaseTrainModelRequest

CompositionalMethod = Literal["ILR", "CLR", "ALR"]
CompositionalScaleType = Literal[
    "StandardScaler",
    "MinMaxScaler",
    "centering",
    "MaxAbsScaler",
]


class TrainModelRequest(BaseTrainModelRequest):
    """Training request with simplex-aware compositional preprocessing options."""

    compositional_groups: list[list[str]] = Field(default_factory=list)
    compositional_method: CompositionalMethod | None = None
    compositional_zero_replacement: float | None = Field(default=1e-6, gt=0, lt=1)
    compositional_closure: bool = True
    compositional_alr_reference: int | str = -1
    compositional_scale_type: CompositionalScaleType | None = None

    @field_validator("compositional_method", mode="before")
    @classmethod
    def normalize_compositional_method(cls, value: object) -> object:
        """Accept case-insensitive log-ratio method names."""

        if value is None:
            return None
        if not isinstance(value, str):
            return value
        normalized = value.strip().upper()
        return normalized or None

    @field_validator("compositional_scale_type", mode="before")
    @classmethod
    def normalize_compositional_scale_type(cls, value: object) -> object:
        """Treat an empty Web-form scaler value as no scaling."""

        if isinstance(value, str) and not value.strip():
            return None
        return value

    @model_validator(mode="after")
    def validate_compositional_settings(self) -> "TrainModelRequest":
        """Validate independent sum-constrained feature groups."""

        normalized_groups: list[list[str]] = []
        seen: set[str] = set()
        numeric_columns = set(self.num_cols)

        for index, group in enumerate(self.compositional_groups):
            columns = [column.strip() for column in group]
            if len(columns) < 2:
                raise ValueError(
                    f"compositional_groups[{index}] must contain at least two columns."
                )
            if any(not column for column in columns):
                raise ValueError(
                    f"compositional_groups[{index}] must not contain blank column names."
                )
            if len(columns) != len(set(columns)):
                raise ValueError(
                    f"compositional_groups[{index}] must not contain duplicate columns."
                )

            overlap = seen.intersection(columns)
            if overlap:
                raise ValueError(
                    "A feature column cannot belong to multiple compositional groups: "
                    f"{sorted(overlap)}"
                )

            unknown = sorted(set(columns).difference(numeric_columns))
            if unknown:
                raise ValueError(
                    "Compositional groups must use columns listed in num_cols: "
                    f"{unknown}"
                )

            seen.update(columns)
            normalized_groups.append(columns)

        if normalized_groups and self.compositional_method is None:
            raise ValueError(
                "compositional_method is required when compositional_groups is specified."
            )

        if self.compositional_method == "ALR" and normalized_groups:
            reference = self.compositional_alr_reference
            if isinstance(reference, int):
                for index, group in enumerate(normalized_groups):
                    if not -len(group) <= reference < len(group):
                        raise ValueError(
                            "compositional_alr_reference is out of range for "
                            f"compositional_groups[{index}]."
                        )
            elif isinstance(reference, str):
                missing_reference = [
                    index
                    for index, group in enumerate(normalized_groups)
                    if reference not in group
                ]
                if missing_reference:
                    raise ValueError(
                        "A string compositional_alr_reference must exist in every "
                        f"compositional group; missing from groups {missing_reference}."
                    )

        self.compositional_groups = normalized_groups
        return self


__all__ = [
    "CompositionalMethod",
    "CompositionalScaleType",
    "TrainModelRequest",
]