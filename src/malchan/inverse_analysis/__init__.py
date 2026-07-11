"""Public inverse-analysis API."""

from typing import Any

from .models import inverse_analysis as _inverse_analysis


def _resolve_model_value(model: Any, name: str) -> Any:
    """Resolve one model attribute from local or shared storage."""

    value = getattr(model, name, None)
    context = getattr(model, "context", None)
    if value is None and context is not None:
        value = getattr(context, name, None)
    return value


def _infer_category_candidates(model: Any) -> dict[str, list[Any]]:
    """Infer categorical-like search candidates from fitted training data."""

    training_data = _resolve_model_value(model, "X")
    if training_data is None and hasattr(model, "_get_X"):
        training_data = model._get_X()
    if training_data is None:
        return {}

    columns: list[str] = []
    for name in ("cat_cols", "smiles_cols", "comp_cols"):
        values = _resolve_model_value(model, name)
        if values is not None:
            columns.extend(list(values))

    candidates: dict[str, list[Any]] = {}
    for column in columns:
        values = training_data[column].dropna().drop_duplicates().tolist()
        if not values:
            raise ValueError(
                f"No observed category candidates are available for {column!r}."
            )
        candidates[column] = values
    return candidates


def inverse_analysis(model: Any, *args: Any, **kwargs: Any):
    """Run inverse analysis against a model or an application-layer adapter."""

    resolved_model = getattr(model, "_model", model)
    if len(args) < 2:
        inferred_categories = _infer_category_candidates(resolved_model)
        provided_categories = kwargs.get("cat_dict")
        if provided_categories is None:
            kwargs["cat_dict"] = inferred_categories
        else:
            merged_categories = dict(inferred_categories)
            merged_categories.update(dict(provided_categories))
            kwargs["cat_dict"] = merged_categories
    return _inverse_analysis(resolved_model, *args, **kwargs)


__all__ = ["inverse_analysis"]
