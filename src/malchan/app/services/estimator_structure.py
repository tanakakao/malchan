"""Convert fitted estimators into framework-independent structure nodes."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any

from malchan.app.schemas import EstimatorStructureNode

_MAX_DEPTH = 14
_MAX_CHILDREN = 24
_MAX_COLUMNS = 12
_MAX_PARAMETERS = 8
_MAX_TEXT = 72
_STRUCTURAL_PARAMS = {
    "steps",
    "transformers",
    "transformer_list",
    "estimators",
    "estimator",
    "base_estimator",
    "final_estimator",
}
_PRIORITY_PARAMS = (
    "strategy",
    "with_mean",
    "with_std",
    "drop",
    "handle_unknown",
    "n_components",
    "degree",
    "interaction_only",
    "n_estimators",
    "max_depth",
    "learning_rate",
    "kernel",
    "C",
    "alpha",
    "random_state",
)


def _text(value: Any) -> str:
    """Return short and stable display text."""

    if value is None:
        return "None"
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, (int, float, str)):
        text = str(value)
    elif callable(value):
        text = getattr(value, "__name__", type(value).__name__)
    elif isinstance(value, Mapping):
        text = "{" + ", ".join(str(key) for key in list(value)[:4]) + "}"
    elif isinstance(value, (list, tuple, set, frozenset)):
        values = list(value)
        preview = ", ".join(_text(item) for item in values[:5])
        suffix = f", … +{len(values) - 5}" if len(values) > 5 else ""
        text = f"[{preview}{suffix}]"
    else:
        text = type(value).__name__
    text = re.sub(r"0x[0-9A-Fa-f]+", "0x…", text)
    return text if len(text) <= _MAX_TEXT else f"{text[: _MAX_TEXT - 1]}…"


def _parameters(estimator: Any) -> dict[str, str]:
    """Select a compact set of non-structural parameters."""

    getter = getattr(estimator, "get_params", None)
    if not callable(getter):
        return {}
    try:
        raw = getter(deep=False)
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return {}
    if not isinstance(raw, Mapping):
        return {}

    available = {
        str(key): value
        for key, value in raw.items()
        if key not in _STRUCTURAL_PARAMS and value is not None
    }
    names = [name for name in _PRIORITY_PARAMS if name in available]
    names.extend(sorted(name for name in available if name not in names))
    return {name: _text(available[name]) for name in names[:_MAX_PARAMETERS]}


def _columns(selector: Any) -> list[str]:
    """Normalize a ColumnTransformer selector."""

    if selector is None:
        return []
    if isinstance(selector, str):
        return [selector]
    if isinstance(selector, slice):
        return [f"slice({selector.start}:{selector.stop}:{selector.step})"]
    if callable(selector):
        return [getattr(selector, "__name__", "callable selector")]
    if isinstance(selector, Iterable):
        try:
            values = list(selector)
        except TypeError:
            values = [selector]
    else:
        values = [selector]
    labels = [_text(value) for value in values[:_MAX_COLUMNS]]
    if len(values) > _MAX_COLUMNS:
        labels.append(f"… +{len(values) - _MAX_COLUMNS}")
    return labels


def _terminal(name: str, estimator: Any, note: str) -> EstimatorStructureNode:
    """Return a terminal reference node."""

    return EstimatorStructureNode(
        name=name,
        class_name=type(estimator).__name__,
        kind="reference",
        parameters={"note": note},
    )


def _limit(children: list[EstimatorStructureNode]) -> list[EstimatorStructureNode]:
    """Collapse unusually wide structures."""

    if len(children) <= _MAX_CHILDREN:
        return children
    remaining = len(children) - _MAX_CHILDREN
    return [
        *children[:_MAX_CHILDREN],
        EstimatorStructureNode(
            name=f"ほか {remaining} 件",
            class_name="CollapsedNodes",
            kind="reference",
            parameters={"note": "表示を簡潔にするため省略しました。"},
        ),
    ]


def build_estimator_structure(
    estimator: Any,
    *,
    name: str = "model",
    columns: Any = None,
    seen: set[int] | None = None,
    depth: int = 0,
) -> EstimatorStructureNode:
    """Build one recursive structure node from a fitted estimator."""

    if estimator is None:
        return EstimatorStructureNode(
            name=name,
            class_name="None",
            kind="dropped",
            columns=_columns(columns),
        )
    if isinstance(estimator, str):
        return EstimatorStructureNode(
            name=name,
            class_name=estimator,
            kind="passthrough" if estimator == "passthrough" else "dropped",
            columns=_columns(columns),
        )
    if depth >= _MAX_DEPTH:
        return _terminal(name, estimator, "最大表示階層に達しました。")

    visited = set() if seen is None else seen
    identity = id(estimator)
    if identity in visited:
        return _terminal(name, estimator, "上位階層と同じオブジェクトです。")
    visited = {*visited, identity}

    common = {
        "name": name,
        "class_name": type(estimator).__name__,
        "columns": _columns(columns),
        "parameters": _parameters(estimator),
    }

    steps = getattr(estimator, "steps", None)
    if isinstance(steps, (list, tuple)) and steps:
        children = [
            build_estimator_structure(
                child,
                name=str(step_name),
                seen=visited,
                depth=depth + 1,
            )
            for step_name, child in steps
        ]
        return EstimatorStructureNode(
            **common,
            kind="pipeline",
            children=_limit(children),
        )

    transformers = getattr(estimator, "transformers_", None)
    if not isinstance(transformers, (list, tuple)):
        transformers = getattr(estimator, "transformers", None)
    if isinstance(transformers, (list, tuple)) and transformers:
        children = [
            build_estimator_structure(
                transformer,
                name=str(transformer_name),
                columns=transformer_columns,
                seen=visited,
                depth=depth + 1,
            )
            for transformer_name, transformer, transformer_columns in transformers
        ]
        return EstimatorStructureNode(
            **common,
            kind="branch",
            children=_limit(children),
        )

    transformer_list = getattr(estimator, "transformer_list", None)
    if isinstance(transformer_list, (list, tuple)) and transformer_list:
        children = [
            build_estimator_structure(
                transformer,
                name=str(transformer_name),
                seen=visited,
                depth=depth + 1,
            )
            for transformer_name, transformer in transformer_list
        ]
        return EstimatorStructureNode(
            **common,
            kind="branch",
            children=_limit(children),
        )

    members = getattr(estimator, "estimators", None)
    if isinstance(members, (list, tuple)) and members:
        children = []
        for index, member in enumerate(members):
            if isinstance(member, tuple) and len(member) == 2:
                member_name, child = member
            else:
                member_name, child = f"model_{index + 1}", member
            children.append(
                build_estimator_structure(
                    child,
                    name=str(member_name),
                    seen=visited,
                    depth=depth + 1,
                )
            )
        return EstimatorStructureNode(
            **common,
            kind="ensemble",
            children=_limit(children),
        )

    kind = "estimator" if callable(getattr(estimator, "predict", None)) else "transformer"
    return EstimatorStructureNode(**common, kind=kind)


__all__ = ["build_estimator_structure"]
