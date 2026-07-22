"""Model capability registry used by future LLM-assisted planning.

The registry is intentionally independent from any LLM provider. It gives
prompt builders, validators, and user interfaces a single typed view of the
models that malchan can construct for each task.
"""

from __future__ import annotations

import inspect
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

TaskType = Literal["regression", "classification", "AD"]
ComputationalCost = Literal["low", "medium", "high"]


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Metadata describing one model available to malchan.

    Args:
        name: Public model name accepted by the existing malchan pipeline.
        task: Learning task supported by the model.
        family: Coarse model family used for capability checks.
        default_params: Safe default estimator parameters.
        allowed_params: Estimator parameters that an LLM suggestion may set.
        requires_scaling: Whether numeric scaling is normally recommended.
        native_categorical: Whether the estimator can directly consume
            categorical values in the current malchan pipeline.
        supports_probability: Whether classification probabilities are
            available.
        supports_feature_importance: Whether a native feature importance or
            coefficient is normally available.
        optional_dependency: Optional package required by this model.
        recommended_min_samples: Soft lower bound used to emit warnings.
        computational_cost: Relative training cost.
        notes: Additional concise guidance for planners and user interfaces.
    """

    name: str
    task: TaskType
    family: str
    default_params: Mapping[str, Any] = field(default_factory=dict)
    allowed_params: frozenset[str] = field(default_factory=frozenset)
    requires_scaling: bool = False
    native_categorical: bool = False
    supports_probability: bool = False
    supports_feature_importance: bool = False
    optional_dependency: str | None = None
    recommended_min_samples: int = 2
    computational_cost: ComputationalCost = "medium"
    notes: tuple[str, ...] = field(default_factory=tuple)

    def accepts_parameter(self, name: str) -> bool:
        """Return whether ``name`` is an allowed estimator parameter."""

        return not self.allowed_params or name in self.allowed_params

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""

        data = asdict(self)
        data["default_params"] = dict(self.default_params)
        data["allowed_params"] = sorted(self.allowed_params)
        data["notes"] = list(self.notes)
        return data


class ModelSpecRegistry:
    """Registry of :class:`ModelSpec` objects keyed by task and model name."""

    def __init__(self, specs: Iterable[ModelSpec] | None = None) -> None:
        self._specs: dict[tuple[str, str], ModelSpec] = {}
        for spec in specs or []:
            self.register(spec)

    def register(self, spec: ModelSpec, *, replace: bool = False) -> None:
        """Register one model specification.

        Args:
            spec: Model specification to register.
            replace: Replace an existing specification with the same key.

        Raises:
            ValueError: If the key is already registered and ``replace`` is
                false.
        """

        key = (spec.task, spec.name)
        if key in self._specs and not replace:
            raise ValueError(
                f"Model specification is already registered for task={spec.task!r}, "
                f"name={spec.name!r}."
            )
        self._specs[key] = spec

    def get(self, task: str, name: str) -> ModelSpec:
        """Return a registered specification.

        Raises:
            KeyError: If the model is not registered for the requested task.
        """

        try:
            return self._specs[(task, name)]
        except KeyError as exc:
            raise KeyError(f"Unknown model {name!r} for task {task!r}.") from exc

    def find(self, task: str, name: str) -> ModelSpec | None:
        """Return a registered specification or ``None``."""

        return self._specs.get((task, name))

    def list(self, task: str | None = None) -> list[ModelSpec]:
        """Return registered specifications, optionally filtered by task."""

        specs = [
            spec
            for (registered_task, _), spec in self._specs.items()
            if task is None or registered_task == task
        ]
        return sorted(specs, key=lambda spec: (spec.task, spec.name))

    def names(self, task: str) -> list[str]:
        """Return registered model names for one task."""

        return [spec.name for spec in self.list(task)]

    def to_prompt_payload(self, task: str | None = None) -> list[dict[str, Any]]:
        """Return compact metadata suitable for a structured planner prompt."""

        return [spec.to_dict() for spec in self.list(task)]

    def __len__(self) -> int:
        return len(self._specs)


def _constructor_parameters(model_class: type[Any]) -> frozenset[str]:
    """Return explicit constructor parameter names for an estimator class."""

    try:
        signature = inspect.signature(model_class.__init__)
    except (TypeError, ValueError):
        return frozenset()
    return frozenset(
        name
        for name, parameter in signature.parameters.items()
        if name != "self"
        and parameter.kind
        not in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}
    )


def _model_family(name: str) -> str:
    normalized = name.lower()
    if "近傍" in name:
        return "neighbors"
    if "pls" in normalized:
        return "pls"
    if "ガウス過程" in name:
        return "gaussian_process"
    if "フォレスト" in name or "trees" in normalized:
        return "tree_ensemble"
    if any(token in normalized for token in ("boost", "xgboost", "lightgbm", "catboost")):
        return "boosting"
    if any(token in normalized for token in ("ridge", "lasso", "elastic")) or "線形" in name:
        return "linear"
    if "サポートベクター" in name:
        return "svm"
    if "多層パーセプトロン" in name:
        return "neural_network"
    if "決定木" in name or "回帰木" in name:
        return "tree"
    if "ナイーブベイズ" in name:
        return "naive_bayes"
    if name in {"OneClassSVM", "IsolationForest", "EllipticEnvelope"}:
        return "anomaly_detection"
    return "other"


def _optional_dependency(name: str) -> str | None:
    if name == "XGBoost":
        return "xgboost"
    if name == "LightGBM":
        return "lightgbm"
    if name == "CatBoost":
        return "catboost"
    return None


def build_runtime_model_registry() -> ModelSpecRegistry:
    """Build a registry from malchan's current model dictionaries.

    Heavy and optional model dependencies are imported only when this function
    is called. Importing :mod:`malchan.llm` itself therefore remains light.
    """

    from malchan.models.utils import (  # Imported lazily by design.
        AD_MODEL_DICT,
        CLS_MODEL_DICT,
        REG_MODEL_DICT,
        ad_default_params,
        cls_default_params,
        reg_default_params,
    )

    registry = ModelSpecRegistry()
    task_items = (
        ("regression", REG_MODEL_DICT, reg_default_params),
        ("classification", CLS_MODEL_DICT, cls_default_params),
        ("AD", AD_MODEL_DICT, ad_default_params),
    )
    for task, model_dict, default_params in task_items:
        for name, model_class in model_dict.items():
            family = _model_family(name)
            defaults = dict(default_params.get(name, {}))
            allowed = _constructor_parameters(model_class) | frozenset(defaults)
            registry.register(
                ModelSpec(
                    name=name,
                    task=task,
                    family=family,
                    default_params=defaults,
                    allowed_params=allowed,
                    requires_scaling=family
                    in {
                        "linear",
                        "neighbors",
                        "svm",
                        "neural_network",
                        "gaussian_process",
                        "pls",
                    },
                    native_categorical=name == "CatBoost",
                    supports_probability=(
                        task == "classification"
                        and hasattr(model_class, "predict_proba")
                    ),
                    supports_feature_importance=family
                    in {"linear", "tree", "tree_ensemble", "boosting"},
                    optional_dependency=_optional_dependency(name),
                    recommended_min_samples=(
                        20 if family in {"tree_ensemble", "boosting"} else 5
                    ),
                    computational_cost=(
                        "high"
                        if family in {"gaussian_process", "neural_network"}
                        else "medium"
                    ),
                )
            )
    return registry
