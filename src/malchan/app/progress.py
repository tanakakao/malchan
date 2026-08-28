"""Process-local detailed progress tracking for Web/API workflows.

The core malchan pipelines remain unaware of HTTP.  When the FastAPI application is
running, :func:`install_progress_instrumentation` attaches small wrappers around the
existing training, cross-validation, and Optuna entry points so real counters can be
observed without changing the public model APIs.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from datetime import datetime, timezone
from functools import wraps
import re
from threading import Lock, RLock
from time import monotonic
from typing import Any, Iterator

PROGRESS_HEADER = "X-Malchan-Progress-ID"

_PROGRESS_ID_RE = re.compile(r"^[A-Za-z0-9._:-]{8,128}$")
_PROGRESS_TTL_SECONDS = 300.0
_PROGRESS_MAX_ITEMS = 256

_current_progress_id: ContextVar[str | None] = ContextVar(
    "malchan_current_progress_id",
    default=None,
)
_cv_state: ContextVar[dict[str, Any] | None] = ContextVar(
    "malchan_cv_progress_state",
    default=None,
)
_inverse_trial_state: ContextVar[dict[str, Any] | None] = ContextVar(
    "malchan_inverse_trial_progress_state",
    default=None,
)

_lock = RLock()
_progress: dict[str, dict[str, Any]] = {}
_instrumentation_installed = False


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_progress_id(progress_id: str | None) -> str | None:
    if not progress_id:
        return None
    value = str(progress_id).strip()
    return value if _PROGRESS_ID_RE.fullmatch(value) else None


def _cleanup_locked() -> None:
    now = monotonic()
    expired = [
        progress_id
        for progress_id, item in _progress.items()
        if now - float(item.get("_touched_monotonic", now)) > _PROGRESS_TTL_SECONDS
    ]
    for progress_id in expired:
        _progress.pop(progress_id, None)

    overflow = len(_progress) - _PROGRESS_MAX_ITEMS
    if overflow <= 0:
        return
    oldest = sorted(
        _progress,
        key=lambda key: float(_progress[key].get("_touched_monotonic", now)),
    )[:overflow]
    for progress_id in oldest:
        _progress.pop(progress_id, None)


def _touch_locked(item: dict[str, Any]) -> None:
    item["updated_at"] = _utc_now()
    item["_touched_monotonic"] = monotonic()


def current_progress_id() -> str | None:
    """Return the progress identifier bound to the current request context."""

    return _current_progress_id.get()


def _resolve_item(progress_id: str | None = None) -> tuple[str | None, dict[str, Any] | None]:
    resolved_id = _normalize_progress_id(progress_id) or current_progress_id()
    if resolved_id is None:
        return None, None
    with _lock:
        item = _progress.get(resolved_id)
        return resolved_id, item


@contextmanager
def progress_scope(progress_id: str | None, operation: str) -> Iterator[str | None]:
    """Bind one HTTP request to a process-local progress record."""

    resolved_id = _normalize_progress_id(progress_id)
    if resolved_id is None:
        yield None
        return

    now = _utc_now()
    with _lock:
        _cleanup_locked()
        _progress[resolved_id] = {
            "progress_id": resolved_id,
            "operation": operation,
            "status": "running",
            "started_at": now,
            "updated_at": now,
            "completed_at": None,
            "dimensions": {},
            "_target_plan": [],
            "_touched_monotonic": monotonic(),
        }
    token = _current_progress_id.set(resolved_id)
    try:
        yield resolved_id
    except BaseException:
        finish_progress("error", progress_id=resolved_id)
        raise
    else:
        _, item = _resolve_item(resolved_id)
        if item is not None and item.get("status") == "running":
            finish_progress("success", progress_id=resolved_id)
    finally:
        _current_progress_id.reset(token)


def finish_progress(status: str, *, progress_id: str | None = None) -> None:
    """Mark one operation complete while retaining its snapshot briefly for polling."""

    resolved_id = _normalize_progress_id(progress_id) or current_progress_id()
    if resolved_id is None:
        return
    with _lock:
        item = _progress.get(resolved_id)
        if item is None:
            return
        item["status"] = "error" if status == "error" else "success"
        item["completed_at"] = _utc_now()
        _touch_locked(item)


def get_progress_snapshot(progress_id: str) -> dict[str, Any] | None:
    """Return a JSON-safe copy of the latest detailed progress snapshot."""

    resolved_id = _normalize_progress_id(progress_id)
    if resolved_id is None:
        return None
    with _lock:
        _cleanup_locked()
        item = _progress.get(resolved_id)
        if item is None:
            return None
        snapshot = {
            key: value
            for key, value in item.items()
            if not key.startswith("_")
        }
        return deepcopy(snapshot)


def report_dimension(
    kind: str,
    current: int,
    total: int,
    *,
    label: str | None = None,
    detail: str | None = None,
    progress_id: str | None = None,
) -> None:
    """Publish one real counter such as target, Optuna trial, or CV fold."""

    resolved_id = _normalize_progress_id(progress_id) or current_progress_id()
    if resolved_id is None:
        return
    resolved_total = max(0, int(total))
    resolved_current = max(0, int(current))
    if resolved_total:
        resolved_current = min(resolved_current, resolved_total)

    with _lock:
        item = _progress.get(resolved_id)
        if item is None:
            return
        item["dimensions"][str(kind)] = {
            "current": resolved_current,
            "total": resolved_total,
            "label": "" if label is None else str(label),
            "detail": "" if detail is None else str(detail),
        }
        _touch_locked(item)


def set_target_plan(targets: list[str] | tuple[str, ...]) -> None:
    """Register the ordered target list for a multi-output operation."""

    progress_id = current_progress_id()
    if progress_id is None:
        return
    plan = [str(target) for target in targets if target is not None]
    with _lock:
        item = _progress.get(progress_id)
        if item is None:
            return
        item["_target_plan"] = plan
        _touch_locked(item)
    if plan:
        report_dimension("target", 0, len(plan), label=plan[0])


def mark_target(target: str | None) -> None:
    """Mark the target currently being trained or evaluated."""

    if target in (None, "AD"):
        return
    progress_id = current_progress_id()
    if progress_id is None:
        return
    target_name = str(target)
    with _lock:
        item = _progress.get(progress_id)
        if item is None:
            return
        plan = list(item.get("_target_plan") or [])
    if target_name in plan:
        current = plan.index(target_name) + 1
        total = len(plan)
    else:
        current = 1
        total = max(1, len(plan))
    report_dimension("target", current, total, label=target_name)


def _argument(args: tuple[Any, ...], kwargs: dict[str, Any], name: str, position: int, default: Any = None) -> Any:
    if name in kwargs:
        return kwargs[name]
    return args[position] if len(args) > position else default


def _install_service_target_plans() -> None:
    from malchan.app.services import InMemoryModelService

    original_train = InMemoryModelService.train

    @wraps(original_train)
    def train(self: Any, request: Any, *args: Any, **kwargs: Any) -> Any:
        set_target_plan(list(request.resolved_target_cols))
        return original_train(self, request, *args, **kwargs)

    InMemoryModelService.train = train

    for method_name in ("evaluate_model", "run_comparison", "tune_best_comparison"):
        original = getattr(InMemoryModelService, method_name, None)
        if not callable(original):
            continue

        @wraps(original)
        def target_operation(self: Any, model_id: str, *args: Any, __original=original, **kwargs: Any) -> Any:
            registered = self._get_registered(model_id)
            set_target_plan(list(registered.info.target_cols))
            return __original(self, model_id, *args, **kwargs)

        setattr(InMemoryModelService, method_name, target_operation)


def _install_pipeline_counters() -> None:
    from malchan.models import training
    from malchan.pipeline.single_output import SingleOutputMLModelPipeline

    original_fit = SingleOutputMLModelPipeline.fit

    @wraps(original_fit)
    def fit(self: Any, *args: Any, **kwargs: Any) -> Any:
        mark_target(_argument(args, kwargs, "target_col", 1))
        return original_fit(self, *args, **kwargs)

    SingleOutputMLModelPipeline.fit = fit

    original_fit_from_context = SingleOutputMLModelPipeline.fit_from_context

    @wraps(original_fit_from_context)
    def fit_from_context(self: Any, *args: Any, **kwargs: Any) -> Any:
        mark_target(_argument(args, kwargs, "target_col", 1))
        return original_fit_from_context(self, *args, **kwargs)

    SingleOutputMLModelPipeline.fit_from_context = fit_from_context

    original_cv_score = SingleOutputMLModelPipeline.cv_score

    @wraps(original_cv_score)
    def cv_score(self: Any, *args: Any, **kwargs: Any) -> Any:
        mark_target(getattr(self, "target_col", None))
        progress_id = current_progress_id()
        if progress_id is None:
            return original_cv_score(self, *args, **kwargs)

        method = str(_argument(args, kwargs, "method", 0, "kfold") or "kfold")
        n_splits = int(_argument(args, kwargs, "n_splits", 1, 5) or 5)
        X = _argument(args, kwargs, "X", 2)
        rows = X if X is not None else self._get_X()
        total = n_splits if method == "kfold" else len(rows)
        report_dimension("fold", 0, total, label="CV fold", progress_id=progress_id)
        state = {
            "progress_id": progress_id,
            "current": 0,
            "total": total,
        }
        token = _cv_state.set(state)
        try:
            return original_cv_score(self, *args, **kwargs)
        finally:
            _cv_state.reset(token)

    SingleOutputMLModelPipeline.cv_score = cv_score

    original_cv_fit = training.cv_fit

    @wraps(original_cv_fit)
    def cv_fit(*args: Any, **kwargs: Any) -> Any:
        result = original_cv_fit(*args, **kwargs)
        state = _cv_state.get()
        if state is not None:
            state["current"] = min(state["total"], state["current"] + 1)
            report_dimension(
                "fold",
                state["current"],
                state["total"],
                label="CV fold",
                progress_id=state["progress_id"],
            )
        return result

    training.cv_fit = cv_fit

    original_search = training.OptunaSearchCV

    def progress_optuna_search_cv(*args: Any, **kwargs: Any) -> Any:
        progress_id = current_progress_id()
        n_trials = kwargs.get("n_trials", 10)
        if progress_id is None or n_trials is None:
            return original_search(*args, **kwargs)

        total = max(0, int(n_trials))
        report_dimension("trial", 0, total, label="Optuna", progress_id=progress_id)
        callback_lock = Lock()
        completed = {"value": 0}
        callbacks = list(kwargs.get("callbacks") or [])

        def progress_callback(_study: Any, _trial: Any) -> None:
            with callback_lock:
                completed["value"] = min(total, completed["value"] + 1)
                current = completed["value"]
            report_dimension(
                "trial",
                current,
                total,
                label="Optuna",
                progress_id=progress_id,
            )

        callbacks.append(progress_callback)
        kwargs["callbacks"] = callbacks
        return original_search(*args, **kwargs)

    training.OptunaSearchCV = progress_optuna_search_cv


def _install_inverse_trial_counter() -> None:
    try:
        import optuna
        import malchan.inverse_analysis as public_inverse
        from malchan.inverse_analysis import models as inverse_models
    except ImportError:
        return

    original_inverse = inverse_models.inverse_analysis

    @wraps(original_inverse)
    def inverse_analysis(*args: Any, **kwargs: Any) -> Any:
        progress_id = current_progress_id()
        if progress_id is None:
            return original_inverse(*args, **kwargs)

        total = max(0, int(kwargs.get("trials", 250)))
        report_dimension("trial", 0, total, label="Optuna", progress_id=progress_id)
        state = {
            "progress_id": progress_id,
            "current": 0,
            "total": total,
            "lock": Lock(),
        }
        token = _inverse_trial_state.set(state)
        try:
            return original_inverse(*args, **kwargs)
        finally:
            _inverse_trial_state.reset(token)

    inverse_models.inverse_analysis = inverse_analysis
    if getattr(public_inverse, "_inverse_analysis", None) is original_inverse:
        public_inverse._inverse_analysis = inverse_analysis

    original_optimize = optuna.study.Study.optimize

    @wraps(original_optimize)
    def optimize(self: Any, func: Any, *args: Any, **kwargs: Any) -> Any:
        state = _inverse_trial_state.get()
        if state is None:
            return original_optimize(self, func, *args, **kwargs)

        callbacks = list(kwargs.get("callbacks") or [])

        def progress_callback(_study: Any, _trial: Any) -> None:
            with state["lock"]:
                state["current"] = min(state["total"], state["current"] + 1)
                current = state["current"]
            report_dimension(
                "trial",
                current,
                state["total"],
                label="Optuna",
                progress_id=state["progress_id"],
            )

        callbacks.append(progress_callback)
        kwargs["callbacks"] = callbacks
        return original_optimize(self, func, *args, **kwargs)

    optuna.study.Study.optimize = optimize


def install_progress_instrumentation() -> None:
    """Install app-only wrappers that expose real target/trial/fold counters."""

    global _instrumentation_installed
    if _instrumentation_installed:
        return
    _instrumentation_installed = True

    _install_service_target_plans()
    try:
        _install_pipeline_counters()
    except ImportError:
        # The API can be imported with the lightweight ``web`` extra.  Detailed
        # model counters become available when the optional model dependencies
        # required for training are installed.
        pass
    _install_inverse_trial_counter()


__all__ = [
    "PROGRESS_HEADER",
    "current_progress_id",
    "finish_progress",
    "get_progress_snapshot",
    "install_progress_instrumentation",
    "mark_target",
    "progress_scope",
    "report_dimension",
    "set_target_plan",
]
