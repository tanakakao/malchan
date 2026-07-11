"""Serve the optional React build from the FastAPI application."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def resolve_web_dist(configured_path: str | None = None) -> Path | None:
    """Resolve the first available React build directory.

    Args:
        configured_path: Explicit build directory supplied by application settings.

    Returns:
        Directory containing ``index.html`` or ``None`` when no build exists.
    """

    candidates: list[Path] = []
    if configured_path:
        candidates.append(Path(configured_path).expanduser())

    package_static = Path(__file__).resolve().parent / "static"
    repository_dist = Path(__file__).resolve().parents[4] / "frontend" / "dist"
    candidates.extend([package_static, repository_dist])

    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_dir() and (resolved / "index.html").is_file():
            return resolved
    return None


def mount_web_ui(app: Any, configured_path: str | None = None) -> Path | None:
    """Mount a built React single-page application at the root path.

    API and OpenAPI routes must be registered before calling this function so
    the root static mount remains the final fallback.

    Args:
        app: Configured FastAPI application.
        configured_path: Optional explicit frontend build directory.

    Returns:
        Mounted build directory, or ``None`` when no build is available.
    """

    web_dist = resolve_web_dist(configured_path)
    if web_dist is None:
        return None

    from fastapi.staticfiles import StaticFiles

    app.mount(
        "/",
        StaticFiles(directory=str(web_dist), html=True),
        name="malchan-web",
    )
    return web_dist


__all__ = ["mount_web_ui", "resolve_web_dist"]
