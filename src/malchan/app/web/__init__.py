"""React frontend integration for the malchan FastAPI application."""

from .static import mount_web_ui, resolve_web_dist

__all__ = ["mount_web_ui", "resolve_web_dist"]
