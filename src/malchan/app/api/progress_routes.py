"""FastAPI polling endpoint for detailed process-local progress."""

from fastapi import APIRouter, HTTPException, status

from malchan.app.progress import get_progress_snapshot


def create_progress_router() -> APIRouter:
    """Create the lightweight progress polling router."""

    router = APIRouter()

    @router.get("/progress/{progress_id}", tags=["system"])
    def get_progress(progress_id: str) -> dict:
        snapshot = get_progress_snapshot(progress_id)
        if snapshot is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Progress record not found.",
            )
        return snapshot

    return router


__all__ = ["create_progress_router"]
