"""FastAPI routes for cached model explainability data."""

from typing import Any

from fastapi import APIRouter, HTTPException, Query, status

from malchan.app.schemas import (
    RecomputeXaiRequest,
    XaiImportanceResponse,
    XaiPdpResponse,
    XaiShapResponse,
    XaiSummaryResponse,
)
from malchan.app.services import ModelNotFoundError, XaiNotReadyError


def create_xai_router(service: Any) -> APIRouter:
    """Create XAI routes bound to one model service instance."""

    router = APIRouter(tags=["xai"])

    @router.get(
        "/models/{model_id}/xai",
        response_model=XaiSummaryResponse,
    )
    def get_xai_summary(model_id: str) -> XaiSummaryResponse:
        """Return cached XAI availability without recalculating explanations."""

        try:
            return service.get_xai_summary(model_id)
        except ModelNotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Model not found.",
            ) from exc

    @router.post(
        "/models/{model_id}/xai/recompute",
        response_model=XaiSummaryResponse,
    )
    def recompute_xai(
        model_id: str,
        request: RecomputeXaiRequest,
    ) -> XaiSummaryResponse:
        """Explicitly rebuild SHAP, importance, and PDP caches."""

        try:
            return service.recompute_xai(model_id, request)
        except ModelNotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Model not found.",
            ) from exc
        except (ImportError, RuntimeError, TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(exc),
            ) from exc

    @router.get(
        "/models/{model_id}/xai/{target}/importance",
        response_model=XaiImportanceResponse,
    )
    def get_xai_importance(
        model_id: str,
        target: str,
        method: str = Query(default="shap"),
        combined: bool = Query(default=True),
        top_n: int = Query(default=20, ge=1, le=1000),
    ) -> XaiImportanceResponse:
        """Return cached model, PFI, or SHAP importance values."""

        try:
            return service.get_xai_importance(
                model_id,
                target,
                method=method,
                combined=combined,
                top_n=top_n,
            )
        except ModelNotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Model not found.",
            ) from exc
        except XaiNotReadyError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=str(exc),
            ) from exc
        except (TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(exc),
            ) from exc

    @router.get(
        "/models/{model_id}/xai/{target}/shap",
        response_model=XaiShapResponse,
    )
    def get_xai_shap(
        model_id: str,
        target: str,
        feature: str = Query(min_length=1),
    ) -> XaiShapResponse:
        """Return cached SHAP scatter records for one raw feature."""

        try:
            return service.get_xai_shap(model_id, target, feature)
        except ModelNotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Model not found.",
            ) from exc
        except XaiNotReadyError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=str(exc),
            ) from exc
        except (TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(exc),
            ) from exc

    @router.get(
        "/models/{model_id}/xai/{target}/pdp",
        response_model=XaiPdpResponse,
    )
    def get_xai_pdp(
        model_id: str,
        target: str,
        feature: str = Query(min_length=1),
        include_ice: bool = Query(default=False),
        max_ice: int = Query(default=30, ge=1, le=300),
    ) -> XaiPdpResponse:
        """Return cached PDP means and optional ICE curves."""

        try:
            return service.get_xai_pdp(
                model_id,
                target,
                feature,
                include_ice=include_ice,
                max_ice=max_ice,
            )
        except ModelNotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Model not found.",
            ) from exc
        except XaiNotReadyError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=str(exc),
            ) from exc
        except (TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(exc),
            ) from exc

    return router


__all__ = ["create_xai_router"]
