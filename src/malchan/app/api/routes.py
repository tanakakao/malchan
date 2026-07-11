"""FastAPI routes for model lifecycle and inference."""

from typing import Any

from fastapi import APIRouter, HTTPException, Response, status

from malchan import __version__
from malchan.app.schemas import (
    HealthResponse,
    ModelInfo,
    ModelListResponse,
    PredictRequest,
    PredictionResponse,
    TrainModelRequest,
)
from malchan.app.services import ModelNotFoundError


def create_api_router(service: Any, app_name: str) -> APIRouter:
    """Create API routes bound to a model service instance.

    Args:
        service: Model training and inference service.
        app_name: Human-readable application name used by the health endpoint.

    Returns:
        APIRouter: Configured API router.
    """

    router = APIRouter()

    @router.get("/health", response_model=HealthResponse, tags=["system"])
    def health_check() -> HealthResponse:
        """Return lightweight process health metadata."""

        return HealthResponse(status="ok", service=app_name, version=__version__)

    @router.post(
        "/models",
        response_model=ModelInfo,
        status_code=status.HTTP_201_CREATED,
        tags=["models"],
    )
    def train_model(request: TrainModelRequest) -> ModelInfo:
        """Train and register a single-output model."""

        try:
            return service.train(request)
        except (ImportError, RuntimeError, TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(exc),
            ) from exc

    @router.get("/models", response_model=ModelListResponse, tags=["models"])
    def list_models() -> ModelListResponse:
        """List models registered in this process."""

        return ModelListResponse(models=service.list_models())

    @router.get("/models/{model_id}", response_model=ModelInfo, tags=["models"])
    def get_model(model_id: str) -> ModelInfo:
        """Return metadata for one registered model."""

        try:
            return service.get_model(model_id)
        except ModelNotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Model not found.",
            ) from exc

    @router.post(
        "/models/{model_id}/predict",
        response_model=PredictionResponse,
        tags=["models"],
    )
    def predict(model_id: str, request: PredictRequest) -> PredictionResponse:
        """Generate predictions from a registered model."""

        try:
            predictions = service.predict(model_id, request)
        except ModelNotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Model not found.",
            ) from exc
        except (TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(exc),
            ) from exc
        return PredictionResponse(model_id=model_id, predictions=predictions)

    @router.delete(
        "/models/{model_id}",
        status_code=status.HTTP_204_NO_CONTENT,
        tags=["models"],
    )
    def delete_model(model_id: str) -> Response:
        """Delete one registered model."""

        try:
            service.delete(model_id)
        except ModelNotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Model not found.",
            ) from exc
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    return router
