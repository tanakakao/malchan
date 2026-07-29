"""FastAPI routes for model lifecycle, comparison, and inverse analysis."""

from typing import Any

from fastapi import APIRouter, HTTPException, Request, Response, status

from malchan import __version__
from malchan.app.schemas import (
    CompareModelsRequest,
    HealthResponse,
    InverseAnalysisRequest,
    InverseAnalysisResponse,
    ModelBundleImportResponse,
    ModelComparisonResponse,
    ModelEvaluationRequest,
    ModelEvaluationResponse,
    ModelInfo,
    ModelListResponse,
    ModelParameterSchemaResponse,
    ModelVisualizationResponse,
    PredictRequest,
    PredictionResponse,
    TrainModelRequest,
    TuneBestModelRequest,
)
from malchan.app.services import (
    ComparisonNotFoundError,
    InvalidModelBundleError,
    ModelBundleTooLargeError,
    ModelBundleUnavailableError,
    ModelNotFoundError,
)

from .visualization_routes import create_visualization_router
from .xai_routes import create_xai_router

_MODEL_BUNDLE_MEDIA_TYPE = "application/vnd.malchan.model"


def create_api_router(
    service: Any,
    app_name: str,
    app_version: str = __version__,
) -> APIRouter:
    """Create API routes bound to a model service instance.

    Args:
        service: Model service used by route handlers.
        app_name: Service name returned by the health endpoint.
        app_version: Service version returned by the health endpoint.

    Returns:
        Router containing the malchan API endpoints.
    """

    router = APIRouter()

    @router.get("/health", response_model=HealthResponse, tags=["system"])
    def health_check() -> HealthResponse:
        """Return lightweight process health metadata."""

        return HealthResponse(status="ok", service=app_name, version=app_version)

    @router.get(
        "/model-parameters",
        response_model=ModelParameterSchemaResponse,
        tags=["models"],
    )
    def get_model_parameters(
        task: str,
        model_name: str,
    ) -> ModelParameterSchemaResponse:
        """Return Web controls derived from one model's tuning search space."""

        try:
            return service.get_model_parameter_schema(task, model_name)
        except (ImportError, RuntimeError, TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(exc),
            ) from exc

    @router.post(
        "/models",
        response_model=ModelInfo,
        status_code=status.HTTP_201_CREATED,
        tags=["models"],
    )
    def train_model(request: TrainModelRequest) -> ModelInfo:
        """Train and register a single-output or multi-output model."""

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

    @router.post(
        "/model-bundles/import",
        response_model=ModelBundleImportResponse,
        status_code=status.HTTP_201_CREATED,
        tags=["model-bundles"],
    )
    async def import_model_bundle(request: Request) -> ModelBundleImportResponse:
        """Verify a signed model file and restore it only to process memory."""

        configured_limit = int(getattr(service, "_model_bundle_max_bytes", 256 * 1024 * 1024))
        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                if int(content_length) > configured_limit:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail="モデルファイルが設定された上限を超えています。",
                    )
            except ValueError:
                pass
        bundle = await request.body()
        try:
            return service.import_model_bundle(bundle)
        except ModelBundleUnavailableError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(exc),
            ) from exc
        except ModelBundleTooLargeError as exc:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=str(exc),
            ) from exc
        except InvalidModelBundleError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(exc),
            ) from exc
        except RuntimeError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(exc),
            ) from exc

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

    @router.get(
        "/models/{model_id}/export",
        response_class=Response,
        tags=["model-bundles"],
    )
    def export_model_bundle(model_id: str) -> Response:
        """Download a signed model file without writing it to server storage."""

        try:
            bundle, filename = service.export_model_bundle(model_id)
        except ModelNotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Model not found.",
            ) from exc
        except ModelBundleUnavailableError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(exc),
            ) from exc
        except ModelBundleTooLargeError as exc:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=str(exc),
            ) from exc
        except InvalidModelBundleError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(exc),
            ) from exc
        return Response(
            content=bundle,
            media_type=_MODEL_BUNDLE_MEDIA_TYPE,
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "Cache-Control": "no-store",
                "X-Content-Type-Options": "nosniff",
            },
        )

    @router.get(
        "/models/{model_id}/visualization",
        response_model=ModelVisualizationResponse,
        tags=["models"],
    )
    def get_model_visualization(model_id: str) -> ModelVisualizationResponse:
        """Return native estimator structures and cached validation scores."""

        try:
            return service.get_model_visualization(model_id)
        except ModelNotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Model not found.",
            ) from exc
        except (RuntimeError, TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(exc),
            ) from exc

    @router.post(
        "/models/{model_id}/predict",
        response_model=PredictionResponse,
        tags=["models"],
    )
    def predict(model_id: str, request: PredictRequest) -> PredictionResponse:
        """Generate single-output or multi-output predictions."""

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

    @router.post(
        "/models/{model_id}/evaluate",
        response_model=ModelEvaluationResponse,
        tags=["evaluation"],
    )
    def evaluate_model(
        model_id: str,
        request: ModelEvaluationRequest,
    ) -> ModelEvaluationResponse:
        """Cross-validate the registered model without selecting candidates."""

        try:
            return service.evaluate_model(model_id, request)
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

    @router.post(
        "/models/{model_id}/compare",
        response_model=ModelComparisonResponse,
        tags=["comparison"],
    )
    def compare_models(
        model_id: str,
        request: CompareModelsRequest,
    ) -> ModelComparisonResponse:
        """Compare candidate model families and optionally tune the best."""

        try:
            return service.run_comparison(model_id, request)
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
        "/models/{model_id}/comparison",
        response_model=ModelComparisonResponse,
        tags=["comparison"],
    )
    def get_model_comparison(model_id: str) -> ModelComparisonResponse:
        """Return the latest comparison and tuning state for a model."""

        try:
            return service.get_comparison(model_id)
        except ModelNotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Model not found.",
            ) from exc
        except ComparisonNotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Run model comparison before requesting its result.",
            ) from exc

    @router.post(
        "/models/{model_id}/comparison/tune-best",
        response_model=ModelComparisonResponse,
        tags=["comparison"],
    )
    def tune_best_model(
        model_id: str,
        request: TuneBestModelRequest,
    ) -> ModelComparisonResponse:
        """Tune selected best candidates from the latest comparison."""

        try:
            return service.tune_best_comparison(model_id, request)
        except ModelNotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Model not found.",
            ) from exc
        except ComparisonNotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Run model comparison before tuning its best model.",
            ) from exc
        except (ImportError, RuntimeError, TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(exc),
            ) from exc

    @router.post(
        "/models/{model_id}/inverse-analysis",
        response_model=InverseAnalysisResponse,
        response_model_exclude_none=True,
        tags=["inverse-analysis"],
    )
    def run_inverse_analysis(
        model_id: str,
        request: InverseAnalysisRequest,
    ) -> InverseAnalysisResponse:
        """Search feature candidates satisfying requested objectives."""

        try:
            return service.run_inverse_analysis(model_id, request)
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

    @router.delete(
        "/models/{model_id}",
        status_code=status.HTTP_204_NO_CONTENT,
        tags=["models"],
    )
    def delete_model(model_id: str) -> Response:
        """Delete one registered model and its comparison state."""

        try:
            service.delete(model_id)
        except ModelNotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Model not found.",
            ) from exc
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    router.include_router(create_xai_router(service))
    router.include_router(create_visualization_router(service))
    return router
