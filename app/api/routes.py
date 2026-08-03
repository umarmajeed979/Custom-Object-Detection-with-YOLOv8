"""
API route definitions for the YOLOv8 custom object detection service.

The model is loaded lazily, on first request, via get_predictor() below
— not at import time. This matters because app/__init__.py imports this
module, so anything that imports `app` (scripts, tests, data prep)
would otherwise try to load ONNX weights that may not exist yet.
"""

import time
from functools import lru_cache

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.api.schemas import BoundingBox, DetectionResult, HealthResponse, PredictResponse
from app.config import settings
from app.core.model import ModelLoadError
from app.core.predictor import Predictor
from app.utils.image_utils import decode_image
from app.utils.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png"}


@lru_cache
def get_predictor() -> Predictor:
    """Build (and cache) the Predictor for the active task, on first use."""
    return Predictor()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Report service status and whether the active task's model loaded OK."""
    try:
        get_predictor()
        model_loaded = True
    except ModelLoadError:
        logger.exception("Health check: model failed to load")
        model_loaded = False

    return HealthResponse(
        status="ok" if model_loaded else "degraded",
        model_loaded=model_loaded,
        task=settings.task.value,
    )


@router.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)) -> PredictResponse:
    """Run detection on an uploaded image for the active task."""
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=400, detail="File must be a JPEG or PNG image")

    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > settings.max_upload_size_mb:
        raise HTTPException(status_code=400, detail=f"File exceeds {settings.max_upload_size_mb}MB limit")

    try:
        image = decode_image(contents)
    except Exception:
        logger.exception("Failed to decode uploaded image")
        raise HTTPException(status_code=400, detail="Could not decode image")

    try:
        predictor = get_predictor()
    except ModelLoadError as exc:
        logger.exception("Model not available for prediction")
        raise HTTPException(status_code=503, detail=str(exc))

    start = time.perf_counter()
    detections = predictor.predict(image)
    inference_time_ms = (time.perf_counter() - start) * 1000

    return PredictResponse(
        detections=[
            DetectionResult(
                label=d.class_name,
                confidence=d.confidence,
                box=BoundingBox(x1=d.box[0], y1=d.box[1], x2=d.box[2], y2=d.box[3]),
            )
            for d in detections
        ],
        inference_time_ms=inference_time_ms,
        model_version=settings.api_version,
        task=settings.task.value,
    )