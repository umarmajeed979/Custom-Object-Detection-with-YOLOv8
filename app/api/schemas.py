"""
Pydantic request/response contracts. Keep these separate from the
model's internal data structures (in core/) so the API shape doesn't
change just because the model architecture changes.
"""

from pydantic import BaseModel


class BoundingBox(BaseModel):
    """Pixel coordinates in the original (unresized) image."""

    x1: float
    y1: float
    x2: float
    y2: float


class DetectionResult(BaseModel):
    """A single detected object."""

    label: str
    confidence: float
    box: BoundingBox


class PredictResponse(BaseModel):
    detections: list[DetectionResult]
    inference_time_ms: float
    model_version: str
    task: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    task: str