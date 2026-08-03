"""
Model loading for the YOLOv8 custom object detection API.

Loads the exported ONNX weights for the currently active task
(``app.config.settings.task``) into an ``onnxruntime.InferenceSession``.
Training and evaluation work with raw Ultralytics ``.pt`` checkpoints
directly inside ``scripts/`` — this module only loads the deployment
artifact that ``core/predictor.py`` runs inference against.
"""

from functools import lru_cache
from pathlib import Path

import onnxruntime as ort

from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class ModelLoadError(Exception):
    """Raised when the configured ONNX weights can't be found or loaded."""


def _build_session(weights_path: Path) -> ort.InferenceSession:
    """Create an ONNX Runtime session, preferring GPU if one is available."""
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    try:
        return ort.InferenceSession(str(weights_path), providers=providers)
    except Exception as exc:
        logger.exception("Failed to load ONNX model from %s", weights_path)
        raise ModelLoadError(f"Could not load model weights at {weights_path}") from exc


@lru_cache
def get_model() -> ort.InferenceSession:
    """Return a cached ONNX Runtime session for the active task's weights."""
    weights_path = settings.model_weights_path
    if not weights_path.exists():
        raise ModelLoadError(
            f"No weights found at {weights_path} — export a model for task "
            f"'{settings.task.value}' first (scripts/export_model.py)."
        )
    logger.info("Loading model for task '%s' from %s", settings.task.value, weights_path)
    session = _build_session(weights_path)
    logger.info(
        "Model loaded — input: %s, outputs: %s",
        session.get_inputs()[0].name,
        [o.name for o in session.get_outputs()],
    )
    return session


def input_name() -> str:
    """Name of the model's single input tensor."""
    return get_model().get_inputs()[0].name


def output_names() -> list[str]:
    """Names of the model's output tensor(s)."""
    return [o.name for o in get_model().get_outputs()]