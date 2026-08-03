"""
Entry point for exporting the active task's trained checkpoint to ONNX.

Converts data/models/<task>/weights/best.pt (produced by
scripts/train_model.py) into the ONNX file app/core/model.py loads for
serving, at settings.model_weights_path. Run after training (and
ideally after scripts/evaluate_model.py) finishes.
"""

import shutil
from pathlib import Path

from ultralytics import YOLO

from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class ExportError(Exception):
    """Raised when there's no trained checkpoint to export."""


def export_to_onnx() -> Path:
    """Export the active task's best.pt checkpoint to ONNX at settings.model_weights_path."""
    checkpoint = settings.models_dir / settings.task.value / "weights" / "best.pt"
    if not checkpoint.exists():
        raise ExportError(
            f"No trained checkpoint found at {checkpoint} — run scripts/train_model.py first."
        )

    logger.info("Exporting %s to ONNX (imgsz=%d)", checkpoint, settings.image_size)
    model = YOLO(str(checkpoint))
    exported_path = Path(model.export(format="onnx", imgsz=settings.image_size))

    destination = settings.model_weights_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(exported_path), str(destination))

    logger.info("Exported ONNX model to %s", destination)
    return destination


def main() -> None:
    """Export the active task's trained checkpoint, logging the result path."""
    try:
        export_to_onnx()
    except ExportError:
        logger.exception("Export failed")
        raise


if __name__ == "__main__":
    main()