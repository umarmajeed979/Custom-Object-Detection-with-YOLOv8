"""
Training entry point for the active task (app.config.settings.task).

Run this on Colab (GPU), after scripts/prepare_data.py has populated
data/processed/. Fine-tunes a YOLOv8s checkpoint on the active task's
class list from app/config.py, and saves the run under
data/models/<task>/ — scripts/export_model.py picks up the resulting
best.pt from there and converts it to ONNX for serving.
"""

from pathlib import Path

import yaml
from ultralytics import YOLO

from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

BASE_MODEL = "yolov8s.pt"
EPOCHS = 100
BATCH_SIZE = 16  # reduce if you hit GPU memory limits on Colab's free tier


def _write_training_yaml() -> Path:
    """Write a data.yaml pointing at data/processed/ for the active task."""
    processed = settings.processed_data_dir
    config = {
        "path": str(processed),
        "train": "train/images",
        "val": "validation/images",
        "test": "test/images",
        "names": {i: name for i, name in enumerate(settings.class_names)},
    }
    yaml_path = processed / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.safe_dump(config, f)
    return yaml_path


def main() -> None:
    """Fine-tune YOLOv8s on the active task and save the run under data/models/<task>/."""
    data_yaml = _write_training_yaml()
    logger.info(
        "Starting training — task '%s', base model '%s', %d epochs",
        settings.task.value,
        BASE_MODEL,
        EPOCHS,
    )

    model = YOLO(BASE_MODEL)
    results = model.train(
        data=str(data_yaml),
        epochs=EPOCHS,
        imgsz=settings.image_size,
        batch=BATCH_SIZE,
        project=str(settings.models_dir),
        name=settings.task.value,
        exist_ok=True,
    )

    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    logger.info("Training complete — best weights at %s", best_weights)


if __name__ == "__main__":
    main()