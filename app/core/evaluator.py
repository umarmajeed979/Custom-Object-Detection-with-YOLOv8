"""
Offline evaluation metrics for the active task's trained model.

Runs Ultralytics' built-in validation against the test split and
reports mAP, precision, and recall. This evaluates the trained
Ultralytics checkpoint (best.pt) — not the exported ONNX model, since
mAP evaluation is a training-time concern that happens before
scripts/export_model.py ever runs. Called by scripts/evaluate_model.py.
"""

from dataclasses import dataclass
from pathlib import Path

from ultralytics import YOLO

from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class EvaluationError(Exception):
    """Raised when no trained checkpoint or data.yaml is available to evaluate."""


@dataclass
class EvalMetrics:
    """Summary metrics from an evaluation run."""

    map50: float
    map50_95: float
    precision: float
    recall: float


def _best_checkpoint_path() -> Path:
    """Path to the best.pt checkpoint produced by scripts/train_model.py."""
    return settings.models_dir / settings.task.value / "weights" / "best.pt"


def evaluate(data_yaml: Path | None = None) -> EvalMetrics:
    """Run validation against the test split and return summary metrics."""
    checkpoint = _best_checkpoint_path()
    if not checkpoint.exists():
        raise EvaluationError(
            f"No trained checkpoint found at {checkpoint} — run scripts/train_model.py first "
            "(and make sure it has actually finished)."
        )

    data_yaml = data_yaml or (settings.processed_data_dir / "data.yaml")
    if not data_yaml.exists():
        raise EvaluationError(f"No data.yaml found at {data_yaml} — run scripts/prepare_data.py first.")

    logger.info("Evaluating task '%s' checkpoint at %s", settings.task.value, checkpoint)
    model = YOLO(str(checkpoint))
    results = model.val(data=str(data_yaml), split="test", imgsz=settings.image_size)

    metrics = EvalMetrics(
        map50=float(results.box.map50),
        map50_95=float(results.box.map),
        precision=float(results.box.mp),
        recall=float(results.box.mr),
    )
    logger.info(
        "Evaluation complete — mAP50: %.3f, mAP50-95: %.3f, precision: %.3f, recall: %.3f",
        metrics.map50,
        metrics.map50_95,
        metrics.precision,
        metrics.recall,
    )
    return metrics