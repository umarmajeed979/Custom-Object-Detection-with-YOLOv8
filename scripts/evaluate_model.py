"""
Entry point for evaluating the active task's trained model.

Run after scripts/train_model.py finishes — reports mAP, precision,
and recall against the test split via app/core/evaluator.py.
"""

from app.config import settings
from app.core.evaluator import EvaluationError, evaluate
from app.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    """Evaluate the active task's trained checkpoint and log summary metrics."""
    logger.info("Evaluating model for task '%s'", settings.task.value)
    try:
        evaluate()
    except EvaluationError:
        logger.exception("Evaluation failed")
        raise


if __name__ == "__main__":
    main()