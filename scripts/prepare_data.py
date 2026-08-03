"""
Entry point for preparing the active task's dataset for training.

Validates the raw Roboflow export (app/data/loader.py) and copies it
into data/processed/ (app/data/preprocessor.py). Run this once after
downloading and unzipping a dataset into data/raw/<task>/, and before
scripts/train_model.py.
"""

from app.config import settings
from app.data.loader import DatasetError
from app.data.preprocessor import PreprocessError, prepare_dataset
from app.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    """Validate and copy the active task's raw dataset into data/processed/."""
    logger.info("Preparing dataset for task '%s'", settings.task.value)
    try:
        counts = prepare_dataset()
    except (DatasetError, PreprocessError):
        logger.exception("Dataset preparation failed")
        raise

    logger.info("Dataset ready for task '%s': %s", settings.task.value, counts)


if __name__ == "__main__":
    main()