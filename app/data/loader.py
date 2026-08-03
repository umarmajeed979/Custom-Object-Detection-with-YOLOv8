"""
Dataset loading for the active task (app.config.settings.task).

Locates the raw Roboflow export under data/raw/<task>/, validates it
against the class list in app/config.py, and exposes a small
DatasetInfo used by scripts/prepare_data.py, scripts/train_model.py,
and core/evaluator.py. Roboflow exports already ship pre-split
(train/valid/test, each with images/ and labels/), so "loading" here
means finding and validating that structure — not creating it.
"""

from dataclasses import dataclass
from pathlib import Path

import yaml

from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

REQUIRED_SPLITS = ("train", "valid", "test")


class DatasetError(Exception):
    """Raised when the raw dataset is missing, incomplete, or doesn't match config.py."""


@dataclass
class DatasetInfo:
    """Resolved paths and metadata for the active task's raw dataset."""

    root: Path
    data_yaml: Path
    class_names: list[str]
    split_dirs: dict[str, Path]


def _read_data_yaml(path: Path) -> dict:
    """Parse a Roboflow-exported data.yaml file."""
    with open(path) as f:
        return yaml.safe_load(f)


def load_dataset() -> DatasetInfo:
    """Validate and load the raw dataset for the currently active task."""
    root = settings.task_data_dir
    data_yaml_path = root / "data.yaml"

    if not root.exists():
        raise DatasetError(
            f"No dataset found at {root} — download the Roboflow export for "
            f"task '{settings.task.value}' and unzip it there first."
        )
    if not data_yaml_path.exists():
        raise DatasetError(f"data.yaml not found at {data_yaml_path}")

    data_yaml = _read_data_yaml(data_yaml_path)
    yaml_classes = list(data_yaml.get("names", []))

    if yaml_classes != settings.class_names:
        logger.warning(
            "data.yaml classes %s don't match TASK_CLASSES %s in app/config.py for task "
            "'%s' — update config.py to match the actual dataset before training.",
            yaml_classes,
            settings.class_names,
            settings.task.value,
        )

    split_dirs: dict[str, Path] = {}
    for split in REQUIRED_SPLITS:
        split_dir = root / split
        if not split_dir.exists():
            raise DatasetError(f"Missing '{split}' split at {split_dir}")
        if not (split_dir / "images").exists() or not (split_dir / "labels").exists():
            raise DatasetError(f"'{split}' split at {split_dir} must contain images/ and labels/")
        split_dirs[split] = split_dir

    logger.info(
        "Loaded dataset for task '%s' — %d classes, splits: %s",
        settings.task.value,
        len(yaml_classes),
        list(split_dirs),
    )
    return DatasetInfo(root=root, data_yaml=data_yaml_path, class_names=yaml_classes, split_dirs=split_dirs)


def count_images(split_dirs: dict[str, Path]) -> dict[str, int]:
    """Count images per split — useful for sanity-checking before training."""
    return {split: sum(1 for _ in (path / "images").glob("*.*")) for split, path in split_dirs.items()}