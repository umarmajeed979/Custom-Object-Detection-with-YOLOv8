"""
Dataset preprocessing for the active task.

Roboflow's export already provides a train/valid/test split, so this
module doesn't re-split anything — it normalizes that export into the
project's data/processed/ layout (train/validation/test) and
sanity-checks every image/label pair before training ever sees them.
Run via scripts/prepare_data.py.
"""

import shutil
from pathlib import Path

from app.config import settings
from app.data.loader import DatasetInfo, load_dataset
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Roboflow names the validation split "valid"; our layout calls it "validation".
SPLIT_NAME_MAP = {"train": "train", "valid": "validation", "test": "test"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


class PreprocessError(Exception):
    """Raised when a split can't be validated or copied cleanly."""


def _validate_pairs(images_dir: Path, labels_dir: Path) -> list[Path]:
    """Check every image has a matching label file, return the image file list."""
    image_files = sorted(p for p in images_dir.glob("*.*") if p.suffix.lower() in IMAGE_SUFFIXES)
    missing = [img.name for img in image_files if not (labels_dir / f"{img.stem}.txt").exists()]
    if missing:
        raise PreprocessError(
            f"{len(missing)} image(s) in {images_dir} have no matching label file, e.g. {missing[:3]}"
        )
    return image_files


def _copy_split(src_split_dir: Path, dest_split_dir: Path) -> int:
    """Validate then copy one split's images/ and labels/ into data/processed/<split>/."""
    src_images, src_labels = src_split_dir / "images", src_split_dir / "labels"
    image_files = _validate_pairs(src_images, src_labels)

    dest_images, dest_labels = dest_split_dir / "images", dest_split_dir / "labels"
    dest_images.mkdir(parents=True, exist_ok=True)
    dest_labels.mkdir(parents=True, exist_ok=True)

    for image_path in image_files:
        shutil.copy2(image_path, dest_images / image_path.name)
        label_path = src_labels / f"{image_path.stem}.txt"
        shutil.copy2(label_path, dest_labels / label_path.name)

    return len(image_files)


def prepare_dataset(dataset: DatasetInfo | None = None) -> dict[str, int]:
    """Copy and validate the raw dataset into data/processed/, split by split."""
    dataset = dataset or load_dataset()
    counts: dict[str, int] = {}

    for raw_split, processed_split in SPLIT_NAME_MAP.items():
        src_dir = dataset.split_dirs[raw_split]
        dest_dir = settings.processed_data_dir / processed_split
        counts[processed_split] = _copy_split(src_dir, dest_dir)
        logger.info(
            "Prepared '%s' split — %d images copied to %s",
            processed_split,
            counts[processed_split],
            dest_dir,
        )

    return counts