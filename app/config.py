"""
Central configuration for the Custom Object Detection with YOLOv8 project.

Every path, model setting, and per-task class list lives here — nowhere
else in the codebase should hardcode a path, threshold, or class name
(see docs/CODING_STANDARDS.md, rule 2). Switch between the three
supported tasks (helmet, ppe, waste) by setting TASK in `.env`; the
class list, confidence threshold, dataset folder, and weights path all
follow automatically from that one value.
"""

from enum import Enum
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Task(str, Enum):
    """Supported detection tasks for this project."""

    HELMET = "helmet"
    PPE = "ppe"
    WASTE = "waste"


# Per-task class lists, in the exact order used when training each
# dataset's data.yaml. Add a new task here (plus a new folder under
# data/raw/<task>/) to extend the project.
TASK_CLASSES: dict[Task, list[str]] = {
    Task.HELMET: ["head", "helmet"],
    Task.PPE: ["Hardhat", "Mask", "NO-Hardhat", "NO-Mask", "NO-Safety Vest", "Person", "Safety Vest"],
    Task.WASTE: ["cardboard", "glass", "metal", "paper", "plastic", "trash"],
}

# Default inference confidence threshold per task — tune after
# evaluating each model, don't hardcode this in predictor.py.
TASK_CONF_THRESHOLD: dict[Task, float] = {
    Task.HELMET: 0.5,
    Task.PPE: 0.45,
    Task.WASTE: 0.4,
}

BASE_DIR = Path(__file__).resolve().parent.parent  # project root


class Settings(BaseSettings):
    """Application settings, loaded from environment variables / .env."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- Task selection ---
    task: Task = Task.HELMET

    # --- Paths ---
    base_dir: Path = BASE_DIR
    data_dir: Path = BASE_DIR / "data"
    raw_data_dir: Path = BASE_DIR / "data" / "raw"
    processed_data_dir: Path = BASE_DIR / "data" / "processed"
    models_dir: Path = BASE_DIR / "data" / "models"
    log_dir: Path = BASE_DIR / "logs"

    # --- Model ---
    model_weights_filename: str = "best.onnx"
    image_size: int = 640

    # --- API ---
    api_title: str = "YOLOv8 Custom Object Detection API"
    api_version: str = "0.1.0"
    max_upload_size_mb: int = 10

    # --- Logging (read directly by the shared app/utils/logger.py) ---
    LOG_LEVEL: str = "INFO"

    @property
    def class_names(self) -> list[str]:
        """Class list for the currently active task."""
        return TASK_CLASSES[self.task]

    @property
    def confidence_threshold(self) -> float:
        """Default inference confidence threshold for the active task."""
        return TASK_CONF_THRESHOLD[self.task]

    @property
    def task_data_dir(self) -> Path:
        """Raw dataset directory for the active task (data/raw/<task>/)."""
        return self.raw_data_dir / self.task.value

    @property
    def model_weights_path(self) -> Path:
        """Full path to the exported weights for the active task."""
        return self.models_dir / self.task.value / self.model_weights_filename


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance (one Settings object per process)."""
    return Settings()


settings = get_settings()