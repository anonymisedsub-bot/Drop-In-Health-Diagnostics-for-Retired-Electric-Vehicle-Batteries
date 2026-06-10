"""Shared notebook helpers for reproducible figure generation."""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_MARKER = "notebook_setup.py"
DATA_ENV_VAR = "BATTERY_DATA_DIR"


def find_project_root(start: Path | None = None) -> Path:
    """Return the repository root that contains this helper module."""
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / PROJECT_MARKER).exists():
            return candidate
    return current


PROJECT_ROOT = find_project_root()
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"


def get_dataset_path(filename: str) -> Path:
    """Resolve a dataset file from BATTERY_DATA_DIR or the local data folder."""
    data_dir = Path(os.environ.get(DATA_ENV_VAR, DEFAULT_DATA_DIR)).expanduser()
    dataset_path = data_dir / filename
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {dataset_path}. "
            f"Place '{filename}' in '{DEFAULT_DATA_DIR}' or set {DATA_ENV_VAR}."
        )
    return dataset_path
