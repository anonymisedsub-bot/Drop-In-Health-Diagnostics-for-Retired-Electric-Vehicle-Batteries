"""Small shared utilities for baseline group 3."""

from __future__ import annotations

import json
from pathlib import Path
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_json(value, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)


def denorm_y(normalized: np.ndarray, norm: dict) -> np.ndarray:
    return np.asarray(normalized) * (norm["ymax"] - norm["ymin"]) + norm["ymin"]
