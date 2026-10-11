"""Shared CETL loading, seeded pool split, and train-only normalization."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import loadmat

import config as cfg


def load_q_mat(mat_path: str) -> np.ndarray:
    path = Path(mat_path)
    if not path.is_file():
        raise FileNotFoundError(f"LFP .mat file not found: {path}. Set DATA_MAT_PATH in config.py to a valid processed LFP MAT.")
    matrix = np.asarray(loadmat(path)["processed_Qd_mat"])
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError("processed_Qd_mat must be a finite (time, cells) matrix")
    return matrix


def active_compute_split_sizes(n_cells: int) -> dict[str, int]:
    initial = int(cfg.num_tr_initial)
    step = int(cfg.num_tr_per_iter)
    if step <= 0 or initial <= 0:
        raise ValueError("Initial label count and acquisition step must be positive")
    pool = int(round(cfg.al_pool_ratio * n_cells))
    pool = (pool // step) * step
    validation = int(round(cfg.al_val_ratio * n_cells))
    if initial + validation >= n_cells:
        raise ValueError("Dataset is too small for nonempty train, validation, and test sets")
    if initial + pool + validation >= n_cells:
        pool = max(0, n_cells - initial - validation - 1)
        pool = (pool // step) * step
    test = n_cells - initial - pool - validation
    return {"train_initial": initial, "pool": pool, "val": validation, "test": test}


def active_split_indices(n_cells: int, seed: int = 0) -> dict[str, np.ndarray]:
    sizes = active_compute_split_sizes(n_cells)
    shuffled = np.random.RandomState(seed).permutation(n_cells)
    first = sizes["train_initial"]
    second = first + sizes["pool"]
    third = second + sizes["val"]
    return {
        "train": np.sort(shuffled[:first]),
        "pool": np.sort(shuffled[first:second]),
        "val": np.sort(shuffled[second:third]),
        "test": np.sort(shuffled[third:]),
    }


def build_XY_from_global(
    matrix: np.ndarray, indices: np.ndarray, segment_length: int
) -> tuple[np.ndarray, np.ndarray]:
    if segment_length < 1 or segment_length > matrix.shape[0]:
        raise ValueError("Tail length exceeds the available voltage grid")
    x = matrix[:segment_length, indices].T[:, None, :]
    y = matrix[-1, indices].reshape(-1, 1)
    return x, y


def fit_normalizer(train_x: np.ndarray, train_y: np.ndarray, mode: str) -> dict:
    if not len(train_x):
        raise ValueError("The labelled training set is empty")
    if mode == "per_feature":
        xmin = train_x.min(axis=0, keepdims=True)
        xmax = train_x.max(axis=0, keepdims=True)
    elif mode == "per_channel":
        xmin = train_x.min(keepdims=True)
        xmax = train_x.max(keepdims=True)
    else:
        raise ValueError("normalization_mode must be per_feature or per_channel")
    return {
        "xmin": xmin,
        "xmax": xmax,
        "ymin": train_y.min(keepdims=True),
        "ymax": train_y.max(keepdims=True),
        "mode": mode,
    }


def apply_normalizer(x: np.ndarray, y: np.ndarray, norm: dict) -> tuple[np.ndarray, np.ndarray]:
    x_normalized = (x - norm["xmin"]) / np.maximum(norm["xmax"] - norm["xmin"], 1e-8)
    y_normalized = (y - norm["ymin"]) / np.maximum(norm["ymax"] - norm["ymin"], 1e-8)
    return np.clip(x_normalized, 0.0, 1.0), np.clip(y_normalized, 0.0, 1.0)
