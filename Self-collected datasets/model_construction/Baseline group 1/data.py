"""Tail-only data adapter for the shared ReBATT experiment protocol."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


MODEL_ROOT = Path(__file__).resolve().parents[1]
if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))

from baseline_support.rebatt_protocol import build_voltage_windows, role_view


def build_tail_xy(
    data,
    indices,
    role: str,
    vseg_low: float,
    voltage_removal_window: float = 0.20,
):
    """Return one-channel tail inputs and role-safe capacity targets."""

    indices = np.asarray(indices, dtype=np.int64)
    view = role_view(
        data,
        role,
        indices,
        vseg_low=vseg_low,
        voltage_removal_window=voltage_removal_window,
    )
    windows = build_voltage_windows(data.voltage_grid, vseg_low)
    tail = view.q[windows.tail_indices, :]
    if not np.isfinite(tail).all():
        bad = np.argwhere(~np.isfinite(tail))[0]
        raise ValueError(
            f"Tail contains a non-finite value for {role} local sample "
            f"{int(bad[1])}"
        )
    X = tail.T[:, None, :].astype(np.float32, copy=False)
    Y = (
        None
        if view.target is None
        else view.target[:, None].astype(np.float32, copy=False)
    )
    return X, Y


def fit_normalizer(train_X: np.ndarray, train_Y: np.ndarray, mode: str):
    """Fit min-max parameters using labelled partial-discharge tails only."""

    if train_X.size == 0 or train_Y is None or train_Y.size == 0:
        raise ValueError("The labelled training set cannot be empty")
    if mode == "per_feature":
        xmin = train_X.min(axis=0, keepdims=True)
        xmax = train_X.max(axis=0, keepdims=True)
    elif mode == "per_channel":
        xmin = train_X.min(keepdims=True)
        xmax = train_X.max(keepdims=True)
    else:
        raise ValueError("normalization_mode must be 'per_feature' or 'per_channel'")
    return {
        "xmin": xmin,
        "xmax": xmax,
        "ymin": train_Y.min(keepdims=True),
        "ymax": train_Y.max(keepdims=True),
        "mode": mode,
    }


def apply_normalizer(X: np.ndarray, Y: np.ndarray | None, norm):
    """Apply labelled-fit min-max parameters without fitting on other roles."""

    eps = 1e-8
    Xn = (X - norm["xmin"]) / np.maximum(norm["xmax"] - norm["xmin"], eps)
    Xn = np.clip(Xn, 0.0, 1.0).astype(np.float32, copy=False)
    if Y is None:
        return Xn, None
    Yn = (Y - norm["ymin"]) / np.maximum(norm["ymax"] - norm["ymin"], eps)
    return Xn, np.clip(Yn, 0.0, 1.0).astype(np.float32, copy=False)


def extract_tail_features_from_X(X: np.ndarray) -> np.ndarray:
    if X.ndim != 3:
        raise ValueError("Expected tail input with shape (N, C, T)")
    return X.reshape(X.shape[0], -1)
