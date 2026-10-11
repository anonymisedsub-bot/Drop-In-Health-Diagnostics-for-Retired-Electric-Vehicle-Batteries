"""Fixed-length zero-padding adapter for the shared ReBATT protocol."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import config as cfg


MODEL_ROOT = Path(__file__).resolve().parents[1]
if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))

from baseline_support.rebatt_protocol import build_voltage_windows, role_view


def build_padding_xy(
    data,
    indices,
    role: str,
    vseg_low: float,
    voltage_removal_window: float = 0.20,
):
    """Return raw fixed-length inputs; real partial gaps remain NaN for now."""

    indices = np.asarray(indices, dtype=np.int64)
    view = role_view(
        data,
        role,
        indices,
        vseg_low=vseg_low,
        voltage_removal_window=voltage_removal_window,
    )
    windows = build_voltage_windows(data.voltage_grid, vseg_low)
    stop = windows.upstream_windows[-1][1]
    raw = view.q[:stop, :].T[:, None, :]
    X = raw.astype(np.float32, copy=False)
    Y = (
        None
        if view.target is None
        else view.target[:, None].astype(np.float32, copy=False)
    )
    return X, Y


def fit_normalizer(train_X: np.ndarray, train_Y: np.ndarray, mode: str):
    """Fit min-max parameters from observed labelled partial positions only."""

    if train_X.size == 0 or train_Y is None or train_Y.size == 0:
        raise ValueError("The labelled training set cannot be empty")
    if np.isinf(train_X).any() or not np.isfinite(train_Y).all():
        raise ValueError(
            "Labelled inputs cannot contain Inf and capacity targets must be finite"
        )
    finite = np.isfinite(train_X)
    if mode == "per_feature":
        observed_feature = finite.any(axis=0, keepdims=True)
        xmin = np.min(np.where(finite, train_X, np.inf), axis=0, keepdims=True)
        xmax = np.max(np.where(finite, train_X, -np.inf), axis=0, keepdims=True)
        xmin = np.where(observed_feature, xmin, 0.0)
        xmax = np.where(observed_feature, xmax, 1.0)
    elif mode == "per_channel":
        if not finite.any():
            raise ValueError("The labelled inputs must contain observed positions")
        observed_feature = np.ones((1, *train_X.shape[1:]), dtype=bool)
        xmin = np.nanmin(train_X, keepdims=True)
        xmax = np.nanmax(train_X, keepdims=True)
    else:
        raise ValueError("normalization_mode must be 'per_feature' or 'per_channel'")
    return {
        "xmin": xmin,
        "xmax": xmax,
        "ymin": train_Y.min(keepdims=True),
        "ymax": train_Y.max(keepdims=True),
        "mode": mode,
        "observed_feature": observed_feature,
    }


def apply_normalizer_with_padding(
    X: np.ndarray,
    Y: np.ndarray | None,
    norm,
):
    """Normalize observed values, then encode unavailable partial positions as zero."""

    eps = 1e-8
    finite = np.isfinite(X)
    Xn = (X - norm["xmin"]) / np.maximum(norm["xmax"] - norm["xmin"], eps)
    Xn = np.clip(Xn, 0.0, 1.0)
    Xn = np.where(finite & norm["observed_feature"], Xn, 0.0).astype(np.float32, copy=False)
    if Y is None:
        return Xn, None
    Yn = (Y - norm["ymin"]) / np.maximum(norm["ymax"] - norm["ymin"], eps)
    return Xn, np.clip(Yn, 0.0, 1.0).astype(np.float32, copy=False)


def apply_normalizer(X: np.ndarray, Y: np.ndarray | None, norm):
    """Compatibility alias for historical imports."""

    return apply_normalizer_with_padding(X, Y, norm)


def get_voltage_config() -> dict:
    """Voltage boundaries used by training-only prefix masking."""

    return {
        "V_MIN": cfg.V_MIN,
        "V_MAX": cfg.V_MAX,
        "DV": cfg.DV,
        "VSEG_LOW": cfg.VSEG_LOW,
        "VSEG_UPP": cfg.VSEG_UPP,
        "segment_length": cfg.segment_length,
        "TAIL_LEN": cfg.TAIL_LEN,
        "UPSTREAM_LEN": cfg.UPSTREAM_LEN,
        "UPSTREAM_START_IDX": cfg.UPSTREAM_START_IDX,
        "MAX_SEGS": cfg.MAX_SEGS,
        "SUBSEG_LEN": cfg.SUBSEG_LEN,
    }


def get_curriculum_keep(epoch: int, cl_config: dict) -> int:
    """Return the active one-based training epoch curriculum stage."""

    value = 0
    for boundary, keep in zip(
        cl_config["curriculum_epochs"],
        cl_config["curriculum_keep"],
    ):
        if epoch < boundary:
            break
        value = keep
    return min(int(value), cfg.MAX_SEGS)


def count_observed_upper_slots(raw_X: np.ndarray, voltage_cfg: dict) -> np.ndarray:
    """Count per-sample upper slots containing real partial-discharge observations.

    Padding preserves a partly observed final slot, so it is available to keep.
    This must run before normalization, when unavailable points are still NaN.
    """

    start = int(voltage_cfg["UPSTREAM_START_IDX"])
    max_segs = int(voltage_cfg["MAX_SEGS"])
    width = int(voltage_cfg["SUBSEG_LEN"])
    if raw_X.ndim != 3 or raw_X.shape[1] != 1 or raw_X.shape[2] != start + max_segs * width:
        raise ValueError("Raw padding inputs do not match the configured voltage slots")
    observed = np.isfinite(raw_X[:, 0, start:]).reshape(-1, max_segs, width).any(axis=2)
    if np.any(np.maximum.accumulate(~observed, axis=1) & observed):
        raise ValueError("Observed upper slots must form a voltage-prefix")
    return observed.sum(axis=1).astype(np.int64)


def validate_upper_slot_counts(counts: np.ndarray, n_rows: int, max_segs: int) -> np.ndarray:
    counts = np.asarray(counts)
    if (counts.shape != (n_rows,) or not np.issubdtype(counts.dtype, np.integer)
            or np.any(counts < 0) or np.any(counts > max_segs)):
        raise ValueError("Each training row needs an integer upper-slot count within [0, MAX_SEGS]")
    return counts


def apply_training_prefix_mask(
    X: np.ndarray,
    cl_config: dict,
    rng: np.random.RandomState,
    voltage_cfg: dict,
    available_counts: np.ndarray,
) -> np.ndarray:
    """Copy normalized tree inputs and mask one contiguous upper-fragment suffix per row."""

    masked = X.copy()
    max_keep = min(int(cl_config["curriculum_keep"]), voltage_cfg["MAX_SEGS"])
    start = voltage_cfg["UPSTREAM_START_IDX"]
    width = voltage_cfg["SUBSEG_LEN"]
    available_counts = validate_upper_slot_counts(
        available_counts, len(masked), voltage_cfg["MAX_SEGS"]
    )
    for row, available in zip(masked, available_counts):
        k_available = min(max_keep, int(available))
        if k_available == 0 or rng.rand() < cl_config["p_drop_all"]:
            keep = 0
        elif rng.rand() < cl_config["p_keep_one"]:
            keep = rng.randint(1, k_available + 1)
        else:
            keep = k_available
        row[:, start + keep * width:] = 0.0
    return masked


def deterministic_mask_by_k(X_full: np.ndarray, k: int, voltage_cfg: dict):
    """Retained for trainer import compatibility; the ReBATT runner never calls it."""

    X_masked = X_full.copy()
    start = voltage_cfg["UPSTREAM_START_IDX"] + int(k) * voltage_cfg["SUBSEG_LEN"]
    X_masked[:, :, start:] = 0.0
    return X_masked
