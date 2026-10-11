"""KNN pseudo labels for the group-3 MLP (SSL-PL) baseline."""

from __future__ import annotations

import numpy as np


def knn_pseudo_label(
    labelled_x: np.ndarray,
    labelled_y: np.ndarray,
    unlabelled_x: np.ndarray,
    *,
    k: int = 10,
    temperature: float = 0.1,
) -> np.ndarray:
    """Distance-weighted KNN targets in the normalized tail-feature space."""

    labelled_x = np.asarray(labelled_x, dtype=np.float32)
    labelled_y = np.asarray(labelled_y, dtype=np.float32)
    unlabelled_x = np.asarray(unlabelled_x, dtype=np.float32)
    if labelled_x.ndim != 3 or unlabelled_x.ndim != 3:
        raise ValueError("KNN inputs must have shape (samples, channels, sequence)")
    if labelled_x.shape[1:] != unlabelled_x.shape[1:]:
        raise ValueError("Labelled and unlabelled inputs must share a feature shape")
    if labelled_y.shape != (len(labelled_x), 1) or not len(labelled_x):
        raise ValueError("KNN requires a nonempty labelled set and one target per sample")
    if int(k) < 1 or not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("k and temperature must be positive")
    if not np.isfinite(labelled_x).all() or not np.isfinite(unlabelled_x).all():
        raise ValueError("KNN inputs must be finite")
    if not np.isfinite(labelled_y).all():
        raise ValueError("KNN targets must be finite")
    if not len(unlabelled_x):
        return np.empty((0, 1), dtype=np.float32)

    labelled = labelled_x.reshape(len(labelled_x), -1).astype(np.float64)
    unlabelled = unlabelled_x.reshape(len(unlabelled_x), -1).astype(np.float64)
    labelled_norm = np.square(labelled).sum(axis=1)[None, :]
    neighbours = min(int(k), len(labelled))
    pseudo = np.empty((len(unlabelled), 1), dtype=np.float32)

    for start in range(0, len(unlabelled), 512):
        block = unlabelled[start : start + 512]
        distances_sq = np.maximum(
            np.square(block).sum(axis=1)[:, None]
            + labelled_norm
            - 2.0 * block @ labelled.T,
            0.0,
        )
        nearest = np.argpartition(distances_sq, neighbours - 1, axis=1)[:, :neighbours]
        distances = np.sqrt(np.take_along_axis(distances_sq, nearest, axis=1))
        weights = np.exp(-(distances - distances.min(axis=1, keepdims=True)) / temperature)
        weights /= weights.sum(axis=1, keepdims=True)
        pseudo[start : start + len(block), 0] = np.sum(
            weights * labelled_y[nearest, 0], axis=1
        )
    return pseudo


def prepare_ssl_pl_training(
    labelled_x: np.ndarray,
    labelled_y: np.ndarray,
    unlabelled_x: np.ndarray,
    *,
    k: int = 10,
    temperature: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Append KNN-labelled pool inputs; no pool targets are read or required."""

    pseudo_y = knn_pseudo_label(
        labelled_x, labelled_y, unlabelled_x, k=k, temperature=temperature
    )
    if not len(unlabelled_x):
        return labelled_x, labelled_y
    return (
        np.concatenate((labelled_x, unlabelled_x), axis=0),
        np.concatenate((labelled_y, pseudo_y), axis=0),
    )
