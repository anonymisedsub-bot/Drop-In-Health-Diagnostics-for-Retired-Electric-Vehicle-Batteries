"""KNN pseudo labels from known training capacities only."""
from operator import index

import numpy as np
from sklearn.neighbors import NearestNeighbors


def _knn_settings(k, temperature):
    try:
        k = index(k)
    except TypeError as exc:
        raise ValueError("knn_k must be a positive integer") from exc
    temperature = float(temperature)
    if k < 1 or not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("knn_k and knn_temperature must be positive and finite")
    return k, temperature


def _training_arrays(X_lab, Y_lab, X_unlab):
    X_lab, Y_lab, X_unlab = (
        np.asarray(value, dtype=np.float32) for value in (X_lab, Y_lab, X_unlab)
    )
    if X_lab.ndim < 2 or X_unlab.ndim != X_lab.ndim:
        raise ValueError("Labelled and unlabelled inputs must have matching feature axes")
    if X_lab.shape[0] == 0 or X_lab.shape[1:] != X_unlab.shape[1:]:
        raise ValueError("Nonempty labelled inputs and matching feature shapes are required")
    if np.prod(X_lab.shape[1:]) == 0 or Y_lab.shape != (X_lab.shape[0], 1):
        raise ValueError("Features must be nonempty and labels must have shape (N, 1)")
    if not all(np.isfinite(value).all() for value in (X_lab, Y_lab, X_unlab)):
        raise ValueError("KNN inputs and known capacities must be finite")
    return X_lab, Y_lab, X_unlab

def knn_pseudo_label(
    X_lab, Y_lab,
    X_unlab,
    k=8,
    temperature=0.1,
    metric="euclidean",
):
    """
    X_*: Normalized arrays with shape (N, C, T); distances use this feature space.
    Y_lab: Normalized targets with shape (N, 1).
    Returns: Y_unlab_pl with shape (N_unlab, 1).
    """
    k, temperature = _knn_settings(k, temperature)
    X_lab, Y_lab, X_unlab = _training_arrays(X_lab, Y_lab, X_unlab)
    if X_unlab.shape[0] == 0:
        return np.empty((0, 1), dtype=np.float32)

    # Flatten (N, C, T) to (N, C*T).
    Nl = X_lab.shape[0]
    Xu2 = X_unlab.reshape(X_unlab.shape[0], -1)
    Xl2 = X_lab.reshape(Nl, -1)
    Yl  = Y_lab.reshape(Nl, 1)

    nn = NearestNeighbors(n_neighbors=min(k, Nl), metric=metric)
    nn.fit(Xl2)
    dists, idxs = nn.kneighbors(Xu2, return_distance=True)  # (Nu, k)

    # Temperature-weighted neighbors: w_i is proportional to exp(-d_i / tau).
    # Subtract each row's minimum distance for numerical stability.
    d_adj = dists - dists.min(axis=1, keepdims=True)
    w = np.exp(-d_adj / temperature)
    w = w / np.maximum(w.sum(axis=1, keepdims=True), 1e-12)  # Normalize weights.

    Y_neighbors = Yl[idxs].reshape(X_unlab.shape[0], -1)  # (Nu, k)
    Y_pl = np.sum(w * Y_neighbors, axis=1, keepdims=True)  # (Nu,1)
    return Y_pl


def prepare_pseudo_training_data(X_lab, Y_lab, X_unlab, config, seed):
    """Return original inputs, real/pseudo targets, global weights and metadata.

    Distances use the original normalized tail features directly.
    KNN reference points and capacities come exclusively from the labelled set.
    Validation and test arrays are deliberately absent from this interface.
    """
    X_lab, Y_lab, X_unlab = _training_arrays(X_lab, Y_lab, X_unlab)
    k, temperature = _knn_settings(
        getattr(config, "knn_k", 10), getattr(config, "knn_temperature", 0.1)
    )
    weight = float(getattr(config, "knn_pseudo_weight", 1.0))
    if not np.isfinite(weight) or weight < 0:
        raise ValueError("knn_pseudo_weight must be finite and non-negative")
    metric = str(getattr(config, "knn_metric", "euclidean"))
    labelled_count, target_count = len(X_lab), len(X_unlab)
    metadata = {
        "pseudo_label_algorithm": "temperature_weighted_knn",
        "knn_k": k,
        "knn_effective_k": min(k, labelled_count),
        "knn_temperature": temperature,
        "knn_metric": metric,
        "knn_pseudo_weight": weight,
        "pseudo_label_count": 0,
        "pseudo_label_feature_space": "normalized_tail",
        "capacity_loss": "sample_weighted_mse",
        "capacity_loss_normalization": "(sum_labelled_sse + w * sum_pseudo_sse)/(Nl + w * Nu)",
    }
    if target_count == 0 or weight == 0:
        return X_lab, Y_lab, np.ones(labelled_count, dtype=np.float32), metadata

    mixed_x = np.concatenate((X_lab, X_unlab), axis=0)
    pseudo_y = knn_pseudo_label(
        X_lab, Y_lab, X_unlab, k=k, temperature=temperature, metric=metric,
    ).astype(np.float32)
    mixed_y = np.concatenate((Y_lab, pseudo_y), axis=0)
    sample_weights = np.concatenate((
        np.ones(labelled_count, dtype=np.float64),
        np.full(target_count, weight, dtype=np.float64),
    ))
    # Normalize once over all samples. Re-normalizing each mini-batch would
    # cancel the pseudo weight in batches containing only pseudo-labelled rows.
    sample_weights /= sample_weights.mean()
    metadata["pseudo_label_count"] = target_count
    return mixed_x, mixed_y, sample_weights.astype(np.float32), metadata
