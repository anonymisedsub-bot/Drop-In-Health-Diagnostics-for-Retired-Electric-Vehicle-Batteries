import numpy as np
from scipy.io import loadmat
import config as cfg

def load_q_mat(mat_path: str) -> np.ndarray:
    S = loadmat(mat_path)
    Q = S["processed_Qd_mat"]  # 201 x Ncells
    assert Q.ndim == 2, "processed_Qd_mat shape should be (time, cells)"
    return Q  # (T_full, Ncells)

def build_XY(Q: np.ndarray, indices: np.ndarray, segment_length: int):
    """Build X from the first segment_length rows and Y from the final row.
       Returns X: (N, 1, Tseg), Y: (N, 1).
    """
    if indices.size == 0:
        return np.empty((0,1,segment_length)), np.empty((0,1))
    T_full, _ = Q.shape
    assert segment_length <= T_full, "segment_length exceeds the available time steps"
    Q_seg = Q[:segment_length, :][:, indices]      # (Tseg, N)
    Q_target = Q[-1, :][indices]                   # (N,)
    X = np.expand_dims(Q_seg.T, axis=1)            # (N, 1, Tseg)
    Y = Q_target.reshape(-1, 1)                    # (N, 1)
    return X, Y

def fit_normalizer(train_X: np.ndarray, train_Y: np.ndarray, mode: str):
    """Normalize to [0, 1], using per_feature or per_channel statistics."""
    if train_X.size == 0:
        raise ValueError("Training set is empty; cannot fit normalization parameters. Check split ratios and few-shot settings.")
    # X: (N, C=1, T)
    if mode == "per_feature":
        # Compute each time-step feature's statistics across samples.
        xmin = train_X.min(axis=0, keepdims=True)  # (1, 1, T)
        xmax = train_X.max(axis=0, keepdims=True)
    elif mode == "per_channel":
        # Compute channel min/max across all samples and time steps (C=1).
        xmin = train_X.min(keepdims=True)          # (1, 1, 1)
        xmax = train_X.max(keepdims=True)
    else:
        raise ValueError("normalization_mode must be 'per_feature' or 'per_channel'")

    ymin = train_Y.min(keepdims=True)              # (1,1)
    ymax = train_Y.max(keepdims=True)

    return {"xmin": xmin, "xmax": xmax, "ymin": ymin, "ymax": ymax, "mode": mode}

def apply_normalizer(X: np.ndarray, Y: np.ndarray, norm):
    eps = 1e-8
    xmin, xmax = norm["xmin"], norm["xmax"]
    ymin, ymax = norm["ymin"], norm["ymax"]
    Xn = (X - xmin) / (np.maximum(xmax - xmin, eps))
    Yn = (Y - ymin) / (np.maximum(ymax - ymin, eps))
    Xn = np.clip(Xn, 0.0, 1.0); Yn = np.clip(Yn, 0.0, 1.0)
    return Xn, Yn

def active_compute_split_sizes(N: int) -> dict:
    """
    Use cfg.num_tr_initial / cfg.num_tr_per_iter / cfg.al_pool_ratio / cfg.al_val_ratio
    to size the four subsets: train_initial, pool, val, test.
    The pool size must be divisible by num_tr_per_iter; all sizes sum to N.
    """
    num_tr_initial = int(cfg.num_tr_initial)
    num_tr_per_iter = int(cfg.num_tr_per_iter)
    approx_pool = int(round(cfg.al_pool_ratio * N))
    if num_tr_per_iter > 0:
        approx_pool = (approx_pool // num_tr_per_iter) * num_tr_per_iter

    n_val = int(round(cfg.al_val_ratio * N))
    used = num_tr_initial + approx_pool + n_val
    n_test = max(0, N - used)

    if used > N:
        overflow = used - N
        approx_pool = max(0, approx_pool - overflow)
        if num_tr_per_iter > 0:
            approx_pool = (approx_pool // num_tr_per_iter) * num_tr_per_iter
        used = num_tr_initial + approx_pool + n_val
        n_test = max(0, N - used)

    assert num_tr_initial + approx_pool + n_val + n_test == N, "AL split sizes must sum to N"
    return dict(train_initial=num_tr_initial, pool=approx_pool, val=n_val, test=n_test)

def active_split_indices(N: int, seed: int = 0) -> dict:
    """
    Return the global indices of all four subsets.
    """
    sizes = active_compute_split_sizes(N)
    rng = np.random.RandomState(seed)
    all_idx = np.arange(N)
    rng.shuffle(all_idx)

    i0 = 0
    tr_init = np.sort(all_idx[i0:i0+sizes["train_initial"]]); i0 += sizes["train_initial"]
    pool    = np.sort(all_idx[i0:i0+sizes["pool"]]);          i0 += sizes["pool"]
    val     = np.sort(all_idx[i0:i0+sizes["val"]]);           i0 += sizes["val"]
    test    = np.sort(all_idx[i0:])
    return dict(train=tr_init, pool=pool, val=val, test=test)

def build_XY_from_global(Q: np.ndarray, glob_indices: np.ndarray, segment_length: int):
    """
    Like build_XY, but with explicit global indices.
    """
    return build_XY(Q, glob_indices, segment_length)
