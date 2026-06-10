# -*- coding: utf-8 -*-

import numpy as np
from scipy.io import loadmat
import config as cfg

# ======================================================================
# ======================================================================

def load_all_mats(mat_path: str):
    S = loadmat(mat_path)
    Q = S["processed_Qd_mat"]
    V = S["processed_Vd_mat"]
    assert Q.shape == V.shape and Q.shape[0] == 201
    return V, Q

# ======================================================================
# ======================================================================

def build_ordered_windows(cfg):
    grid = np.linspace(cfg.V_MIN, cfg.V_MAX, 201)
    idx_tail_hi = np.searchsorted(grid, cfg.VSEG_LOW, side="right") - 1
    idx_upp_hi = np.searchsorted(grid, cfg.VSEG_UPP, side="right") - 1
    start = idx_tail_hi + 1
    end = idx_upp_hi
    win = []
    if end > start:
        width = max(2, (end - start) // max(1, cfg.MAX_SEGS))
        j = start
        while j + width <= end and len(win) < cfg.MAX_SEGS:
            win.append((j, j + width))
            j += width
    return win, idx_tail_hi, grid

def resample_1d(x: np.ndarray, target_len: int) -> np.ndarray:
    if len(x) == target_len:
        return x.astype(np.float32)
    xp = np.linspace(0, 1, len(x))
    xq = np.linspace(0, 1, target_len)
    return np.interp(xq, xp, x).astype(np.float32)

def make_features(V_seg: np.ndarray, Q_seg: np.ndarray) -> np.ndarray:
    eps = 1e-6
    dQ = np.gradient(Q_seg)
    dV = np.gradient(V_seg)
    dQdV = dQ / np.clip(dV, eps, None)
    return np.stack([Q_seg, dQdV], axis=0).astype(np.float32)

def build_subset(V, Q, idx_array: np.ndarray, cfg, win, idx_tail_hi):
    X_tail, S_list, y_cap = [], [], []
    for i in idx_array:
        V_i = V[:, i]
        Q_i = Q[:, i]
        y_cap.append(np.array([Q_i[-1]], dtype=np.float32))
        
        V_tail = V_i[:idx_tail_hi+1]
        Q_tail = Q_i[:idx_tail_hi+1]
        V_tail_rs = resample_1d(V_tail, cfg.TAIL_LEN)
        Q_tail_rs = resample_1d(Q_tail, cfg.TAIL_LEN)
        X_tail.append(make_features(V_tail_rs, Q_tail_rs))
        
        segs = []
        for (a, b) in win:
            V_rs = resample_1d(V_i[a:b], cfg.SEG_LEN)
            Q_rs = resample_1d(Q_i[a:b], cfg.SEG_LEN)
            segs.append(make_features(V_rs, Q_rs))
        S_list.append(segs)
        
    X_tail = np.stack(X_tail, axis=0).astype(np.float32)
    y_cap = np.stack(y_cap, axis=0).astype(np.float32)
    z_meta = np.zeros((len(idx_array), cfg.model.get("d_meta", 16)), dtype=np.float32)
    return dict(X_tail=X_tail, S_list=S_list, z_meta=z_meta, y_cap=y_cap)

# ======================================================================
# ======================================================================

def active_compute_split_sizes(N: int) -> dict:
    """Compute labelling-budget split sizes for train, pool, val, and test."""
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

    assert num_tr_initial + approx_pool + n_val + n_test == N, "Labelling-budget split sizes must sum to N"
    return dict(train_initial=num_tr_initial, pool=approx_pool, val=n_val, test=n_test)


def active_split_indices(N: int, seed: int = 0) -> dict:
    """Return global cell indices for train, pool, validation, and test sets."""
    sizes = active_compute_split_sizes(N)
    rng = np.random.RandomState(seed)
    all_idx = np.arange(N)
    rng.shuffle(all_idx)

    i0 = 0
    tr_init = np.sort(all_idx[i0 : i0 + sizes["train_initial"]])
    i0 += sizes["train_initial"]
    
    pool = np.sort(all_idx[i0 : i0 + sizes["pool"]])
    i0 += sizes["pool"]
    
    val = np.sort(all_idx[i0 : i0 + sizes["val"]])
    i0 += sizes["val"]
    
    test = np.sort(all_idx[i0:])

    return dict(train=tr_init, pool=pool, val=val, test=test)

# ======================================================================
# ======================================================================

def fit_normalizer_tail(X_tail: np.ndarray, mode: str):
    C, T = X_tail.shape[1], X_tail.shape[2]
    if mode == "per_feature":
        mins = X_tail.min(axis=0)
        maxs = X_tail.max(axis=0)
    else:
        mins = X_tail.min(axis=(0, 2), keepdims=True)
        maxs = X_tail.max(axis=(0, 2), keepdims=True)
        mins = np.repeat(mins, T, axis=2).squeeze(0)
        maxs = np.repeat(maxs, T, axis=2).squeeze(0)
    return {"mins": mins.astype(np.float32), "maxs": maxs.astype(np.float32), "mode": mode}

def apply_normalizer_tail(X_tail: np.ndarray, norm):
    mins, maxs = norm["mins"], norm["maxs"]
    denom = np.clip(maxs - mins, 1e-8, None)
    return ((X_tail - mins) / denom).astype(np.float32)

def fit_normalizer_cap(y_cap: np.ndarray):
    return {"min": float(y_cap.min()), "max": float(y_cap.max())}

def apply_normalizer_cap(y_cap: np.ndarray, ncap):
    denom = max(ncap["max"] - ncap["min"], 1e-8)
    return ((y_cap - ncap["min"]) / denom).astype(np.float32)

def inverse_cap(y_norm: np.ndarray, ncap):
    return y_norm * (ncap["max"] - ncap["min"]) + ncap["min"]

