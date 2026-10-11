# dataset.py
# -*- coding: utf-8 -*-
import numpy as np
from scipy.io import loadmat
from typing import Dict, List
import config as cfg

# ==================================================================
# 1. Data loading and splitting
# ==================================================================

def load_q_mat(mat_path: str) -> np.ndarray:
    S = loadmat(mat_path)
    Q = S["processed_Qd_mat"]
    assert Q.ndim == 2, "processed_Qd_mat shape should be (time, cells)"
    if np.isnan(Q).any():
        raise ValueError("Error: Raw .mat data contains NaN values.")
    return Q

def active_compute_split_sizes(N: int) -> dict:
    num_tr_initial = int(cfg.num_tr_initial)
    num_tr_per_iter = int(cfg.num_tr_per_iter)
    approx_pool = int(round(cfg.al_pool_ratio * N))
    if num_tr_per_iter > 0:
        approx_pool = (approx_pool // num_tr_per_iter) * num_tr_per_iter
    n_val = int(round(cfg.al_val_ratio * N))
    used = num_tr_initial + approx_pool + n_val
    n_test = max(0, N - used)
    return dict(train_initial=num_tr_initial, pool=approx_pool, val=n_val, test=n_test)

def active_split_indices(N: int, seed: int = 0) -> dict:
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

def build_XY_from_global(Q: np.ndarray, indices: np.ndarray, segment_length: int):
    if indices.size == 0:
        return np.empty((0,1,segment_length)), np.empty((0,1))
    Q_seg = Q[:segment_length, :][:, indices]
    Q_target = Q[-1, :][indices]
    X = np.expand_dims(Q_seg.T, axis=1) 
    Y = Q_target.reshape(-1, 1)         
    return X, Y

# ==================================================================
# 2. Normalization utilities
# ==================================================================

def fit_normalizer(train_X: np.ndarray, train_Y: np.ndarray, mode: str):
    if train_X.size == 0:
        raise ValueError("Training set is empty.")
    if mode == "per_feature":
        xmin = train_X.min(axis=0, keepdims=True)
        xmax = train_X.max(axis=0, keepdims=True)
    elif mode == "per_channel":
        xmin = train_X.min(keepdims=True)
        xmax = train_X.max(keepdims=True)
    ymin = train_Y.min(keepdims=True)
    ymax = train_Y.max(keepdims=True)
    return {"xmin": xmin, "xmax": xmax, "ymin": ymin, "ymax": ymax, "mode": mode}

def apply_normalizer(X: np.ndarray, Y: np.ndarray, norm):
    eps = 1e-8
    xmin, xmax, ymin, ymax = norm["xmin"], norm["xmax"], norm["ymin"], norm["ymax"]
    Xn = (X - xmin) / (np.maximum(xmax - xmin, eps))
    Yn = (Y - ymin) / (np.maximum(ymax - ymin, eps))
    return np.clip(Xn, 0.0, 1.0), np.clip(Yn, 0.0, 1.0)

# ==================================================================
# 3. Physics-informed masking
# ==================================================================

def get_voltage_config() -> dict:
    return {
        "V_MIN": cfg.V_MIN, "V_MAX": cfg.V_MAX, "DV": cfg.DV,
        "VSEG_LOW": cfg.VSEG_LOW, "VSEG_UPP": cfg.VSEG_UPP,
        "segment_length": cfg.segment_length, "TAIL_LEN": cfg.TAIL_LEN,
        "UPSTREAM_LEN": cfg.UPSTREAM_LEN, "UPSTREAM_START_IDX": cfg.UPSTREAM_START_IDX,
        "MAX_SEGS": cfg.MAX_SEGS, "SUBSEG_LEN": cfg.SUBSEG_LEN
    }

def simulate_initial_voltage(n_cells: int, seed: int) -> np.ndarray:
    """Use the configured VSEG_LOW as the lower rejection-sampling boundary."""
    V_range_low, V_range_high = 2.7, 3.1
    mu, sigma = (V_range_low + V_range_high) / 2.0, 0.05
    rng = np.random.default_rng(seed)
    V_init = np.empty(n_cells, dtype=float)
    for i in range(n_cells):
        while True:
            v = rng.normal(mu, sigma)
            if v > cfg.VSEG_LOW: 
                V_init[i] = v
                break
    return V_init

def get_available_segments_from_v_init(V_init_batch: np.ndarray, v_cfg: dict) -> List[List[int]]:
    available_indices_list = []
    for v_init in V_init_batch:
        v_init_clipped = np.clip(v_init, v_cfg["V_MIN"], v_cfg["V_MAX"])
        v_init_idx = int(round((v_init_clipped - v_cfg["V_MIN"]) / v_cfg["DV"]))
        if v_init_idx <= v_cfg["UPSTREAM_START_IDX"]:
            num_available = 0
        else:
            num_available = int((v_init_idx - v_cfg["UPSTREAM_START_IDX"] + 1) // v_cfg["SUBSEG_LEN"])
            num_available = min(max(0, num_available), v_cfg["MAX_SEGS"])
        available_indices_list.append(list(range(num_available)))
    return available_indices_list

def get_curriculum_keep(epoch: int, cl_config: dict) -> int:
    epochs, keeps = cl_config["curriculum_epochs"], cl_config["curriculum_keep"]
    k_curr = 0
    for i in range(len(epochs)):
        if epoch >= epochs[i]: k_curr = keeps[i]
        else: break
    return min(k_curr, cfg.MAX_SEGS)

def apply_masking(
    X_full: np.ndarray, V_init_subset: np.ndarray, mode: str, 
    cl_epoch: int, cl_config: dict, rng: np.random.RandomState, voltage_cfg: dict
) -> np.ndarray:
    N = X_full.shape[0]
    if N == 0: return X_full
    X_masked, v_cfg = X_full.copy(), voltage_cfg
    
    if mode == "pool_test":
        base_availability = get_available_segments_from_v_init(V_init_subset, v_cfg)
    elif mode in ["train_dl", "train_ml", "val"]:
        base_availability = [list(range(v_cfg["MAX_SEGS"])) for _ in range(N)]
    else:
        raise ValueError(f"Unknown masking mode: {mode}")

    final_used_indices = []
    if mode == "train_dl":
        k_curr = get_curriculum_keep(cl_epoch, cl_config)
        for base_avail in base_availability:
            k_available = min(len(base_avail), k_curr)
            if k_available == 0 or rng.rand() < cl_config["p_drop_all"]: k_used = 0
            elif rng.rand() < cl_config["p_keep_one"]: k_used = rng.randint(1, max(1, k_available) + 1)
            else: k_used = k_available
            final_used_indices.append(list(np.arange(k_used)))

    elif mode == "train_ml":
        k_curr = cl_config["curriculum_keep"]
        for base_avail in base_availability:
            k_available = min(len(base_avail), k_curr)
            if k_available == 0 or rng.rand() < cl_config["p_drop_all"]: k_used = 0
            elif rng.rand() < cl_config["p_keep_one"]: k_used = rng.randint(1, max(1, k_available) + 1)
            else: k_used = k_available
            final_used_indices.append(list(np.arange(k_used)))

    elif mode == "val":
        for base_avail in base_availability:
            final_used_indices.append(list(np.arange(rng.randint(0, len(base_avail) + 1))))
            
    elif mode == "pool_test":
        final_used_indices = base_availability
        
    seg_mask = np.zeros((N, v_cfg["MAX_SEGS"]), dtype=bool)
    for i, indices in enumerate(final_used_indices):
        if indices: seg_mask[i, indices] = True
            
    upstream_mask = np.repeat(seg_mask, v_cfg["SUBSEG_LEN"], axis=1)
    full_mask = np.ones((N, v_cfg["TAIL_LEN"] + v_cfg["UPSTREAM_LEN"]), dtype=bool)
    full_mask[:, v_cfg["UPSTREAM_START_IDX"]:] = upstream_mask
    
    X_masked = X_masked * full_mask.reshape(N, 1, v_cfg["segment_length"])
    return X_masked



def deterministic_mask_by_k(X_full: np.ndarray, k: int, voltage_cfg: dict) -> np.ndarray:
    """
    Deterministic mask: retain the tail and the first k upstream segments.
    - k=0: retain only the tail (TAIL_LEN).
    - k=MAX_SEGS: retain all input data.
    
    Return a copy of X_full.
    """
    N = X_full.shape[0]
    if N == 0:
        return X_full
    
    v_cfg = voltage_cfg
    X_masked = X_full.copy()
    
    # 1. Identify upstream segments to zero out.
    # Retain [0, k-1], so mask [k, MAX_SEGS-1].
    indices_to_zero_segments = np.arange(k, v_cfg["MAX_SEGS"])
    
    if indices_to_zero_segments.size == 0:
        # No masking is required when k >= MAX_SEGS.
        return X_masked

    # 2. Zero the positions belonging to those segments.
    for seg_idx in indices_to_zero_segments:
        # Compute start/end indices in the full (N, 1, segment_length) tensor.
        start_idx = v_cfg["UPSTREAM_START_IDX"] + (seg_idx * v_cfg["SUBSEG_LEN"])
        end_idx = start_idx + v_cfg["SUBSEG_LEN"]
        
        # Apply the mask by setting values to zero.
        X_masked[:, :, start_idx:end_idx] = 0.0
            
    return X_masked