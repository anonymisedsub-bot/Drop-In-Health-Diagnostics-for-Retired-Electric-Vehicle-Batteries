# -*- coding: utf-8 -*-

import sys
from pathlib import Path

import numpy as np

import config as cfg


MODEL_ROOT = Path(__file__).resolve().parent
if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))

from common.rebatt_protocol import role_view

def build_ordered_windows(cfg, voltage_grid=None):
    dv = float(cfg.DV)
    max_segs = int(cfg.MAX_SEGS)

    if dv <= 0:
        raise ValueError("DV must be positive")
    if max_segs <= 0:
        raise ValueError("MAX_SEGS must be positive")

    if voltage_grid is None:
        n_steps = int(round((cfg.V_MAX - cfg.V_MIN) / dv))
        if n_steps <= 0 or not np.isclose(
            cfg.V_MIN + n_steps * dv,
            cfg.V_MAX,
            rtol=0.0,
            atol=dv * 1e-6,
        ):
            raise ValueError("V_MIN and V_MAX must define a positive uniform DV grid")
        grid = cfg.V_MIN + np.arange(n_steps + 1, dtype=float) * dv
    else:
        grid = np.asarray(voltage_grid, dtype=float).squeeze()
        if grid.ndim != 1 or grid.size < 2:
            raise ValueError("voltage_grid must be a one-dimensional grid")
        if not np.isclose(grid[0], cfg.V_MIN, rtol=0.0, atol=dv * 1e-6):
            raise ValueError("voltage_grid does not start at V_MIN")
        if not np.isclose(grid[-1], cfg.V_MAX, rtol=0.0, atol=dv * 1e-6):
            raise ValueError("voltage_grid does not end at V_MAX")
        if not np.allclose(np.diff(grid), dv, rtol=0.0, atol=dv * 1e-6):
            raise ValueError("voltage_grid does not use the configured DV")

    def aligned_grid_index(voltage: float, name: str) -> int:
        idx = int(round((float(voltage) - cfg.V_MIN) / dv))
        if idx < 0 or idx >= grid.size:
            raise ValueError(f"{name}={voltage} lies outside the voltage grid")
        if not np.isclose(grid[idx], voltage, rtol=0.0, atol=dv * 1e-6):
            raise ValueError(f"{name}={voltage} is not aligned to the DV grid")
        return idx

    idx_tail_hi = aligned_grid_index(cfg.VSEG_LOW, "VSEG_LOW")
    idx_upp_hi = aligned_grid_index(cfg.VSEG_UPP, "VSEG_UPP")

    n_upstream_points = idx_upp_hi - idx_tail_hi
    if n_upstream_points <= 0:
        raise ValueError("VSEG_UPP must be greater than VSEG_LOW")
    if n_upstream_points % max_segs != 0:
        raise ValueError(
            "The upstream grid-point count must be divisible by MAX_SEGS"
        )

    width = n_upstream_points // max_segs
    start = idx_tail_hi + 1
    win = [
        (start + segment_idx * width, start + (segment_idx + 1) * width)
        for segment_idx in range(max_segs)
    ]

    if win[-1][1] - 1 != idx_upp_hi:
        raise RuntimeError("Upstream windows do not include VSEG_UPP")

    return win, idx_tail_hi, grid

def resample_1d(x: np.ndarray, target_len: int) -> np.ndarray:
    if len(x) == target_len:
        return x.astype(np.float32)
    xp = np.linspace(0, 1, len(x))
    xq = np.linspace(0, 1, target_len)
    return np.interp(xq, xp, x).astype(np.float32)

def make_features(Q_seg: np.ndarray, IC_seg: np.ndarray) -> np.ndarray:
    """Stack Q and stored right-endpoint IC values on one voltage grid."""

    q = np.asarray(Q_seg, dtype=np.float32)
    ic = np.asarray(IC_seg, dtype=np.float32)
    if q.ndim != 1 or ic.ndim != 1 or q.shape != ic.shape:
        raise ValueError("Aligned Q and IC features must be equal-length vectors")
    if not np.isfinite(q).all() or not np.isfinite(ic).all():
        raise ValueError("Aligned Q and IC features must be finite")
    return np.stack([q, ic], axis=0)

def build_subset(
    data,
    idx_array: np.ndarray,
    cfg,
    win,
    idx_tail_hi,
    role: str,
):
    """Build role-aware tail and variable-fragment inputs from ReBATT data."""

    idx_array = np.asarray(idx_array, dtype=np.int64)
    view = role_view(
        data, role, idx_array,
        vseg_low=cfg.VSEG_LOW,
        voltage_removal_window=cfg.VOLTAGE_REMOVAL_WINDOW,
    )
    X_tail, S_list, y_cap = [], [], []
    for local_i, global_i in enumerate(idx_array):
        V_i = view.v[:, local_i]
        Q_i = view.q[:, local_i]
        IC_i = view.ic[:, local_i]
        if view.target is not None:
            y_cap.append(np.array([view.target[local_i]], dtype=np.float32))
        
        V_tail = V_i[1:idx_tail_hi+1]
        Q_tail = Q_i[1:idx_tail_hi+1]
        IC_tail = IC_i[:idx_tail_hi]
        if (
            not np.isfinite(V_tail).all()
            or not np.isfinite(Q_tail).all()
            or not np.isfinite(IC_tail).all()
        ):
            raise ValueError(
                f"The configured Q/IC tail is incomplete for {role} global column "
                f"{int(global_i)}"
            )
        if not np.allclose(
            V_tail,
            data.voltage_grid[1:idx_tail_hi+1],
            rtol=0.0,
            atol=float(cfg.DV) * 1e-6,
        ):
            raise ValueError(
                f"Tail voltage coordinates differ from the shared grid for {role} "
                f"global column {int(global_i)}"
            )
        Q_tail_rs = resample_1d(Q_tail, cfg.TAIL_LEN)
        IC_tail_rs = resample_1d(IC_tail, cfg.TAIL_LEN)
        X_tail.append(make_features(Q_tail_rs, IC_tail_rs))
        
        segs = []
        for (a, b) in win:
            q_seg = Q_i[a:b]
            v_seg = V_i[a:b]
            ic_seg = IC_i[a-1:b-1]
            if not np.isfinite(q_seg).all():
                break
            if not np.isfinite(v_seg).all() or not np.isfinite(ic_seg).all():
                raise ValueError(
                    f"Aligned Q/V/IC fragment is inconsistent for {role} global column "
                    f"{int(global_i)}"
                )
            if not np.allclose(
                v_seg,
                data.voltage_grid[a:b],
                rtol=0.0,
                atol=float(cfg.DV) * 1e-6,
            ):
                raise ValueError(
                    f"Fragment voltage coordinates differ from the shared grid for "
                    f"{role} global column {int(global_i)}"
                )
            Q_rs = resample_1d(q_seg, cfg.SEG_LEN)
            IC_rs = resample_1d(ic_seg, cfg.SEG_LEN)
            segs.append(make_features(Q_rs, IC_rs))
        S_list.append(segs)
        
    if X_tail:
        X_tail = np.stack(X_tail, axis=0).astype(np.float32)
    else:
        X_tail = np.empty((0, 2, cfg.TAIL_LEN), dtype=np.float32)
    if view.target is None:
        y_cap_out = None
    elif y_cap:
        y_cap_out = np.stack(y_cap, axis=0).astype(np.float32)
    else:
        y_cap_out = np.empty((0, 1), dtype=np.float32)
    z_meta = np.zeros((len(idx_array), cfg.model.get("d_meta", 16)), dtype=np.float32)
    return dict(
        X_tail=X_tail,
        S_list=S_list,
        z_meta=z_meta,
        y_cap=y_cap_out,
        global_indices=idx_array.copy(),
        role=role,
    )

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
    if y_cap is None:
        return None
    denom = max(ncap["max"] - ncap["min"], 1e-8)
    return ((y_cap - ncap["min"]) / denom).astype(np.float32)

def inverse_cap(y_norm: np.ndarray, ncap):
    return y_norm * (ncap["max"] - ncap["min"]) + ncap["min"]

