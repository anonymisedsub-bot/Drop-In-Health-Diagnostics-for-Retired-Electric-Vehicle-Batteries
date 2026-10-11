# utils.py

import os, json, random
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
from sklearn.metrics import r2_score

# ==================================================================
# 1. Basic utilities (seeding, IO, JSON)
# ==================================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def clone_state_dict(model):
    """Return an independent CPU snapshot of a model's parameters and buffers."""
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def ensure_dir(p):
    """
    Create directory if not exists. Accepts str or Path.
    Returns str path for convenience.
    """
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return str(p)

def ensure_parent(fp):
    """
    Ensure parent directory of a file path exists. Accepts str or Path.
    """
    fp = Path(fp)
    fp.parent.mkdir(parents=True, exist_ok=True)
    return str(fp)

def _to_serializable(obj):
    """Safely convert common objects that JSON cannot serialize directly."""
    # NumPy scalars
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return obj.item()
    # NumPy arrays
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    # PyTorch tensors
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    # pathlib.Path
    if isinstance(obj, Path):
        return str(obj)
    # Sets
    if isinstance(obj, set):
        return list(obj)
    # Fall back to strings for other types; avoid converting large objects.
    return str(obj)

def save_json(obj, path):
    """
    Safe JSON dump that ensures parent directory exists.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

# ==================================================================
# 2. NumPy and array-shape utilities
# ==================================================================

def to_numpy(x):
    if isinstance(x, np.ndarray): return x
    return np.asarray(x)

def flatten_channel_time(X):  # (N, C, T) -> (N, C*T) for sklearn models
    N, C, T = X.shape
    return X.reshape(N, C*T)

def denorm_y(Yn, norm):
    """
    Reverse the normalization of Y.
    Yn may be an (N, 1) or (N, 3) array, or a (mean, std) tuple.
    """
    ymin = np.array(norm["ymin"]).reshape(1,1)
    ymax = np.array(norm["ymax"]).reshape(1,1)
    scale = ymax - ymin
    
    if isinstance(Yn, tuple):
        # RF (mean, std) tuple
        mean_n, std_n = Yn[0], Yn[1]
        mean_den = mean_n * scale + ymin
        # Standard deviations change with scale, but not with the offset.
        std_den_val = scale.ravel()
        if std_den_val.size > 0:
             std_den_val = std_den_val[0]
        else:
             std_den_val = 1.0 # Fall back when the scale array is empty.
        std_den = std_n * std_den_val
        return (mean_den, std_den)
    else:
        # (N, 1) or (N, 3) array
        return Yn * scale + ymin


def denorm_y_std(std_n, norm):
    """Undo target scaling for a standard deviation without adding an offset."""
    ymin = np.asarray(norm["ymin"], dtype=float)
    ymax = np.asarray(norm["ymax"], dtype=float)
    scale = float((ymax - ymin).reshape(-1)[0])
    return np.asarray(std_n, dtype=float) * scale

# ==================================================================
# 3. Evaluation metrics (NumPy)
# ==================================================================

def mape(y_true, y_pred, eps=1e-8):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    return float(np.mean(np.abs((y_true - y_pred) / (np.clip(np.abs(y_true), eps, None)))))

def rmspe(y_true, y_pred, eps=1e-8):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    return float(np.sqrt(np.mean(np.square((y_true - y_pred) / (np.clip(np.abs(y_true), eps, None))))))

def r2(y_true, y_pred):
    return float(r2_score(y_true, y_pred))

def smape(y_true, y_pred, eps=1e-8):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    return float(np.mean(2.0 * np.abs(y_pred - y_true) /
                         np.clip((np.abs(y_true) + np.abs(y_pred)), eps, None)))

def wmape(y_true, y_pred, eps=1e-8):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    num = np.sum(np.abs(y_pred - y_true))
    den = np.clip(np.sum(np.abs(y_true)), eps, None)
    return float(num / den)

def safe_mape(y_true, y_pred, floor_frac=0.1, eps=1e-8):
    """
    Floor the denominator at max(|y|, floor_frac * mean(|y|)) to limit small-denominator inflation.
    """
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    scale = np.maximum(np.abs(y_true), floor_frac * np.mean(np.abs(y_true)))
    scale = np.clip(scale, eps, None)
    return float(np.mean(np.abs((y_pred - y_true) / scale)))

def pinball_loss(y_true, y_pred, q):
    """NumPy pinball loss for evaluation."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    error = y_true - y_pred
    return float(np.mean(np.maximum(q * error, (q - 1) * error)))


# ==================================================================
# 4. Metric aggregation (NumPy)
# ==================================================================

def _summary_metrics_single(y_true, y_pred_single):
    """Compute standard metrics for point predictions (mean or median)."""
    y_t = np.asarray(y_true).ravel()
    y_p = np.asarray(y_pred_single).ravel()
    return {
        "MAE": float(np.mean(np.abs(y_p - y_t))),
        "RMSE": float(np.sqrt(np.mean((y_p - y_t)**2))),
        "MAPE": mape(y_t, y_p),
        "RMSPE": rmspe(y_t, y_p),
        "sMAPE": smape(y_t, y_p),
        "wMAPE": wmape(y_t, y_p),
        "R2": r2(y_t, y_p),
    }

def summary_metrics(y_true, y_pred, quantiles=[0.5]):
    """
    Compute metrics for point or quantile predictions.
    y_true: (N, 1) reference values
    y_pred: (N, 1) means, (N, 3) quantiles, or a (mean, std) tuple
    quantiles: list, e.g., [0.1, 0.5, 0.9]
    """
    y_true_flat = np.asarray(y_true).ravel()
    
    # Case 1: RF prediction (mean, std) tuple
    if isinstance(y_pred, tuple):
        y_mean = y_pred[0].ravel()
        y_std = y_pred[1].ravel()
        
        # Compute standard metrics from the mean predictions.
        metrics = _summary_metrics_single(y_true_flat, y_mean)
        
        # Report the mean predicted standard deviation.
        metrics["Predicted_Std_Mean"] = float(np.mean(y_std))
        return metrics

    # Case 2: array predictions (means or quantiles)
    y_pred_arr = np.asarray(y_pred)
    
    # Locate quantile indices.
    q_indices = {}
    for q in [0.1, 0.5, 0.9]:
        if q in quantiles:
            try: q_indices[q] = quantiles.index(q)
            except ValueError: q_indices[q] = -1
        else:
            q_indices[q] = -1
    
    q10_idx, q50_idx, q90_idx = q_indices.get(0.1, -1), q_indices.get(0.5, -1), q_indices.get(0.9, -1)

    # Case 2a: quantile predictions, shaped (N, 3) or similarly
    if y_pred_arr.ndim == 2 and y_pred_arr.shape[1] == len(quantiles) and q50_idx != -1:
        metrics = {}
        
        # Compute metrics for q50 (the median).
        y_pred_median = y_pred_arr[:, q50_idx]
        metrics_q50 = _summary_metrics_single(y_true_flat, y_pred_median)
        for k, v in metrics_q50.items():
            metrics[f"{k}_q50"] = v
            
        # Compute metrics for q10.
        if q10_idx != -1:
            y_pred_q10 = y_pred_arr[:, q10_idx]
            metrics_q10 = _summary_metrics_single(y_true_flat, y_pred_q10)
            for k, v in metrics_q10.items():
                metrics[f"{k}_q10"] = v

        # Compute metrics for q90.
        if q90_idx != -1:
            y_pred_q90 = y_pred_arr[:, q90_idx]
            metrics_q90 = _summary_metrics_single(y_true_flat, y_pred_q90)
            for k, v in metrics_q90.items():
                metrics[f"{k}_q90"] = v
        
        # Compute pinball loss across all quantiles.
        for i, q in enumerate(quantiles):
            q_pred = y_pred_arr[:, i]
            q_key = f"Pinball_q{int(q*100)}"
            metrics[q_key] = pinball_loss(y_true_flat, q_pred, q)
            
        # Compute the derived standard deviation.
        if q10_idx != -1 and q90_idx != -1:
            q10 = y_pred_arr[:, q10_idx]
            q90 = y_pred_arr[:, q90_idx]
            # Z-score for 0.9 is 1.2815, so range (q90-q10) covers 2 * 1.2815 = 2.563
            std_derived = (q90 - q10) / 2.56308 
            metrics["Derived_Std_Mean"] = float(np.mean(std_derived))
            
        return metrics

    # Case 2b: standard mean predictions, shaped (N, 1) or (N,)
    else:
        return _summary_metrics_single(y_true_flat, y_pred_arr)


# ==================================================================
# 5. PyTorch loss functions
# ==================================================================

class QuantileLoss(nn.Module):
    """
    PyTorch pinball loss for training.
    Assumes y_pred is (N, num_quantiles) and y_true is (N, 1).
    """
    def __init__(self, quantiles=[0.1, 0.5, 0.9]):
        super().__init__()
        self.quantiles = torch.tensor(quantiles, dtype=torch.float32).view(1, -1)

    def forward(self, y_pred, y_true):
        if self.quantiles.device != y_pred.device:
            self.quantiles = self.quantiles.to(y_pred.device)
        
        # Expand targets to match predictions: (N, 1) -> (N, num_quantiles).
        y_true_expanded = y_true.expand(-1, self.quantiles.shape[1])
        
        errors = y_true_expanded - y_pred
        
        # Compute the loss for each quantile.
        # max(q * error, (q - 1) * error)
        loss = torch.max(
            (self.quantiles - 1) * errors,
            self.quantiles * errors
        )
        
        # Average across all samples and quantiles.
        return loss.mean()
