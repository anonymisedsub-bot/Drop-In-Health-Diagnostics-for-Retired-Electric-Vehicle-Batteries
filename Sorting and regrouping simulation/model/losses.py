# losses.py
import torch
import numpy as np

def pinball_loss(pred, target, quantiles):
    """
    Computes the Quantile (Pinball) Loss for probabilistic forecasting.
    """
    loss = 0.0
    for i, q in enumerate(quantiles):
        e = target - pred[..., i]
        loss_q = torch.maximum(q * e, (q - 1) * e).mean()
        loss = loss + loss_q
    return loss

def mape(y_true, y_pred, eps=1e-8):
    """Mean Absolute Percentage Error."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), eps, None)))) * 100

def rmspe(y_true, y_pred, eps=1e-8):
    """Root Mean Square Percentage Error."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.sqrt(np.mean(((y_true - y_pred) / np.clip(np.abs(y_true), eps, None))**2))) * 100

def r2_score(y_true, y_pred):
    """R-squared (Coefficient of Determination)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - y_true.mean())**2)
    return float(1.0 - ss_res / max(ss_tot, 1e-8))