"""Loss and regression metric utilities."""
import torch
import numpy as np

def gaussian_kernel(x, y, gamma=None):
    """Computes the Gaussian kernel matrix K(x, y) for MMD."""
    D = torch.cdist(x, y, p=2).pow(2)
    if gamma is None:
        gamma = 1.0 / (D.mean() + 1e-8)
    return torch.exp(-gamma * D)

def mmd_loss(x, y, gamma=None):
    """Computes the Maximum Mean Discrepancy (MMD) loss."""
    K_xx = gaussian_kernel(x, x, gamma).mean()
    K_yy = gaussian_kernel(y, y, gamma).mean()
    K_xy = gaussian_kernel(x, y, gamma).mean()
    return K_xx + K_yy - 2 * K_xy

def coral_loss(x, y):
    """Computes the Correlation Alignment (CORAL) loss."""
    def covariance_matrix(A):
        N = A.size(0)
        A = A - A.mean(0, keepdim=True)
        return (A.transpose(0, 1) @ A) / (N - 1)

    cov_x = covariance_matrix(x)
    cov_y = covariance_matrix(y)
    return (cov_x - cov_y).pow(2).sum()

def _w(eff_sw: dict, sw: dict, key: str, default: float) -> float:
    """Safely retrieves the current dynamic or static weight for a given loss."""
    return float(eff_sw.get(key, sw.get(key, default)))

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

class DynamicWeighter:
    """
    Dynamic Weight Balancer based on Exponential Weighted Moving Average (EWMA).
    Maintains balance across specified loss terms.
    """
    def __init__(self, cfg_module):
        self.cfg = cfg_module
        lw_cfg = getattr(self.cfg, "loss_weights", {})
        self.mode = lw_cfg.get("mode", "fixed")
        self.enabled = set(lw_cfg.get("enabled", []))
        self.dyn = lw_cfg.get("dynamic", {})

        self.ema = {k: None for k in ["recon", "cap_mse", "cap_pin", "da"]}
        self.beta = float(self.dyn.get("ema_beta", 0.9))
        self.eps = float(self.dyn.get("eps", 1e-8))
        self.floor = float(self.dyn.get("floor", 0.05))
        self.ceil = float(self.dyn.get("ceil", 10.0))
        self.normalise = bool(self.dyn.get("normalise", True))

        base = self.dyn.get("base", {})
        self.base = {
            "recon": float(base.get("recon", 1.0)),
            "cap_mse": float(base.get("cap_mse", 1.0)),
            "cap_pin": float(base.get("cap_pin", 1.0)),
            "da": float(base.get("da", 0.2)),
        }

    def get_effective_weights(self):
        if self.mode != "dynamic":
            return {k: self.base[k] if k in self.enabled else 0.0 for k in self.base}

        ema_vals = {k: self.ema[k] for k in self.ema if self.ema[k] is not None}
        active = [k for k in ema_vals if k in self.enabled]

        if not active:
            return {k: self.base[k] if k in self.enabled else 0.0 for k in self.base}

        mean_ema = np.mean([ema_vals[k] for k in active])
        multipliers = {}
        for k in self.base.keys():
            if k in active:
                m = self.base[k] * (mean_ema / (ema_vals[k] + self.eps))
                m = float(np.clip(m, self.floor, self.ceil))
            else:
                m = self.base[k] if k in self.enabled else 0.0
            multipliers[k] = m

        if self.normalise:
            base_sum = np.sum([self.base[k] for k in self.enabled]) + self.eps
            dyn_sum = np.sum([multipliers[k] for k in self.enabled]) + self.eps
            scale = base_sum / dyn_sum
            for k in self.enabled:
                multipliers[k] *= scale

        return multipliers

    def update_ema(self, observed_losses: dict):
        for k in self.ema.keys():
            val = float(observed_losses.get(k, 0.0))
            if self.ema[k] is None:
                self.ema[k] = val
            else:
                self.ema[k] = self.beta * self.ema[k] + (1 - self.beta) * val
