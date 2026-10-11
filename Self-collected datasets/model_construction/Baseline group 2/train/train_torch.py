# train/train_torch.py
import numpy as np, torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from utils import clone_state_dict, summary_metrics, denorm_y
import config as cfg

class QuantileLoss(nn.Module):
    """MSE for the q50 point-output slot; pinball loss for other quantiles."""
    def __init__(self, quantiles=[0.1, 0.5, 0.9], weights=None):
        super().__init__()
        self.quantiles = quantiles
        self.num_quantiles = len(quantiles)
        
        # Default q50 point-MSE weight is 1.0; outer pinball weights are 0.2.
        if weights is None:
            weights = [0.2 if q != 0.5 else 1.0 for q in quantiles]
        self.weights = weights
        
    def forward(self, predictions, targets):
        """
        predictions: (batch_size, num_quantiles)
        targets: (batch_size, 1)
        """
        losses = []
        for i, q in enumerate(self.quantiles):
            errors = targets - predictions[:, i:i+1]
            if q == 0.5:
                loss = errors.square().mean()
            else:
                loss = torch.mean(torch.max((q-1) * errors, q * errors))
            weighted_loss = loss * self.weights[i]
            losses.append(weighted_loss)
        
        return sum(losses)

class TensorDatasetTS(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

def build_model(name, seq_len, device, output_dim):
    """Build from explicit public settings after checking protocol dimensions."""
    from models.mlp import MLP
    from models.cnn import CNN1D
    from models.lstm import LSTMReg
    from models.transformer import TinyTransformer
    constructors = {
        "mlp": (MLP, cfg.MLP_CONFIG),
        "cnn": (CNN1D, cfg.CNN1D_CONFIG),
        "lstm": (LSTMReg, cfg.LSTM_CONFIG),
        "transformer": (TinyTransformer, cfg.TRANSFORMER_CONFIG),
    }
    if name not in constructors:
        raise ValueError(f"Unknown DL model {name}")
    constructor, settings = constructors[name]
    settings = dict(settings)
    for key in ('seq_len', 'max_seq_len'):
        if key in settings and settings[key] != seq_len:
            raise ValueError(f"{name} {key} must match the input length {seq_len}")
    if settings['in_ch'] != 1 or settings['output_dim'] != output_dim:
        raise ValueError(f"{name} in_ch/output_dim must match the scalar capacity protocol")
    if list(settings['quantiles']) != list(cfg.QUANTILES):
        raise ValueError(f"{name} quantiles must match QUANTILES")
    return constructor(config=settings).to(device)

# ---------- Convert model outputs to the regression tensor yhat. ----------
def _as_yhat(pred):
    """
    Accept direct yhat outputs and autoencoder-style (recon, yhat) outputs.
    """
    if isinstance(pred, (tuple, list)):
        # Convention: pred = (recon, yhat)
        return pred[1]
    return pred

# ---------- Early-stopping comparison ----------
def _is_better(curr, best, mode="min", min_delta=0.0):
    if best is None:
        return True
    if mode == "min":
        return (best - curr) > min_delta
    else:
        return (curr - best) > min_delta

def _apply_step_decay_if_needed(optimizer, cfg, epoch_idx):
    """
    (MOD-10) 
    For the "plateau" policy, also apply staircase decay using epoch_step and step_gamma.
    """
    # Ensure cfg.scheduler exists.
    if not hasattr(cfg, "scheduler"):
        return
        
    sch_cfg = cfg.scheduler
    if sch_cfg.get("policy", "plateau") != "plateau":
        return
        
    step_every = int(sch_cfg.get("epoch_step", 20))
    gamma      = float(sch_cfg.get("step_gamma", 0.8))
    min_lr     = float(sch_cfg.get("min_lr", 0.0))
    
    if step_every > 0 and ((epoch_idx + 1) % step_every == 0):
        print(f"  [LR Step Decay] Applying step decay gamma={gamma} at epoch {epoch_idx+1}")
        for pg in optimizer.param_groups:
            pg["lr"] = max(pg["lr"] * gamma, min_lr)

def train_val_one_model(name, Xtr, Ytr, Xval, Yval, cfg):
    device = cfg.device if torch.cuda.is_available() and cfg.device=="cuda" else "cpu"
    
    # Get quantile settings from config
    quantiles = getattr(cfg, "QUANTILES", [0.1, 0.5, 0.9])
    output_dim = 1  # Output dimension per quantile
    
    model = build_model(name, seq_len=Xtr.shape[-1], device=device, output_dim=output_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    pol = cfg.scheduler.get("policy", "plateau")
    
    # Make lr_min available on cfg for compatibility with older configs.
    lr_min_val = getattr(cfg, "lr_min", 1e-10)    
    if pol == "plateau":
        print(f"[{name}] Using Scheduler: ReduceLROnPlateau (Hybrid Step)")
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min",
            factor=cfg.scheduler.get("factor", 0.5),
            patience=cfg.scheduler.get("patience", 5),
            min_lr=cfg.scheduler.get("min_lr", lr_min_val),
            verbose=False
        )
    elif pol == "cosine_epoch":
        print(f"[{name}] Using Scheduler: CosineAnnealingLR")
        # Ensure T_max exists.
        T_max = getattr(cfg, "cosine_T_max", cfg.num_epochs) 
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=lr_min_val)
    elif pol == "exp":
        print(f"[{name}] Using Scheduler: ExponentialLR")
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    else:
        print(f"[{name}] Using no LR scheduler.")
        scheduler = None

    # Use q50 MSE for capacity supervision, preserving output slots and weights.
    loss_weights = [0.0 if q != 0.5 else 1.0 for q in quantiles]  # q50 MSE only; outer heads disabled.
    loss_fn = QuantileLoss(quantiles=quantiles, weights=loss_weights)

    ds_tr = TensorDatasetTS(Xtr, Ytr)
    ds_val= TensorDatasetTS(Xval, Yval)
    dl_tr = DataLoader(ds_tr, batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    dl_val= DataLoader(ds_val, batch_size=cfg.batch_size, shuffle=False, drop_last=False)

    # ----- Early-stopping settings -----
    es_enable    = getattr(cfg, "es_enable", True)
    # Validation and early stopping always use q50 point-prediction MSE.
    es_mode      = getattr(cfg, "es_mode", "min")
    es_patience  = getattr(cfg, "es_patience", 20)
    es_min_delta = getattr(cfg, "es_min_delta", 0.0)

    best = {"val_metric": None, "state": None, "epoch": 0}
    no_improve = 0

    for epoch in range(1, cfg.num_epochs+1):
        model.train()
        tr_losses = []
        for xb, yb in dl_tr:
            xb, yb = xb.to(device), yb.to(device)
            pred_raw = model(xb)               # Standard model: (N, Q); AE: (recon, yhat)
            yhat = _as_yhat(pred_raw)          # Extract yhat with shape (N, num_quantiles).

            # Main capacity loss: q50 MSE; retain weighted pinball loss for other quantiles.
            loss_ql = loss_fn(yhat, yb)        # yb is (N, 1)

            loss = loss_ql

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tr_losses.append(loss.item())

        # Validation uses MSE at the retained q50 point-output slot.
        model.eval()
        with torch.no_grad():
            vlosses = []
            median_idx = quantiles.index(0.5) if 0.5 in quantiles else len(quantiles)//2
            
            for xb, yb in dl_val:
                xb, yb = xb.to(device), yb.to(device)
                pr_raw = model(xb)  # (N, num_quantiles) or (recon, yhat)
                pr = _as_yhat(pr_raw)
                
                # Validate only the median output slot, without regularization.
                pr_median = pr[:, median_idx:median_idx+1]  # (N, 1)
                median_loss = nn.MSELoss()(pr_median, yb)    # Use MSE for validation.
                vlosses.append(median_loss.item())
        
        vloss = float(np.mean(vlosses)) if len(vlosses)>0 else float('inf')
        trloss= float(np.mean(tr_losses)) if len(tr_losses)>0 else float('inf')
        
        # Check for NaN losses.
        if np.isnan(trloss) or np.isnan(vloss):
            print(f"[{name}] Error: NaN loss at epoch {epoch}. Stopping early.")
            break

        # 1. Apply staircase decay when policy == "plateau".
        _apply_step_decay_if_needed(optimizer, cfg, epoch)

        # 2. Apply the main scheduler.
        if scheduler is not None:
            if pol == "plateau":
                scheduler.step(vloss) # Plateau scheduling requires the validation loss.
            elif pol in ["cosine_epoch", "exp"]:
                scheduler.step() # Cosine/exponential scheduling does not require validation loss.

        # CapacityLoss includes the supervised objective and any enabled regularization.
        print(f"[{name}] Epoch {epoch:03d} | train CapacityLoss {trloss:.4f} | val PointMSE {vloss:.4f}")

        # Monitor validation MSE for the q50 point prediction.
        curr_metric = vloss

        if _is_better(curr_metric, best["val_metric"], mode=es_mode, min_delta=es_min_delta):
            best["val_metric"] = curr_metric
            best["state"] = clone_state_dict(model)
            best["epoch"] = epoch
            no_improve = 0
        else:
            no_improve += 1
            if es_enable and (no_improve >= es_patience):
                print(f"[{name}] Early stopping at epoch {epoch} "
                      f"(best @ {best['epoch']} with val_mse={best['val_metric']:.6f}).")
                break

    # Restore the best weights.
    if best["state"] is not None:
        model.load_state_dict(best["state"])
    return model


def predict(model, X, cfg):
    device = cfg.device if torch.cuda.is_available() and cfg.device=="cuda" else "cpu"
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32).to(device)
        Yp_raw = model(X_t)                    # Output shape: (N, num_quantiles)
        Yp = _as_yhat(Yp_raw).cpu().numpy()    # Extract yhat.
    return Yp

def eval_model(model, X, Y, cfg, norm=None, clip01=False, name_tag="test"):
    """
    Evaluate an individual dataset or return predictions without labels.
    
    Args:
        model: Trained model.
        X: Input data with shape (N, 1, T).
        Y: Targets with shape (N, 1), or None for predictions only.
        cfg: Configuration object.
        norm: Normalization parameters.
        clip01: Whether to clip to [0, 1].
        name_tag: Dataset label: 'train', 'val', 'test', or 'pool'.
    
    Returns:
        dict: Predictions and evaluation metrics.
    """
    out = {}
    quantiles = getattr(cfg, "QUANTILES", [0.1, 0.5, 0.9])
    
    def _pred(X_data):
        """Internal prediction helper."""
        yp = predict(model, X_data, cfg)  # yp is (N, Q)
        if clip01:
            yp = np.clip(yp, 0.0, 1.0)
        return yp

    # Validate the input data.
    if X is None or X.shape[0] == 0:
        out[name_tag] = {}
        return out

    # Generate predictions.
    yp_n = _pred(X)  # (N, Q): predictions for all quantile slots
    
    # Prepare the output.
    out[name_tag] = {}
    
    if norm is not None:
        # Reverse prediction normalization.
        YP_den = denorm_y(yp_n, norm)  # (N, Q)
        out[name_tag]["y_pred_den"] = YP_den
        out[name_tag]["pred"] = yp_n  # Normalized predictions
        
        if Y is not None and Y.shape[0] > 0:
            # Ground-truth labels are available; perform full evaluation.
            Y_den = denorm_y(Y, norm)  # (N, 1)
            
            # Evaluation selects the retained q50 point-output slot.
            median_idx = quantiles.index(0.5) if 0.5 in quantiles else len(quantiles)//2
            yp_median = YP_den[:, median_idx:median_idx+1]  # (N, 1)
            
            # Compute evaluation metrics.
            metrics = summary_metrics(Y_den, yp_median, quantiles=[0.5])
            out[name_tag].update(metrics)
            out[name_tag]["y_true_den"] = Y_den
        else:
            # Predictions only; no ground-truth labels.
            out[name_tag]["y_true_den"] = None
    else:
        # No normalization parameters supplied.
        out[name_tag]["y_pred_den"] = yp_n
        out[name_tag]["pred"] = yp_n
        
        if Y is not None and Y.shape[0] > 0:
            # Ground-truth labels are available; perform full evaluation.
            median_idx = quantiles.index(0.5) if 0.5 in quantiles else len(quantiles)//2
            yp_median = yp_n[:, median_idx:median_idx+1]  # (N, 1)
            
            # Compute evaluation metrics.
            metrics = summary_metrics(Y, yp_median, quantiles=[0.5])
            out[name_tag].update(metrics)
            out[name_tag]["y_true_den"] = Y
        else:
            # Predictions only; no ground-truth labels.
            out[name_tag]["y_true_den"] = None

    return out

# Legacy interface retained for backward compatibility
def eval_model_full(model, Xtr, Ytr, Xval, Yval, Xte, Yte, cfg, norm=None, clip01=False):
    """
    Evaluate all datasets through the backward-compatible interface.
    
    Returns:
        dict: Evaluation results for the train, validation, and test datasets.
    """
    full_out = {}
    
    # Evaluate each dataset separately.
    if Xtr is not None and Xtr.shape[0] > 0:
        tr_out = eval_model(model, Xtr, Ytr, cfg, norm, clip01, "train")
        full_out.update(tr_out)
    
    if Xval is not None and Xval.shape[0] > 0:
        val_out = eval_model(model, Xval, Yval, cfg, norm, clip01, "val")
        full_out.update(val_out)
    
    if Xte is not None and Xte.shape[0] > 0:
        test_out = eval_model(model, Xte, Yte, cfg, norm, clip01, "test")
        full_out.update(test_out)
    
    return full_out
