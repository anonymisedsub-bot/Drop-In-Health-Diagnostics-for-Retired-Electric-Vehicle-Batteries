# train/train_torch.py
import numpy as np, torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from utils import summary_metrics, denorm_y, clone_state_dict
import config as cfg

import random
from data import get_voltage_config, get_curriculum_keep, deterministic_mask_by_k

class CLMaskingDataset(Dataset):
    """
    Dataset for curriculum learning.
    Apply masks dynamically in __getitem__.
    """
    def __init__(self, X_full, Y_full, cl_config, voltage_cfg):
        self.X_full = torch.tensor(X_full, dtype=torch.float32)
        self.Y_full = torch.tensor(Y_full, dtype=torch.float32)
        self.cl_config = cl_config
        self.v_cfg = voltage_cfg
        
        self.N = X_full.shape[0]
        self.base_avail = list(range(self.v_cfg["MAX_SEGS"])) # [0, 1, ..., 9]
        self.k_curr = self.v_cfg["MAX_SEGS"]
        self.p_drop_all = self.cl_config["p_drop_all"]
        self.p_keep_one = self.cl_config["p_keep_one"]
        self.set_epoch(0) # Initialize k_curr.

    def set_epoch(self, epoch):
        """Update the curriculum's current k from the training loop."""
        self.k_curr = get_curriculum_keep(epoch, self.cl_config)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        x_full = self.X_full[idx] # (1, segment_length)
        y = self.Y_full[idx]      # (1,)
        
        # --- Apply the curriculum mask. ---
        k_available = min(len(self.base_avail), self.k_curr)
        
        if k_available == 0:
            k_used = 0
        elif random.random() < self.p_drop_all:
            k_used = 0
        elif random.random() < self.p_keep_one:
            k_used = random.randint(1, max(1, k_available)) # [1, k_available]
        else:
            # Use linearly weighted sampling, favoring smaller k.
            population = list(range(k_available + 1))
            weights    = [k_available - k + 1 for k in population]
            k_used     = random.choices(population, weights=weights, k=1)[0]
            
        if k_used == k_available and k_available == self.v_cfg["MAX_SEGS"]:
            # Return directly when all segments are used.
            return x_full, y

        # Randomly choose k_used segments.
        # chosen_indices = random.sample(self.base_avail, k_used)
        chosen_indices = np.arange(k_used)
        
        # Create the mask.
        # Clone x_full to avoid modifying the original data.
        x_masked = x_full.clone()
        
        seg_mask = torch.zeros(self.v_cfg["MAX_SEGS"], dtype=torch.bool)
        if list(chosen_indices):
            seg_mask[chosen_indices] = True
            
        # Expand each segment mask across its configured sample positions.
        upstream_mask = torch.repeat_interleave(seg_mask, self.v_cfg["SUBSEG_LEN"]) # (UPSTREAM_LEN,)
        
        # Find upstream positions to zero out.
        indices_to_zero_upstream = torch.where(~upstream_mask)[0]
        if indices_to_zero_upstream.numel() > 0:
            # Add the upstream offset.
            indices_to_zero_global = indices_to_zero_upstream + self.v_cfg["UPSTREAM_START_IDX"]
            x_masked[0, indices_to_zero_global] = 0.0
            
        return x_masked, y
    
class QuantileLoss(nn.Module):
    """Use MSE for q50 point predictions and pinball loss for other quantiles."""
    def __init__(self, quantiles=[0.1, 0.5, 0.9], weights=None):
        super().__init__()
        self.quantiles = quantiles
        self.num_quantiles = len(quantiles)
        
        # Train only the median head; retain outer heads with zero loss weight.
        if weights is None:
            weights = [0.0 if q != 0.5 else 1.0 for q in quantiles]
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
                loss = torch.mean(errors.square())
            else:
                loss = torch.mean(torch.max((q - 1) * errors, q * errors))
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
    # Import models from models package
    from models.mlp import MLP
    from models.cnn import CNN1D

    constructors = {"mlp": (MLP, cfg.MLP_CONFIG), "cnn": (CNN1D, cfg.CNN1D_CONFIG)}
    if name not in constructors:
        raise ValueError(f"Unknown DL model {name}")
    constructor, settings = constructors[name]
    model_config = dict(settings)
    for key, expected in (("in_ch", 1), ("output_dim", output_dim), ("seq_len", seq_len)):
        if model_config[key] != expected:
            raise ValueError(f"{name} {key} must equal {expected} for the configured input and target")
    if list(model_config["quantiles"]) != list(cfg.QUANTILES):
        raise ValueError(f"{name} quantiles must match QUANTILES")
    return constructor(config=model_config).to(device)

# ---------- Early-stopping comparison utility ----------
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
    For the "plateau" policy, also apply epoch_step / step_gamma staircase decay.
    """
    # Read scheduler settings, using defaults if absent.
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

def train_val_one_model(
    name, Xtr, Ytr, Xval, Yval, cfg, cl_config=None,
    Xval_n_full=None, Yval_n_full=None, voltage_cfg=None # Optional full-validation context.
):
    device = cfg.device if torch.cuda.is_available() and cfg.device=="cuda" else "cpu"
    
    # Get quantile settings from config
    quantiles = getattr(cfg, "QUANTILES", [0.1, 0.5, 0.9])
    output_dim = 1  # Output width per quantile.
    
    model = build_model(name, seq_len=Xtr.shape[-1], device=device, output_dim=output_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    pol = cfg.scheduler.get("policy", "plateau")
    lr_min_val = getattr(cfg, "lr_min", 1e-10)

    if pol == "plateau":
        print(f"[{name}] Using Scheduler: ReduceLROnPlateau (Hybrid Step)")
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min",
            factor=cfg.scheduler.get("factor", 0.5),
            patience=cfg.scheduler.get("patience", 5),
            min_lr=cfg.scheduler.get("min_lr", lr_min_val),
        )
    elif pol == "cosine_epoch":
        print(f"[{name}] Using Scheduler: CosineAnnealingLR")
        # Default the cosine horizon to the configured epoch count.
        T_max = getattr(cfg, "cosine_T_max", cfg.num_epochs) 
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=lr_min_val)
    elif pol == "exp":
        print(f"[{name}] Using Scheduler: ExponentialLR")
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    else:
        print(f"[{name}] Using no LR scheduler.")
        scheduler = None

    # Train only the q50 point-prediction head, using MSE.
    loss_weights = [0.0 if q != 0.5 else 1.0 for q in quantiles]
    loss_fn = QuantileLoss(quantiles=quantiles, weights=loss_weights)

    # Use a standard Dataset for validation.
    ds_val = TensorDatasetTS(Xval, Yval)
    dl_val = DataLoader(ds_val, batch_size=cfg.batch_size, shuffle=False, drop_last=False)

    # Check whether curriculum learning is enabled for training.
    # cl_config is supplied only for neural models.
    if cl_config:
        print(f"[{name}] Dynamic curriculum learning (CL) enabled.")
        voltage_cfg = get_voltage_config()
        ds_tr = CLMaskingDataset(Xtr, Ytr, cl_config, voltage_cfg)
    else:
        # Use a standard Dataset when no curriculum configuration is supplied.
        # Tree-model inputs are already statically masked in the budget worker.
        print(f"[{name}] Using static training data (no dynamic CL).")
        ds_tr = TensorDatasetTS(Xtr, Ytr)
        
    dl_tr = DataLoader(ds_tr, batch_size=cfg.batch_size, shuffle=True, drop_last=False)

    # ----- Early-stopping configuration -----
    es_enable    = getattr(cfg, "es_enable", True)
    es_mode      = getattr(cfg, "es_mode", "min")
    es_patience  = getattr(cfg, "es_patience", 20)
    es_min_delta = getattr(cfg, "es_min_delta", 0.0)

    best = {"val_metric": None, "state": None, "epoch": 0}
    no_improve = 0

    for epoch in range(1, cfg.num_epochs+1):
        # --- Update the curriculum at each epoch. ---
        if isinstance(ds_tr, CLMaskingDataset):
            ds_tr.set_epoch(epoch)

        model.train()
        tr_losses = []
        for xb, yb in dl_tr:
            xb, yb = xb.to(device), yb.to(device)
            yhat = model(xb)

            # Main quantile loss
            loss = loss_fn(yhat, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tr_losses.append(loss.item())

        # Use only the median quantile for validation loss.
        model.eval()
        # Check the validation-sweep switch.
        use_sweep = getattr(cfg, "eval_sweep", {}).get("enable", False)

        if use_sweep and Xval_n_full is not None and Yval_n_full is not None and voltage_cfg is not None:
            # --- Mode 1: evaluation sweep, k=0 through MAX_SEGS. ---
            vlosses_k_weighted = [] # Store loss multiplied by weight.
            weight_k0 = getattr(cfg, "eval_sweep", {}).get("weight_k0", 1.0)
            
            with torch.no_grad():
                for k in range(voltage_cfg["MAX_SEGS"] + 1):
                    # 1. Create deterministically masked data for k.
                    Xk = deterministic_mask_by_k(Xval_n_full, k, voltage_cfg)
                    Yk = Yval_n_full
                    
                    # 2. Create a temporary data loader.
                    ds_k = TensorDatasetTS(Xk, Yk)
                    dl_k = DataLoader(ds_k, batch_size=cfg.batch_size, shuffle=False, drop_last=False)
                    
                    # 3. Compute q50 MSE for k.
                    losses_k = []
                    median_idx = quantiles.index(0.5) if 0.5 in quantiles else len(quantiles)//2
                    
                    for xb_k, yb_k in dl_k:
                        xb_k, yb_k = xb_k.to(device), yb_k.to(device)
                        pr = model(xb_k)
                        pr_median = pr[:, median_idx:median_idx+1]
                        median_loss = nn.MSELoss()(pr_median, yb_k)
                        losses_k.append(median_loss.item())
                    
                    avg_loss_k = float(np.mean(losses_k)) if losses_k else float('inf')
                    
                    # 4. Apply the weight.
                    weight = weight_k0 if k == 0 else 1.0
                    vlosses_k_weighted.append(avg_loss_k * weight)

            # 5. Compute the weighted mean validation loss.
            total_weight_factor = (voltage_cfg["MAX_SEGS"] * 1.0) + weight_k0
            vloss = float(np.sum(vlosses_k_weighted)) / max(1.0, total_weight_factor)

        else:
            # --- Mode 2: standard validation using the randomly masked dl_val data. ---
            with torch.no_grad():
                vlosses = []
                median_idx = quantiles.index(0.5) if 0.5 in quantiles else len(quantiles)//2
                
                for xb, yb in dl_val:
                    xb, yb = xb.to(device), yb.to(device)
                    pr = model(xb)
                    pr_median = pr[:, median_idx:median_idx+1]
                    median_loss = nn.MSELoss()(pr_median, yb)
                    vlosses.append(median_loss.item())

            vloss = float(np.mean(vlosses)) if len(vlosses)>0 else float('inf')

        trloss= float(np.mean(tr_losses)) if len(tr_losses)>0 else float('inf')
        
        # Check for NaN losses.
        if np.isnan(trloss) or np.isnan(vloss):
            print(f"[{name}] Error: loss became NaN at epoch {epoch}. Stopping early.")
            break

        # 1. Apply manual staircase decay for the plateau policy.
        _apply_step_decay_if_needed(optimizer, cfg, epoch)

        # 2. Apply the main scheduler.
        if scheduler is not None:
            if pol == "plateau":
                scheduler.step(vloss) # Plateau requires the validation loss.
            elif pol in ["cosine_epoch", "exp"]:
                scheduler.step() # Cosine and exponential schedules do not.

        print(f"[{name}] Epoch {epoch:03d} | train Loss {trloss:.4f} | val MSE {vloss:.4f}")

        # Monitor validation q50 MSE for early stopping.
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

    # Restore the best state.
    if best["state"] is not None:
        model.load_state_dict(best["state"])
    return model


def predict(model, X, cfg):
    device = cfg.device if torch.cuda.is_available() and cfg.device=="cuda" else "cpu"
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32).to(device)
        Yp = model(X_t).cpu().numpy()
    return Yp

def eval_model(model, X, Y, cfg, norm=None, clip01=True, name_tag="test"):
    """
    Evaluate an individual dataset, optionally without reference targets.
    
    Args:
        model: trained model
        X: input data (N, 1, T)
        Y: targets (N, 1), or None for prediction only
        cfg: configuration object
        norm: normalization parameters
        clip01: whether to clip to [0, 1]
        name_tag: dataset label: 'train', 'val', 'test', or 'pool'
    
    Returns:
        dict: predictions and evaluation metrics
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
    yp_n = _pred(X)  # (N, Q): predictions for all quantiles.
    
    # Prepare the output.
    out[name_tag] = {}
    
    if norm is not None:
        # Reverse prediction normalization.
        YP_den = denorm_y(yp_n, norm)  # (N, Q)
        out[name_tag]["y_pred_den"] = YP_den
        out[name_tag]["pred"] = yp_n  # Normalized predictions.
        
        if Y is not None and Y.shape[0] > 0:
            # Evaluate fully when reference targets are available.
            Y_den = denorm_y(Y, norm)  # (N, 1)
            
            # Use only the median quantile for evaluation.
            median_idx = quantiles.index(0.5) if 0.5 in quantiles else len(quantiles)//2
            yp_median = YP_den[:, median_idx:median_idx+1]  # (N, 1)
            
            # Compute evaluation metrics.
            metrics = summary_metrics(Y_den, yp_median, quantiles=[0.5])
            out[name_tag].update(metrics)
            out[name_tag]["y_true_den"] = Y_den
        else:
            # Predictions without reference targets
            out[name_tag]["y_true_den"] = None
    else:
        # Without normalization
        out[name_tag]["y_pred_den"] = yp_n
        out[name_tag]["pred"] = yp_n
        
        if Y is not None and Y.shape[0] > 0:
            # Evaluate fully when reference targets are available.
            median_idx = quantiles.index(0.5) if 0.5 in quantiles else len(quantiles)//2
            yp_median = yp_n[:, median_idx:median_idx+1]  # (N, 1)
            
            # Compute evaluation metrics.
            metrics = summary_metrics(Y, yp_median, quantiles=[0.5])
            out[name_tag].update(metrics)
            out[name_tag]["y_true_den"] = Y
        else:
            # Predictions without reference targets
            out[name_tag]["y_true_den"] = None

    return out

# Backward-compatible evaluation wrapper
def eval_model_full(model, Xtr, Ytr, Xval, Yval, Xte, Yte, cfg, norm=None, clip01=True):
    """
    Evaluate all three subsets through the backward-compatible interface.
    
    Returns:
        dict: evaluation results for train, val, and test
    """
    full_out = {}
    
    # Evaluate each subset separately.
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
