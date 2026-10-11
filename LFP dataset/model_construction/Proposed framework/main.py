"""
Labelling-Budget Benchmark Execution Pipeline.
Handles battery data routing, random labelling of m samples at each budget step,
Gaussian-derived prediction intervals via scipy.stats, and joint-training epochs.
"""
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as stats

import torch
from torch.utils.data import DataLoader

import config as cfg
from data import (
    load_all_mats, build_ordered_windows, build_subset, active_split_indices,
    fit_normalizer_tail, apply_normalizer_tail, 
    fit_normalizer_cap, apply_normalizer_cap, inverse_cap
)

from model import ConditionalDecoder
from losses import mape, rmspe, r2_score
from train_model import (
    train_epoch, eval_sweep_k
)

def set_global_seed(seed: int) -> None:
    """Set all RNGs used by the experiment as early as possible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)


def make_torch_generator(seed: int, offset: int = 0) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(int(seed) + int(offset))
    return generator


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# ======================================================================
# Physics-informed Missing Simulation
# ======================================================================
def simulate_initial_voltage(n_cells: int, seed: int) -> np.ndarray:
    """
    Assigns an initial charging voltage (V_init) to each battery cell 
    using a truncated normal distribution to simulate real-world data fragmentation.
    """
    V_low, V_high = 2.7, 3.0
    mu = (V_low + V_high) / 2.0
    sigma = (V_high - V_low) / 3.0
    rng = np.random.default_rng(seed)
    
    V_init = np.empty(n_cells, dtype=float)
    for i in range(n_cells):
        while True:
            v = rng.normal(mu, sigma)
            if cfg.VSEG_LOW < v:
                V_init[i] = v
                break
    return V_init

def compute_max_segments_from_Vinit(V0: float, win, grid) -> int:
    """Calculates the number of visible upstream segments based on V_init."""
    if len(win) == 0: return 0
    seg_upper = [grid[b - 1] for (a, b) in win]
    k = 0
    for v_up in seg_upper:
        if v_up <= V0: k += 1
        else: break
    return k

def apply_segment_availability_to_pack(pack: dict, idx_cells: np.ndarray, V_init: np.ndarray, win, grid, role: str) -> dict:
    """Applies dynamic truncation to target and test domains."""
    if role in ("source", "val") or len(win) == 0:
        return pack
        
    S_list = pack.get("S_list", [])
    new_S_list = []
    for local_i, global_idx in enumerate(idx_cells):
        V0 = float(V_init[global_idx])
        k_max = min(compute_max_segments_from_Vinit(V0, win, grid), len(S_list[local_i]))
        new_S_list.append(S_list[local_i][:k_max])
        
    pack["S_list"] = new_S_list
    return pack

# ======================================================================
# Dataset & Iterators
# ======================================================================
class BatterySet(torch.utils.data.Dataset):
    def __init__(self, pack):
        self.X_tail = torch.from_numpy(pack["X_tail"]).float()
        self.Z_meta = torch.from_numpy(pack["z_meta"]).float()

        self.y_cap = torch.from_numpy(pack["y_cap"][:, 0]).float()

        self.S_list = [
            [torch.from_numpy(s).float() for s in segs]
            for segs in pack["S_list"]
        ]

    def __len__(self):
        return self.X_tail.shape[0]

    def __getitem__(self, i):
        return self.X_tail[i], self.S_list[i], self.Z_meta[i], self.y_cap[i]

def collate_varlen(batch):
    X_tail, S_list, Z_meta, Y = [], [], [], []
    for xt, segs, meta, y in batch:
        X_tail.append(xt)
        S_list.append(segs)
        Z_meta.append(meta)
        Y.append(y)
    
    return torch.stack(X_tail), S_list, torch.stack(Z_meta), torch.stack(Y)

class InfiniteIterator:
    """Infinite iterator for Domain Adaptation oversampling."""
    def __init__(self, dl):
        self._dataloader = dl
        self._it = iter(dl)
    def __next__(self):
        try:
            return next(self._it)
        except StopIteration:
            self._it = iter(self._dataloader)
            return next(self._it)
    def __len__(self):
        return len(self._dataloader)

def _apply_norm_to_pack(pack, tail_norm, cap_norm):
    if pack["X_tail"].shape[0] == 0: return pack
    pack = dict(pack)
    pack["X_tail"] = apply_normalizer_tail(pack["X_tail"], tail_norm)
    mins = tail_norm["mins"].min(axis=1, keepdims=True)
    maxs = tail_norm["maxs"].max(axis=1, keepdims=True)
    denom = np.clip(maxs - mins, 1e-8, None)
    pack["S_list"] = [[((s - mins) / denom).astype(np.float32) for s in segs] for segs in pack["S_list"]]
    pack["y_cap"] = apply_normalizer_cap(pack["y_cap"], cap_norm)
    return pack

def _apply_step_decay_if_needed(optimizer, cfg, epoch_idx):
    """Apply the configured epoch staircase after plateau scheduling."""
    sch_cfg = cfg.scheduler
    if sch_cfg.get("policy", "plateau") != "plateau": return
    step_every = int(sch_cfg.get("epoch_step", 20))
    gamma      = float(sch_cfg.get("step_gamma", 0.8))
    min_lr     = float(sch_cfg.get("min_lr", 0.0))
    if step_every > 0 and ((epoch_idx + 1) % step_every == 0):
        for pg in optimizer.param_groups:
            pg["lr"] = max(pg["lr"] * gamma, min_lr)

# ======================================================================
# Main Training Logic
# ======================================================================
def train_one_round(train_idx, pool_idx, val_idx, test_idx, V_init, seed, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    from inference import snapshot
    snapshot(out_dir, seed, "new config-driven Proposed training")
    set_global_seed(seed)
    
    num_workers = 0  
    
    pers_workers = True if num_workers > 0 else False

    V, Q = load_all_mats(cfg.DATA_MAT_PATH)
    win, idx_tail_hi, grid = build_ordered_windows(cfg)

    train_idx = np.asarray(train_idx, dtype=int)
    pool_idx  = np.asarray(pool_idx, dtype=int)
    val_idx   = np.asarray(val_idx, dtype=int)
    test_idx  = np.asarray(test_idx, dtype=int)

    sup_pack  = build_subset(V, Q, train_idx, cfg, win, idx_tail_hi)
    val_pack  = build_subset(V, Q, val_idx,   cfg, win, idx_tail_hi)
    tst_pack  = build_subset(V, Q, test_idx,  cfg, win, idx_tail_hi)
    ssl_pack  = build_subset(V, Q, pool_idx,  cfg, win, idx_tail_hi) if len(pool_idx)>0 else None

    # Fit normalization on the labelled source domain only.
    tail_norm = fit_normalizer_tail(sup_pack["X_tail"], cfg.normalization_mode)
    cap_norm  = fit_normalizer_cap(sup_pack["y_cap"])

    supN = _apply_norm_to_pack(sup_pack, tail_norm, cap_norm)
    valN = _apply_norm_to_pack(val_pack, tail_norm, cap_norm)
    tstN = _apply_norm_to_pack(tst_pack, tail_norm, cap_norm)
    sslN = _apply_norm_to_pack(ssl_pack, tail_norm, cap_norm) if ssl_pack is not None else None

    # Simulate realistic Missing Segments for target/test sets
    if sslN is not None:
        sslN = apply_segment_availability_to_pack(sslN, pool_idx, V_init, win, grid, role="target")
    tstN = apply_segment_availability_to_pack(tstN, test_idx, V_init, win, grid, role="test")

    sup_generator = make_torch_generator(seed, offset=1000)
    ssl_generator = make_torch_generator(seed, offset=2000)

    sup_loader = DataLoader(BatterySet(supN), batch_size=cfg.batch_size, shuffle=True, 
                            collate_fn=collate_varlen, num_workers=num_workers, 
                            pin_memory=cfg.device.startswith("cuda"), persistent_workers=pers_workers,
                            generator=sup_generator, worker_init_fn=seed_worker)
                            
    val_loader = DataLoader(BatterySet(valN), batch_size=cfg.batch_size, shuffle=False, 
                            collate_fn=collate_varlen, num_workers=num_workers, 
                            pin_memory=cfg.device.startswith("cuda"), persistent_workers=pers_workers)
                            
    sup_iter = InfiniteIterator(sup_loader)
    
    # Graceful degradation if pool is empty
    if sslN is not None and sslN["X_tail"].shape[0] > 0:
        ssl_iter = InfiniteIterator(
            DataLoader(
                BatterySet(sslN),
                batch_size=cfg.batch_size,
                shuffle=True,
                collate_fn=collate_varlen,
                num_workers=num_workers,
                pin_memory=cfg.device.startswith("cuda"),
                generator=ssl_generator,
                worker_init_fn=seed_worker,
            )
        )
    else:
        ssl_iter = None

    net = ConditionalDecoder(cfg).to(cfg.device)

    optimizer = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    
    # Configure the learning-rate scheduler.
    pol = cfg.scheduler.get("policy", "plateau")
    if pol == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=cfg.scheduler.get("factor", 0.5), patience=cfg.scheduler.get("patience", 5), min_lr=cfg.scheduler["min_lr"])
    elif pol == "cosine_epoch":
        T_max = cfg.num_epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=cfg.scheduler["min_lr"])
    else:
        scheduler = None

    best_loss  = 1e9
    best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    patience   = getattr(cfg, "patience_epochs", 30)

    log_path = out_dir / "log.jsonl"
    open(log_path, "w").close()
    
    # Joint training and best-checkpoint selection.
    print(f"[BUDGET] Joint-train ({cfg.num_epochs} epochs)")
    bad_epochs = 0
    
    for epoch in range(cfg.num_epochs):
        ret = train_epoch(net, sup_iter, ssl_iter, optimizer, current_global_epoch=epoch)
        tr, w = (ret if isinstance(ret, tuple) else (ret, {}))

        va = eval_sweep_k(net, val_loader, cfg.MAX_SEGS, cfg.val_tail_weight)

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(va["total"]); _apply_step_decay_if_needed(optimizer, cfg, epoch)
        elif scheduler is not None:
            scheduler.step()

        lr_now = optimizer.param_groups[0]["lr"]
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"epoch": epoch, "train": tr, "val": va, "stage": "joint", "lr": lr_now, "weights": w}) + "\n")

        print_str = (
            f"[BUDGET][JT] epoch {epoch} "
            f"Tr_S(cap_mse={tr['cap_mse']:.4f}) Tr_T(tail_mse_t={tr['tail_mse_t']:.4f}, DA={tr.get('da_loss', 0.0):.4f}) "
            f"Val(total={va['total']:.4f}) | lr={lr_now:.3e}"
        )
        print(print_str)

        if va["total"] < best_loss:
            best_loss, best_state = va["total"], {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"[BUDGET] Early stop joint-train at epoch {epoch}")
                break

    np.savez(out_dir / "normalizers.npz", tail_mins=tail_norm["mins"], tail_maxs=tail_norm["maxs"], cap_min=cap_norm["min"], cap_max=cap_norm["max"])
    with open(out_dir / "splits.json", "w", encoding="utf-8") as handle:
        json.dump(dict(train=train_idx.tolist(), pool=pool_idx.tolist(), val=val_idx.tolist(), test=test_idx.tolist()), handle, indent=2)
    np.save(out_dir / "V_init.npy", V_init)
    torch.save(best_state, out_dir / "best.pt")
    net.load_state_dict(best_state)
    net.eval().to(cfg.device)

    # ================== Test Evaluation & Gaussian-derived prediction intervals ==================
    @torch.no_grad()
    def _pred_on_pack(packN):
        dl = DataLoader(BatterySet(packN), batch_size=cfg.batch_size, shuffle=False, 
                        collate_fn=collate_varlen, num_workers=0, pin_memory=cfg.device.startswith("cuda"))
        
        # Dynamically infer Z-scores from quantiles array
        q = cfg.model["quantiles"]
        idx_lower, idx_median, idx_upper = 0, q.index(0.5), -1
        
        z_upper = stats.norm.ppf(q[idx_upper])
        z_lower = stats.norm.ppf(q[idx_lower])
        z_distance = z_upper - z_lower

        m_median, m_lower, m_upper = [], [], []
        for X_tail, S_list, Z_meta, _ in dl:
            X_tail = X_tail.to(cfg.device, non_blocking=True)
            Z_meta = Z_meta.to(cfg.device, non_blocking=True)
            
            S_stacked_cpu = []
            S_indices_cpu = []
            for i, segs in enumerate(S_list):
                if len(segs) > 0:
                    S_stacked_cpu.extend(segs)
                    S_indices_cpu.extend([i] * len(segs))
                    
            if len(S_stacked_cpu) > 0:
                S_stacked = torch.stack(S_stacked_cpu).to(cfg.device, non_blocking=True)
                S_indices = torch.tensor(S_indices_cpu, dtype=torch.long, device=cfg.device)
            else:
                S_stacked = torch.empty(0, 2, cfg.SEG_LEN, device=cfg.device)
                S_indices = torch.empty(0, dtype=torch.long, device=cfg.device)

            _, cap_out, _ = net(X_tail, S_stacked, S_indices, Z_meta)
            
            m_median.append(cap_out[:, idx_median].cpu().numpy())
            m_lower.append(cap_out[:, idx_lower].cpu().numpy())
            m_upper.append(cap_out[:, idx_upper].cpu().numpy())
            
        m_median = np.concatenate(m_median, axis=0)[:,None]
        m_lower  = np.concatenate(m_lower, axis=0)[:,None]
        m_upper  = np.concatenate(m_upper, axis=0)[:,None]
        
        y_true = inverse_cap(packN["y_cap"], cap_norm).squeeze(-1)
        y_pred = inverse_cap(m_median,       cap_norm).squeeze(-1)
        val_lo = inverse_cap(m_lower,        cap_norm).squeeze(-1)
        val_hi = inverse_cap(m_upper,        cap_norm).squeeze(-1)
        
        # Estimate standard deviation from the model quantile interval.
        sigma = (val_hi - val_lo) / (z_distance + 1e-12)
        ci_95_lower = y_pred - 1.96 * sigma
        ci_95_upper = y_pred + 1.96 * sigma
        
        return y_true, y_pred, ci_95_lower, ci_95_upper

    y_true_te, y_pred_te, y_lo95_te, y_hi95_te = _pred_on_pack(tstN)

    metrics = dict(MAPE=mape(y_true_te, y_pred_te), RMSPE=rmspe(y_true_te, y_pred_te), R2=r2_score(y_true_te, y_pred_te))

    df_perf = pd.DataFrame({
        "y_true": y_true_te,
        "y_pred": y_pred_te,
        "y_pred_lo95": y_lo95_te,    
        "y_pred_hi95": y_hi95_te,    
    })
    df_perf["error"] = df_perf["y_pred"] - df_perf["y_true"]
    df_perf.insert(0, "cell_index_zero_based", test_idx)
    df_perf.to_csv(out_dir / "summary_performance.csv", index=False)

    np.save(out_dir / "pool_indices.npy", pool_idx)

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    return metrics, y_true_te, y_pred_te


# ======================================================================
# Main Loop
# ======================================================================
def run_one_seed(seed: int, results_root: Path):
    """Train independently at selected budgets using one fixed nested acquisition order."""
    from runtime import selected_rounds, save_config
    set_global_seed(seed)
    rng = np.random.RandomState(seed + 123)
    _, capacity = load_all_mats(cfg.DATA_MAT_PATH)
    splits = active_split_indices(capacity.shape[1], seed=seed)
    train_idx, pool_idx, val_idx, test_idx = (splits[key] for key in ('train', 'pool', 'val', 'test'))
    initial_pool_size = len(pool_idx)
    if cfg.num_tr_per_iter < 1:
        raise ValueError('num_tr_per_iter must be positive')
    rounds = selected_rounds(cfg.ROUND_NUMBERS, range(1, initial_pool_size // cfg.num_tr_per_iter + 2))
    results_root.mkdir(parents=True, exist_ok=True)
    save_config(results_root / 'config_snapshot.json', cfg)
    (results_root / 'splits_init.json').write_text(json.dumps({key: value.tolist() for key, value in splits.items()}, indent=2), encoding='utf-8')
    initial_voltage = simulate_initial_voltage(capacity.shape[1], seed=seed + 999)
    np.save(results_root / 'V_init.npy', initial_voltage)
    summaries = []
    for round_number in range(1, max(rounds) + 1):
        if round_number in rounds:
            output = results_root / f'iter_{round_number}'
            print(f'[BUDGET] Seed {seed}, round {round_number}, labelled={len(train_idx)}, pool={len(pool_idx)}', flush=True)
            metrics, _, _ = train_one_round(train_idx, pool_idx, val_idx, test_idx, initial_voltage, seed, output)
            pd.DataFrame({'train_global_indices': train_idx}).to_csv(output / 'train_indices.csv', index=False)
            pd.DataFrame({'pool_global_indices': pool_idx}).to_csv(output / 'pool_indices.csv', index=False)
            summary = dict(iter=round_number, seed=int(seed), labelled_count=len(train_idx), pool_left=len(pool_idx), **metrics)
            (output / 'summary_iter.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
            summaries.append(summary)
        count = min(cfg.num_tr_per_iter, len(pool_idx))
        chosen = np.arange(count) if count == len(pool_idx) else rng.choice(len(pool_idx), size=count, replace=False)
        train_idx = np.sort(np.concatenate([train_idx, pool_idx[chosen]]))
        pool_idx = np.delete(pool_idx, chosen)
    (results_root / 'al_summary_all_iters.json').write_text(json.dumps(summaries, indent=2), encoding='utf-8')
    return summaries


def main():
    """Execute RUN_MODE using only config.py settings."""
    from runtime import new_campaign, validate_settings
    validate_settings(cfg)
    torch.set_num_threads(cfg.THREADS)
    if cfg.RUN_MODE == 'check':
        voltage, capacity = load_all_mats(cfg.DATA_MAT_PATH)
        windows, _, _ = build_ordered_windows(cfg)
        result = dict(cells=capacity.shape[1], grid_points=voltage.shape[0], segments=len(windows), device=cfg.device)
        print(json.dumps(result, indent=2))
        return result
    output = new_campaign(cfg)
    if cfg.RUN_MODE == 'train':
        for seed in cfg.random_seeds:
            run_one_seed(seed, output / f'Proposed_random_seed{seed}')
    else:
        from inference import infer
        checkpoint = cfg.CHECKPOINT or Path(__file__).resolve().parent / 'demo/best.pt'
        infer(checkpoint, output, cfg.INFERENCE_INDICES, cfg.INFERENCE_K)
    print(f'[OUTPUT] {output}', flush=True)
    return output


if __name__ == '__main__':
    import sys
    if len(sys.argv) != 1:
        raise SystemExit('Edit config.py and run main.py without command-line arguments.')
    sys.modules['main'] = sys.modules[__name__]
    main()
