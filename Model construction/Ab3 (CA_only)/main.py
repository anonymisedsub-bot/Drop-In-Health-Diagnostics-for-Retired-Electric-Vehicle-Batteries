"""
Labelling-Budget Benchmark Execution Pipeline.
Handles battery data routing, random labelling of m samples at each budget step,
dynamic confidence interval generation via scipy.stats, and joint-training epochs.
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
    train_epoch, eval_epoch, eval_sweep_k
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
    """Applies staircase decay in conjunction with Plateau."""
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
    set_global_seed(seed)
    
    # =======================================================
    # =======================================================
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

    # =======================================================
    # Persistent_workers
    # =======================================================
    sup_generator = make_torch_generator(seed, offset=1000)
    ssl_generator = make_torch_generator(seed, offset=2000)

    sup_loader = DataLoader(BatterySet(supN), batch_size=cfg.batch_size, shuffle=True, 
                            collate_fn=collate_varlen, num_workers=num_workers, 
                            pin_memory=True, persistent_workers=pers_workers,
                            generator=sup_generator, worker_init_fn=seed_worker)
                            
    val_loader = DataLoader(BatterySet(valN), batch_size=cfg.batch_size, shuffle=False, 
                            collate_fn=collate_varlen, num_workers=num_workers, 
                            pin_memory=True, persistent_workers=pers_workers)
                            
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
                pin_memory=True,
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
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=cfg.scheduler.get("factor", 0.5), patience=cfg.scheduler.get("patience", 5), min_lr=cfg.scheduler.get("min_lr", cfg.lr_min), verbose=False)
    elif pol == "cosine_epoch":
        T_max = cfg.num_pretrain_epochs + cfg.num_finetune_epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=cfg.lr_min)
    else:
        scheduler = None

    V_tail_seq = torch.tensor(np.linspace(cfg.V_MIN, cfg.VSEG_LOW, cfg.TAIL_LEN), dtype=torch.float32)
    best_loss  = 1e9
    best_state = {k: v.cpu() for k, v in net.state_dict().items()}
    patience   = getattr(cfg, "patience_epochs", 30)

    log_path = out_dir / "log.jsonl"
    open(log_path, "w").close()
    
    # ================== Phase 1: Joint Training ==================
    # Note: Pre-training loop is omitted structurally if cfg.num_pretrain_epochs = 0
    print(f"[BUDGET] Joint-train ({cfg.num_finetune_epochs} epochs)")
    setattr(train_one_round, "_bad_jt", 0)
    
    for epoch in range(cfg.num_finetune_epochs):
        global_ep = cfg.num_pretrain_epochs + epoch
        ret = train_epoch(net, sup_iter, ssl_iter, optimizer, V_tail_seq, is_pretrain=False, current_global_epoch=global_ep)
        tr, w = (ret if isinstance(ret, tuple) else (ret, {}))

        va = eval_sweep_k(net, val_loader, V_tail_seq, cfg.MAX_SEGS, cfg.val_tail_weight) \
             if getattr(cfg, "val_mode_finetune", "random_k") == "sweep_k" \
             else eval_epoch(net, val_loader, V_tail_seq, cfg.MAX_SEGS)

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
            best_loss, best_state = va["total"], {k: v.cpu() for k, v in net.state_dict().items()}
            setattr(train_one_round, "_bad_jt", 0)
        else:
            bad = getattr(train_one_round, "_bad_jt", 0) + 1
            setattr(train_one_round, "_bad_jt", bad)
            if bad >= patience:
                print(f"[BUDGET] Early stop joint-train at epoch {epoch}")
                break

    torch.save(best_state, out_dir / "best.pt")
    net.load_state_dict(best_state)
    net.eval().to(cfg.device)

    # ================== Test Evaluation & 95% CI Generation ==================
    @torch.no_grad()
    def _pred_on_pack(packN):
        dl = DataLoader(BatterySet(packN), batch_size=cfg.batch_size, shuffle=False, 
                        collate_fn=collate_varlen, num_workers=0, pin_memory=True)
        
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
    df_perf.to_excel(out_dir / "summary_performance.xlsx", index=False)

    np.save(out_dir / "pool_indices.npy", pool_idx)

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    return metrics, y_true_te, y_pred_te


# ======================================================================
# Main Loop
# ======================================================================
def run_one_seed(seed: int, al_name: str, results_root: Path):
    set_global_seed(seed)
    rng = np.random.RandomState(seed + 123)

    _, Q = load_all_mats(cfg.DATA_MAT_PATH)
    N = Q.shape[1]
    splits = active_split_indices(N, seed=seed)
    train_idx = splits["train"]; pool_idx = splits["pool"]; val_idx = splits["val"]; test_idx = splits["test"]

    results_root.mkdir(parents=True, exist_ok=True)
    
    with open(results_root / "splits_init.json", "w", encoding="utf-8") as f:
        json.dump({k: v.tolist() for k, v in splits.items()}, f, ensure_ascii=False, indent=2)

    V_init = simulate_initial_voltage(N, seed=seed + 999) 
    np.save(results_root / "V_init.npy", V_init)
    
    init_pool_size = len(pool_idx)
    it = 1; all_iters = []

    # Labelling-budget steps stop when the candidate set is empty.
    while len(pool_idx) + 1 > 0 and it <= init_pool_size / cfg.num_tr_per_iter + 1:
        iter_dir = results_root / f"iter_{it}"
        print(f"\n[BUDGET] === Step {it} (train={len(train_idx)}, candidates={len(pool_idx)}) ===")

        metrics, _, _ = train_one_round(train_idx, pool_idx, val_idx, test_idx, V_init, seed, iter_dir)

        k = int(cfg.num_tr_per_iter)
        if k <= 0: break

        # Randomly label m samples from the candidate set for the next budget step.
        if len(pool_idx) <= k:
            chosen_rel = np.arange(len(pool_idx), dtype=int)
        else:
            chosen_rel = rng.choice(len(pool_idx), size=k, replace=False)

        chosen_abs = pool_idx[chosen_rel]

        with pd.ExcelWriter(iter_dir / "al_results.xlsx", engine="openpyxl") as w:
            pd.DataFrame({"train_global_indices": train_idx}).to_excel(w, sheet_name="train_indices", index=False)
            pd.DataFrame({"pool_global_indices": pool_idx}).to_excel(w, sheet_name="pool_indices", index=False)

        ratio_labeled = ((len(train_idx) - cfg.num_tr_initial) / max(1, init_pool_size) if init_pool_size > 0 else 0.0)
        summary_iter = dict(
            iter=it, seed=int(seed), al_strategy=al_name,
            train_size=int(len(train_idx)) - cfg.num_tr_initial,
            pool_left=int(len(pool_idx)), train_size_over_init_pool=float(ratio_labeled),
            MAPE=float(metrics["MAPE"]), RMSPE=float(metrics["RMSPE"]), R2=float(metrics["R2"]),
        )
        all_iters.append(summary_iter)
        
        with open(iter_dir / "summary_iter.json", "w", encoding="utf-8") as f:
            json.dump(summary_iter, f, ensure_ascii=False, indent=2)

        # Move selected samples from the pool into the labelled training set.
        train_idx = np.sort(np.concatenate([train_idx, chosen_abs]))
        mask = np.ones(len(pool_idx), dtype=bool); mask[chosen_rel] = False
        pool_idx = pool_idx[mask]
        it += 1

    with open(results_root / "al_summary_all_iters.json", "w", encoding="utf-8") as f:
        json.dump(all_iters, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    seeds   = getattr(cfg, "random_seeds", [0])
    for seed in seeds:
        root = script_dir / f"Results_Vtail_{cfg.TAIL_LEN_ORIGIN}_seed_{seed}"
        root.mkdir(parents=True, exist_ok=True)
        out_dir = root / f"Ab3_random_seed{seed}"
        print(f"[OUTPUT] {out_dir}")
        run_one_seed(seed, "random", out_dir)
