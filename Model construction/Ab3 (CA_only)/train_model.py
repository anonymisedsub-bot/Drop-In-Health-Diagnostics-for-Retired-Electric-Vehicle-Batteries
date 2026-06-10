import random
import torch

import config as cfg
from losses import DynamicWeighter, _w, coral_loss, mmd_loss, pinball_loss

# =========================================================================
# Helper Functions
# =========================================================================
def get_curriculum_keep(epoch, cfg):
    """Determines the maximum number of upstream segments kept based on the curriculum schedule."""
    ms_cfg = cfg.missing_sim
    for i in range(len(ms_cfg["curriculum_epochs"])):
        if epoch >= ms_cfg["curriculum_epochs"][i]:
            k_curr = ms_cfg["curriculum_keep"][i]
        else:
            break
    return min(k_curr, cfg.MAX_SEGS)

# =========================================================================
# Training Epoch
# =========================================================================
def pack_segments_to_device(S_list, device, seg_len, k_used_list=None):
    """
    Convert a batch of variable-length segment lists into flattened tensors.

    Args:
        S_list:
            List with length B. Each element is a list of segment tensors.
            Each segment has shape (2, SEG_LEN).
        device:
            Target device.
        seg_len:
            Segment length.
        k_used_list:
            Optional list of integers with length B. If provided, only the first
            k_used_list[i] segments are used for sample i. If None, all available
            segments are used.

    Returns:
        S_stacked:
            Tensor with shape (N_total, 2, SEG_LEN).
        S_indices:
            LongTensor with shape (N_total,), indicating the sample index of
            each segment in the batch.
    """
    S_stacked_cpu = []
    S_indices_cpu = []

    for i, segs in enumerate(S_list):
        if k_used_list is None:
            k_used = len(segs)
        else:
            k_used = int(k_used_list[i])

        if k_used > 0:
            S_stacked_cpu.extend(segs[:k_used])
            S_indices_cpu.extend([i] * k_used)

    if len(S_stacked_cpu) > 0:
        S_stacked = torch.stack(S_stacked_cpu).to(device, non_blocking=True).float()
        S_indices = torch.tensor(S_indices_cpu, dtype=torch.long, device=device)
    else:
        S_stacked = torch.empty(0, 2, seg_len, device=device, dtype=torch.float32)
        S_indices = torch.empty(0, dtype=torch.long, device=device)

    return S_stacked, S_indices


def sample_source_k_used(S_list, k_curr, p_drop_all, p_keep_one):
    """
    Sample the number of upstream segments used for each source-domain sample.

    This follows the original curriculum/missing-segment simulation logic:
    - with probability p_drop_all, use k=0;
    - with probability p_keep_one, randomly choose k in [1, k_available];
    - otherwise, sample k in [0, k_available] using a decreasing-weight scheme.
    """
    k_used_list = []

    for segs in S_list:
        k_available = min(len(segs), k_curr)

        if k_available == 0:
            k_used = 0
        elif random.random() < p_drop_all:
            k_used = 0
        elif random.random() < p_keep_one:
            k_used = random.randint(1, k_available)
        else:
            population = list(range(k_available + 1))
            weights = [k_available - k + 1 for k in population]
            k_used = random.choices(population, weights=weights, k=1)[0]

        k_used_list.append(k_used)

    return k_used_list


def train_epoch(
    net,
    sup_iter,
    ssl_iter,
    opt,
    V_tail_seq=None,
    is_pretrain=False,
    current_global_epoch=0,
):
    """
    Train the model for one epoch.

    This function supports:
    - supervised source-domain training;
    - unsupervised target-domain tail reconstruction;
    - optional domain adaptation via CORAL or MMD;
    - dynamic loss weighting;
    - curriculum-based missing-segment simulation for source-domain samples.

    Notes:
    -----
    The current model interface is assumed to be:

        tail_out, cap_out, da_feat = net(X_tail, S_stacked, S_indices, Z_meta)

    where:
        X_tail:    (B, 2, TAIL_LEN)
        S_stacked: (N_total, 2, SEG_LEN)
        S_indices: (N_total,)
        Z_meta:    (B, d_meta)
    """

    net.train()

    q = cfg.model["quantiles"]
    idx_median = q.index(0.5)
    sw = cfg.loss_weights

    # ------------------------------------------------------------------
    # Dynamic loss weighter
    # ------------------------------------------------------------------
    if not hasattr(net, "_dyn_weighter"):
        net._dyn_weighter = DynamicWeighter(cfg)
    dynw = net._dyn_weighter

    # ------------------------------------------------------------------
    # Enable / disable segment aggregator during pretraining
    # ------------------------------------------------------------------
    for p in net.seg_agg.parameters():
        p.requires_grad = not is_pretrain

    # ------------------------------------------------------------------
    # Domain adaptation configuration
    # ------------------------------------------------------------------
    da_mode = sw.get("da_mode", "Coral")
    da_base_weight = float(sw.get("dynamic", {}).get("base", {}).get("da", 0.0))
    compute_da = (da_base_weight > 0.0) and ("da" in sw.get("enabled", []))

    # ------------------------------------------------------------------
    # Epoch meters
    # ------------------------------------------------------------------
    meters = {
        "total": 0.0,
        "cap_mse": 0.0,
        "tail_mse": 0.0,
        "cap_pin": 0.0,
        "da_loss": 0.0,
        "tail_mse_t": 0.0,
    }

    weight_sums = {
        "recon": 0.0,
        "cap_mse": 0.0,
        "cap_pin": 0.0,
        "da": 0.0,
    }
    weight_steps = 0

    n_sup = 0
    n_tgt = 0
    n_total_for_loss = 0

    num_sup_batches = len(sup_iter._dataloader)
    num_ssl_batches = len(ssl_iter._dataloader) if ssl_iter is not None else 0
    steps_per_epoch = max(num_sup_batches, num_ssl_batches)

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------
    for _ in range(steps_per_epoch):
        eff_sw = dynw.get_effective_weights(sw)

        for key in weight_sums:
            weight_sums[key] += float(eff_sw.get(key, 0.0))
        weight_steps += 1

        opt.zero_grad(set_to_none=True)

        loss_total = torch.zeros((), device=cfg.device, dtype=torch.float32)
        has_loss = False

        da_feat_s = None
        da_feat_t = None

        # Explicit per-batch scalar observations for dynamic weighting.
        # This avoids using locals(), which is fragile and unsuitable for
        # public release code.
        loss_tail_mse_s_val = 0.0
        loss_cap_mse_s_val = 0.0
        loss_cap_pin_s_val = 0.0
        loss_tail_mse_t_val = 0.0
        loss_da_val = 0.0

        batch_sup_size = 0
        batch_tgt_size = 0

        # ==============================================================
        # 1. Source domain: labelled data
        # ==============================================================
        try:
            X_s, S_s_list, Z_s, y_s = next(sup_iter)
            batch_sup_size = X_s.size(0)
            n_sup += batch_sup_size

            X_s = X_s.to(cfg.device, non_blocking=True).float()
            Z_s = Z_s.to(cfg.device, non_blocking=True).float()
            y_s = y_s.to(cfg.device, non_blocking=True).float()

            # Safety check: y_s must be shape (B,), not (B, 1).
            # Otherwise PyTorch broadcasting will silently create a (B, B)
            # loss matrix in capacity MSE and pinball loss.
            if y_s.ndim != 1:
                raise ValueError(
                    f"Expected y_s to have shape (B,), but got {tuple(y_s.shape)}. "
                    "Please ensure BatterySet uses pack['y_cap'][:, 0]."
                )

            if is_pretrain:
                k_used_list_s = [0 for _ in S_s_list]
            else:
                ms_cfg = cfg.missing_sim
                k_curr = get_curriculum_keep(current_global_epoch, cfg)
                k_used_list_s = sample_source_k_used(
                    S_list=S_s_list,
                    k_curr=k_curr,
                    p_drop_all=float(ms_cfg["p_drop_all"]),
                    p_keep_one=float(ms_cfg["p_keep_one"]),
                )

            S_s_stacked, S_s_indices = pack_segments_to_device(
                S_s_list,
                device=cfg.device,
                seg_len=cfg.SEG_LEN,
                k_used_list=k_used_list_s,
            )

            tail_out_s, cap_out_s, da_feat_s = net(
                X_s,
                S_s_stacked,
                S_s_indices,
                Z_s,
            )

            cap_pred_s = cap_out_s[:, idx_median]

            if cap_pred_s.shape != y_s.shape:
                raise ValueError(
                    f"Shape mismatch: cap_pred_s has shape {tuple(cap_pred_s.shape)}, "
                    f"but y_s has shape {tuple(y_s.shape)}."
                )

            Qp_s = tail_out_s[..., idx_median]
            Qt_s = X_s[:, 0, :]

            loss_tail_mse_s = (Qp_s - Qt_s).pow(2).mean()
            loss_cap_mse_s = (cap_pred_s - y_s).pow(2).mean()
            loss_cap_pin_s = pinball_loss(cap_out_s, y_s, q)

            loss_sup = (
                _w(eff_sw, sw, "recon", 1.0) * loss_tail_mse_s
                + _w(eff_sw, sw, "cap_mse", 1.0) * loss_cap_mse_s
                + _w(eff_sw, sw, "cap_pin", 1.0) * loss_cap_pin_s
            )

            loss_total = loss_total + loss_sup
            has_loss = True

            loss_tail_mse_s_val = float(loss_tail_mse_s.detach())
            loss_cap_mse_s_val = float(loss_cap_mse_s.detach())
            loss_cap_pin_s_val = float(loss_cap_pin_s.detach())

            meters["tail_mse"] += loss_tail_mse_s_val * batch_sup_size
            meters["cap_mse"] += loss_cap_mse_s_val * batch_sup_size
            meters["cap_pin"] += loss_cap_pin_s_val * batch_sup_size

        except StopIteration:
            pass

        # ==============================================================
        # 2. Target domain: unlabelled data
        # ==============================================================
        if ssl_iter is not None:
            try:
                X_t, S_t_list, Z_t, _ = next(ssl_iter)
                batch_tgt_size = X_t.size(0)
                n_tgt += batch_tgt_size

                X_t = X_t.to(cfg.device, non_blocking=True).float()
                Z_t = Z_t.to(cfg.device, non_blocking=True).float()

                # Target domain uses all currently available segments after
                # physical missing-segment simulation.
                S_t_stacked, S_t_indices = pack_segments_to_device(
                    S_t_list,
                    device=cfg.device,
                    seg_len=cfg.SEG_LEN,
                    k_used_list=None,
                )

                tail_out_t, cap_out_t, da_feat_t = net(
                    X_t,
                    S_t_stacked,
                    S_t_indices,
                    Z_t,
                )

                Qp_t = tail_out_t[..., idx_median]
                Qt_t = X_t[:, 0, :]

                loss_tail_mse_t = (Qp_t - Qt_t).pow(2).mean()
                loss_tail_mse_t_val = float(loss_tail_mse_t.detach())

                loss_recon_t = _w(eff_sw, sw, "recon", 1.0) * loss_tail_mse_t
                loss_total = loss_total + loss_recon_t
                has_loss = True

                meters["tail_mse_t"] += loss_tail_mse_t_val * batch_tgt_size

                # ------------------------------------------------------
                # Domain adaptation
                # ------------------------------------------------------
                if compute_da and (da_feat_s is not None) and (da_feat_t is not None):
                    B_da = min(da_feat_s.size(0), da_feat_t.size(0))

                    if B_da > 4:
                        X_s_da = da_feat_s[:B_da]
                        X_t_da = da_feat_t[:B_da]

                        if da_mode == "MMD":
                            loss_da = mmd_loss(X_s_da, X_t_da)
                        elif da_mode == "Coral":
                            loss_da = coral_loss(X_s_da, X_t_da)
                        else:
                            raise ValueError(
                                f"Unsupported domain adaptation mode: {da_mode}. "
                                "Expected 'MMD' or 'Coral'."
                            )

                        loss_da_val = float(loss_da.detach())

                        loss_total = loss_total + _w(eff_sw, sw, "da", 0.0) * loss_da
                        has_loss = True

                        meters["da_loss"] += loss_da_val * B_da

            except StopIteration:
                pass

        # ==============================================================
        # 3. Backpropagation
        # ==============================================================
        if has_loss:
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()

            batch_effective_size = max(batch_sup_size, batch_tgt_size, 1)
            meters["total"] += float(loss_total.detach()) * batch_effective_size
            n_total_for_loss += batch_effective_size

        # ==============================================================
        # 4. Update dynamic loss weighter
        # ==============================================================
        obs = {
            "recon": loss_tail_mse_s_val + loss_tail_mse_t_val,
            "cap_mse": loss_cap_mse_s_val,
            "cap_pin": loss_cap_pin_s_val,
            "da": loss_da_val,
        }
        dynw.update_ema(obs)

    # ------------------------------------------------------------------
    # Average effective dynamic weights
    # ------------------------------------------------------------------
    if weight_steps > 0:
        avg_eff_sw = {
            key: weight_sums[key] / weight_steps
            for key in weight_sums
        }
    else:
        avg_eff_sw = {
            key: float(sw.get("dynamic", {}).get("base", {}).get(key, 0.0))
            for key in weight_sums
        }

    # ------------------------------------------------------------------
    # Normalise epoch meters
    # ------------------------------------------------------------------
    meters["total"] = meters["total"] / max(n_total_for_loss, 1)
    meters["tail_mse"] = meters["tail_mse"] / max(n_sup, 1)
    meters["cap_mse"] = meters["cap_mse"] / max(n_sup, 1)
    meters["cap_pin"] = meters["cap_pin"] / max(n_sup, 1)
    meters["tail_mse_t"] = meters["tail_mse_t"] / max(n_tgt, 1)
    meters["da_loss"] = meters["da_loss"] / max(n_tgt, 1)

    return meters, avg_eff_sw

# =========================================================================
# Evaluators
# =========================================================================
@torch.no_grad()
def eval_epoch(net, loader, V_tail_seq, max_segs):
    """Random-K Validation (Fast mode for pre-training)."""
    net.eval()
    q = cfg.model["quantiles"]
    idx_median = q.index(0.5)
    
    res = {"k0_cap_mse": 0.0, "brand_k_cap_mse": 0.0, "total": 0.0}
    n = 0

    for X_tail, S_list, Z_meta, y_cap in loader:
        B = X_tail.size(0)
        X_tail = X_tail.to(cfg.device, non_blocking=True)
        Z_meta = Z_meta.to(cfg.device, non_blocking=True)
        y_cap  = y_cap.to(cfg.device, non_blocking=True)
        
        # --- Pass 1: k=0 (Empty Segments) ---
        empty_S = torch.empty(0, 2, cfg.SEG_LEN, device=cfg.device)
        empty_idx = torch.empty(0, dtype=torch.long, device=cfg.device)
        _, cap_out_k0, _ = net(X_tail, empty_S, empty_idx, Z_meta)
        loss_k0 = (cap_out_k0[:, idx_median] - y_cap).pow(2).mean()

        # --- Pass 2: Random k (CPU-Side Flattening) ---
        S_stacked_cpu = []
        S_indices_cpu = []
        for i, segs in enumerate(S_list):
            k_avail = len(segs)
            k_used = random.randint(1, k_avail) if k_avail > 0 else 0
            if k_used > 0:
                S_stacked_cpu.extend(segs[:k_used])
                S_indices_cpu.extend([i] * k_used)
                
        if len(S_stacked_cpu) > 0:
            S_stacked = torch.stack(S_stacked_cpu).to(cfg.device, non_blocking=True)
            S_indices = torch.tensor(S_indices_cpu, dtype=torch.long, device=cfg.device)
        else:
            S_stacked = torch.empty(0, 2, cfg.SEG_LEN, device=cfg.device)
            S_indices = torch.empty(0, dtype=torch.long, device=cfg.device)

        _, cap_out_rand, _ = net(X_tail, S_stacked, S_indices, Z_meta)
        loss_rand = (cap_out_rand[:, idx_median] - y_cap).pow(2).mean()

        n += B
        res["k0_cap_mse"] += float(loss_k0) * B
        res["brand_k_cap_mse"] += float(loss_rand) * B

    res["total"] = (res["k0_cap_mse"] + res["brand_k_cap_mse"]) / (2 * max(n, 1))
    return res

@torch.no_grad()
def eval_sweep_k(net, loader, V_tail_seq, max_segs, val_tail_weight):
    """Internal helper for the experiment pipeline."""
    net.eval()
    q = cfg.model["quantiles"]
    idx_median = q.index(0.5)

    total_weighted_loss = 0.0
    total_samples = 0

    for X_tail, S_list, Z_meta, y_cap in loader:
        B = X_tail.size(0)
        total_samples += B
        X_tail = X_tail.to(cfg.device, non_blocking=True)
        Z_meta = Z_meta.to(cfg.device, non_blocking=True)
        y_cap  = y_cap.to(cfg.device, non_blocking=True)

        for k in range(max_segs + 1):
            S_stacked_cpu = []
            S_indices_cpu = []
            
            for i, segs in enumerate(S_list):
                used = segs[:k]
                if len(used) > 0:
                    S_stacked_cpu.extend(used)
                    S_indices_cpu.extend([i] * len(used))
            
            if len(S_stacked_cpu) > 0:
                S_stacked = torch.stack(S_stacked_cpu).to(cfg.device, non_blocking=True)
                S_indices = torch.tensor(S_indices_cpu, dtype=torch.long, device=cfg.device)
            else:
                S_stacked = torch.empty(0, 2, cfg.SEG_LEN, device=cfg.device)
                S_indices = torch.empty(0, dtype=torch.long, device=cfg.device)

            _, cap_out, _ = net(X_tail, S_stacked, S_indices, Z_meta)
            
            loss_k = (cap_out[:, idx_median] - y_cap).pow(2).mean()
            weight = val_tail_weight if k == 0 else 1.0
            total_weighted_loss += float(loss_k) * B * weight

    total_weight_factor = (max_segs * 1.0) + val_tail_weight
    avg_weighted_loss = total_weighted_loss / max(1, total_samples * total_weight_factor)
    return {"total": avg_weighted_loss}
