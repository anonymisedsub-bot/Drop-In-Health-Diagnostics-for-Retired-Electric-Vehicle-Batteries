"""Neural network modules for battery capacity diagnosis experiments."""
import torch
import torch.nn as nn
def scatter_mean(src, index, dim=0, dim_size=None):
    """Parameter-free mean reduction preserving empty groups (native PyTorch)."""
    if dim != 0 or index.ndim != 1:
        raise ValueError("This model reduces rows with one-dimensional indices")
    size = int(dim_size) if dim_size is not None else (int(index.max()) + 1 if index.numel() else 0)
    result = src.new_zeros((size,) + src.shape[1:])
    result.index_add_(0, index, src)
    counts = src.new_zeros(size)
    counts.index_add_(0, index, src.new_ones(index.numel()))
    return result / counts.clamp_min(1).reshape((size,) + (1,) * (src.ndim - 1))


# ==========================================
# 1. Tail sequence encoder
# ==========================================
class TailEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        dims = [2] + list(cfg.model["d_tail"])

        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Conv1d(dims[i], dims[i + 1], kernel_size=3, padding=1))
            layers.append(nn.BatchNorm1d(dims[i + 1]))
            layers.append(nn.ReLU())

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x: (B, 2, TAIL_LEN)

        Returns:
            h_tail_raw: (B, d_tail[-1], TAIL_LEN)
        """
        return self.net(x)


# ==========================================
# 2. Upstream segment encoder
# ==========================================
class SegmentEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        dims = [2] + list(cfg.model["d_seg"])

        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Conv1d(dims[i], dims[i + 1], kernel_size=3, padding=1))
            layers.append(nn.BatchNorm1d(dims[i + 1]))
            layers.append(nn.ReLU())

        self.net = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        """
        Args:
            x: (N_total_segments, 2, SEG_LEN)

        Returns:
            f_seg: (N_total_segments, d_seg[-1])
        """
        feat = self.net(x)
        return self.global_pool(feat).squeeze(-1)


# ==========================================
# 3. Upstream segment set aggregator
# ==========================================
class SegmentSetAggregator(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.seg_enc = SegmentEncoder(cfg)
        self.hidden_dim = cfg.model["hidden_dim"]
        self.use_learned_zero = cfg.model.get("use_learned_zero", True)

        if self.use_learned_zero:
            self.learned_zero = nn.Parameter(torch.zeros(1, self.hidden_dim))

    def forward(self, S_stacked, S_batch_indices, batch_size: int):
        """
        Args:
            S_stacked:
                Flattened upstream segments, shape (N_total, 2, SEG_LEN).
                It can be empty when k=0 for all samples.
            S_batch_indices:
                Batch index of each segment, shape (N_total,).
            batch_size:
                Original batch size B. This is essential when S_stacked is empty.

        Returns:
            z_agg: (B, hidden_dim)
        """
        B = int(batch_size)
        device = S_stacked.device

        # Case 1: all samples have no upstream segments, e.g., k=0 evaluation.
        if S_stacked.numel() == 0:
            if self.use_learned_zero:
                return self.learned_zero.expand(B, -1)
            return torch.zeros(B, self.hidden_dim, device=device)

        # Case 2: at least one segment exists in the batch.
        f_segs = self.seg_enc(S_stacked)  # (N_total, hidden_dim)

        # Use dim_size=B to keep rows for samples with zero available segments.
        z_agg = scatter_mean(
            f_segs,
            S_batch_indices,
            dim=0,
            dim_size=B,
        )  # (B, hidden_dim)

        # Replace empty rows by learned zero.
        if self.use_learned_zero:
            counts = torch.zeros(B, device=S_batch_indices.device)
            counts.scatter_add_(
                0,
                S_batch_indices,
                torch.ones_like(S_batch_indices, dtype=torch.float),
            )

            empty_mask = (counts == 0).unsqueeze(-1)  # (B, 1)
            learned_zero_full = self.learned_zero.expand(B, -1)

            z_agg = torch.where(empty_mask, learned_zero_full, z_agg)

        return z_agg


# ==========================================
# 4. Tail reconstruction decoder
# ==========================================
class TailDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        # Do not hard-code 64. Use config instead.
        self.seq_len = int(cfg.TAIL_LEN)

        self.net = nn.Sequential(
            nn.Linear(cfg.model["hidden_dim"], 256),
            nn.ReLU(),
            nn.Linear(256, self.seq_len),
        )

    def forward(self, x):
        """
        Args:
            x: (B, hidden_dim)

        Returns:
            recon_tail: (B, TAIL_LEN)
        """
        return self.net(x)


# ==========================================
# 5. Conditional decoder with gated FiLM fusion
# ==========================================
class ConditionalDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.hidden_dim = cfg.model["hidden_dim"]
        self.d_meta = cfg.model.get("d_meta", 0)

        self.tail_only = cfg.model.get("tail_only", False)
        self.use_gating = cfg.model.get("use_gating", True)
        self.use_film = cfg.model.get("use_film", True)

        # Tail-only mode disables all upstream fusion components.
        if self.tail_only:
            self.use_gating = False
            self.use_film = False

        self.tail_enc = TailEncoder(cfg)
        self.seg_agg = SegmentSetAggregator(cfg)
        self.tail_dec = TailDecoder(cfg)

        self.tail_proj = nn.Linear(cfg.model["d_tail"][-1], self.hidden_dim)
        self.seg_proj = nn.Linear(cfg.model["d_seg"][-1], self.hidden_dim)

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # The gate is only required when FiLM and gating are both enabled.
        if self.use_film and self.use_gating:
            self.gate_layer = nn.Sequential(
                nn.Linear(self.hidden_dim * 2, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.Sigmoid(),
            )

        if self.use_film:
            self.film_gen = nn.Linear(self.hidden_dim, self.hidden_dim * 2)

        input_dim = self.hidden_dim + self.d_meta
        self.cap_dec = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, len(cfg.model["quantiles"])),
        )

    def forward(self, X_tail, S_stacked, S_batch_indices, Z_meta):
        """
        Args:
            X_tail:
                Tail sequence, shape (B, 2, TAIL_LEN).
            S_stacked:
                Flattened upstream segments, shape (N_total, 2, SEG_LEN).
                It can be empty in tail-only mode.
            S_batch_indices:
                Batch indices for upstream segments, shape (N_total,).
                It can be empty in tail-only mode.
            Z_meta:
                Metadata, shape (B, d_meta). It can be shape (B, 0).

        Returns:
            recon_out:
                Tail reconstruction, shape (B, TAIL_LEN).
            cap_out:
                Quantile capacity prediction, shape (B, num_quantiles).
            z_latent:
                Latent representation, shape (B, hidden_dim).
        """
        B = X_tail.size(0)

        # A. Encode the low-voltage tail sequence.
        h_tail_raw = self.tail_enc(X_tail)  # (B, d_tail[-1], TAIL_LEN)

        h_tail = self.tail_proj(
            h_tail_raw.transpose(1, 2)
        ).transpose(1, 2)  # (B, hidden_dim, TAIL_LEN)

        # B. Pool the tail feature for regression and reconstruction.
        g_tail = self.global_pool(h_tail).squeeze(-1)  # (B, hidden_dim)

        # C. Use the tail-only path unless upstream modulation is enabled.
        h_fused = h_tail

        # D. Apply optional upstream-conditioned FiLM modulation.
        # Tail-only mode disables all upstream fusion components.
        if (not self.tail_only) and self.use_film:
            z_seg = self.seg_agg(
                S_stacked,
                S_batch_indices,
                batch_size=B,
            )  # (B, hidden_dim)

            z_opt = self.seg_proj(z_seg)  # (B, hidden_dim)

            params = self.film_gen(z_opt)
            delta_gamma, beta = torch.split(params, self.hidden_dim, dim=-1)

            delta_gamma_seq = delta_gamma.unsqueeze(-1)  # (B, hidden_dim, 1)
            beta_seq = beta.unsqueeze(-1)                # (B, hidden_dim, 1)

            modulation = delta_gamma_seq * h_tail + beta_seq

            if self.use_gating:
                combined = torch.cat([g_tail, z_opt], dim=-1)
                gate = self.gate_layer(combined)
                gate_seq = gate.unsqueeze(-1)

                h_fused = h_tail + gate_seq * modulation
            else:
                h_fused = h_tail + modulation

        # E. Pool the fused representation into the latent feature.
        z_latent = self.global_pool(h_fused).squeeze(-1)

        # F. Predict capacity quantiles.
        if self.d_meta == 0:
            z_final = z_latent
        else:
            z_final = torch.cat([z_latent, Z_meta], dim=-1)

        cap_out = self.cap_dec(z_final)

        # G. Reconstruct the tail sequence for the auxiliary objective.
        recon_out = self.tail_dec(g_tail)

        return recon_out, cap_out, z_latent
