"""
Model Architecture for Battery Health Diagnosis.

Ab4 cross-attention-only version:
- Condition token is used as Query.
- Projected tail local tokens are used as Key/Value.
- The attention-refined condition token is injected into tail tokens before the residual trunk.
- FiLM is completely removed.

Expected interface:
    tail_out, cap_out, cap_da_feat = net(X_tail, S_stacked, S_batch_indices, Z_meta)

Expected output shapes:
    tail_out:    (B, TAIL_LEN, num_quantiles)
    cap_out:     (B, num_quantiles)
    cap_da_feat: (B, 64)
"""

import torch
import torch.nn as nn
import config as cfg

try:
    from torch_scatter import scatter_softmax, scatter_sum
except ImportError:
    raise ImportError(
        "Please install `torch-scatter` to support efficient batched segment aggregation."
    )


# =============================================================================
# 1. Convolutional sequence encoder
# =============================================================================
class ConvEncoder(nn.Module):
    """
    One-dimensional convolutional encoder for voltage-domain battery sequences.

    Args:
        in_ch:
            Number of input channels. For this framework, in_ch=2, corresponding
            to Q(V) and dQ/dV.
        d_model:
            List of output channel dimensions for each convolutional layer.
        n_layers:
            Number of convolutional layers. Must match len(d_model).

    Input:
        x: (B, in_ch, T)

    Returns:
        h: (B, d_model[-1], T), local feature map.
        g: (B, d_model[-1]), global pooled representation.
    """

    def __init__(self, in_ch, d_model=(16, 64, 128), n_layers=3):
        super().__init__()

        if len(d_model) != n_layers:
            raise ValueError(
                f"Length of d_model ({len(d_model)}) must equal n_layers ({n_layers})."
            )

        layers = []
        ch_in = in_ch

        for ch_out in d_model:
            layers += [
                nn.Conv1d(
                    in_channels=ch_in,
                    out_channels=ch_out,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                nn.GELU(),
            ]
            ch_in = ch_out

        self.net = nn.Sequential(*layers)
        self.out_features = d_model[-1]
        self.pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, x):
        h = self.net(x)
        g = self.pool(h).squeeze(-1)
        return h, g


# =============================================================================
# 2. Variable-length upstream fragment aggregator
# =============================================================================
class SegmentSetAggregator(nn.Module):
    """
    Batched set aggregator for opportunistically available upstream fragments.

    The aggregator first encodes each upstream fragment with a shared ConvEncoder,
    then performs query-based attention pooling within each cell using scatter
    operations. This supports a variable number of upstream fragments per cell.
    """

    def __init__(
        self,
        in_ch=2,
        d_model=(16, 64, 128),
        n_layers=3,
        use_learned_zero=False,
    ):
        super().__init__()

        self.seg_encoder = ConvEncoder(
            in_ch=in_ch,
            d_model=d_model,
            n_layers=n_layers,
        )
        d_seg = d_model[-1]

        self.query = nn.Parameter(torch.randn(1, d_seg))
        self.use_learned_zero = use_learned_zero

        if self.use_learned_zero:
            self.z_zero_learned = nn.Parameter(torch.zeros(1, d_seg))
        else:
            self.register_buffer("z_zero_hard", torch.zeros(1, d_seg))

    def _get_zero_vector(self, device):
        if self.use_learned_zero:
            return self.z_zero_learned.squeeze(0)
        return self.z_zero_hard.squeeze(0).to(device)

    @torch._dynamo.disable
    def forward(self, S_stacked, S_batch_indices, B: int):
        device = S_stacked.device

        if S_stacked.numel() == 0:
            z_zero = self._get_zero_vector(device)
            return z_zero.unsqueeze(0).expand(B, -1), None

        _, G_flat = self.seg_encoder(S_stacked)

        att_scores = (self.query * G_flat).sum(dim=-1)
        att_weights = scatter_softmax(att_scores, S_batch_indices, dim=0)

        G_weighted = G_flat * att_weights.unsqueeze(-1)
        z_aggregated = scatter_sum(
            G_weighted,
            S_batch_indices,
            dim=0,
            dim_size=B,
        )

        if self.use_learned_zero:
            has_seg_mask = torch.zeros(B, dtype=torch.bool, device=device)
            has_seg_mask.scatter_(
                dim=0,
                index=S_batch_indices.unique(),
                value=True,
            )

            z_zero = self._get_zero_vector(device)
            z_final = torch.where(
                has_seg_mask.unsqueeze(-1),
                z_aggregated,
                z_zero,
            )
            return z_final, None

        return z_aggregated, None


# =============================================================================
# 3. Cross-attention block
# =============================================================================
class CrossAttnBlock(nn.Module):
    """
    Cross-attention block with pre-normalisation and residual connection.

    In this Ab4 setting:
        Query  = condition token, shape (B, 1, d_fusion)
        Memory = projected tail tokens, shape (B, TAIL_LEN, d_fusion)

    Output:
        refined condition token, shape (B, 1, d_fusion)
    """

    def __init__(self, d_model=48, n_heads=8):
        super().__init__()

        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})."
            )

        self.norm_q = nn.LayerNorm(d_model)
        self.norm_mem = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            batch_first=True,
        )

    def forward(self, q, mem, mem_valid_mask=None):
        q_norm = self.norm_q(q)
        mem_norm = self.norm_mem(mem)

        key_padding_mask = None
        if mem_valid_mask is not None:
            key_padding_mask = ~mem_valid_mask

        out, _ = self.attn(
            q_norm,
            mem_norm,
            mem_norm,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )

        return out + q


# =============================================================================
# 4. Residual down-projection trunk
# =============================================================================
class ResidualDownBlock(nn.Module):
    """
    Residual fully connected block operating on the last dimension.

    Input:
        x: (B, T, d_in)

    Output:
        y: (B, T, d_out)
    """

    def __init__(self, d_in, d_out):
        super().__init__()

        self.proj = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()
        self.net = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, d_out),
            nn.ReLU(),
            nn.Linear(d_out, d_out),
        )

    def forward(self, x):
        y = self.net(x)
        res = self.proj(x)
        return torch.relu(y + res)


class ResidualDownStack(nn.Module):
    """
    Stack of residual down-projection blocks.

    Example:
        dims=[48, 32, 16, 8]
        maps (B, T, 48) -> (B, T, 8)
    """

    def __init__(self, dims):
        super().__init__()

        if len(dims) < 2:
            raise ValueError("ResidualDownStack requires at least two dimensions.")

        self.blocks = nn.ModuleList(
            [ResidualDownBlock(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


# =============================================================================
# 5. Conditional decoder
# =============================================================================
class ConditionalDecoder(nn.Module):
    """
    Cross-attention-only Conditional Decoder.

    Execution logic:
        1. Encode the universally available tail segment:
              X_tail -> h_tail, g_tail
        2. Encode and aggregate opportunistically available upstream fragments:
              S_stacked -> z_opt
        3. Construct a global condition vector:
              cond = Proj([g_tail, z_opt, Z_meta])
        4. Use cond as the Query and projected h_tail as Key/Value:
              refined_cond = CrossAttn(q=cond, mem=tail_tokens)
        5. Inject the attention-refined condition into local tail tokens:
              q_tokens = tail_tokens + refined_cond
        6. Feed q_tokens into the residual trunk.
        7. Produce:
              tail_out: tail reconstruction quantiles
              cap_out:  maximum-capacity quantiles
              cap_da_feat: 64-dimensional latent feature for DA
    """

    def __init__(self, cfg):
        super().__init__()

        mcfg = cfg.model
        self.mcfg = mcfg

        # ------------------------------------------------------------------
        # Tail encoder
        # ------------------------------------------------------------------
        tail_d_model = mcfg["d_tail"]
        self.tail_enc = ConvEncoder(
            in_ch=2,
            d_model=tail_d_model,
            n_layers=len(tail_d_model),
        )

        # ------------------------------------------------------------------
        # Upstream fragment set aggregator
        # ------------------------------------------------------------------
        seg_d_model = mcfg["d_seg"]
        if len(seg_d_model) != mcfg["n_layers_seg"]:
            raise ValueError(
                f"Config mismatch: len(d_seg)={len(seg_d_model)} "
                f"but n_layers_seg={mcfg['n_layers_seg']}."
            )

        self.seg_agg = SegmentSetAggregator(
            in_ch=2,
            d_model=seg_d_model,
            n_layers=mcfg["n_layers_seg"],
            use_learned_zero=mcfg.get("use_learned_zero", False),
        )

        # ------------------------------------------------------------------
        # Cross-attention conditioning
        # ------------------------------------------------------------------
        d_model = mcfg["d_fusion"]
        self.d_model = d_model
        self.use_cross = mcfg.get("use_cross_attn", True)

        d_meta = mcfg.get("d_meta", 0)
        d_in_proj = mcfg["d_tail"][-1] + mcfg["d_seg"][-1] + d_meta

        self.proj_cond = nn.Linear(d_in_proj, d_model)

        self.tail_proj = nn.Conv1d(
            in_channels=mcfg["d_tail"][-1],
            out_channels=d_model,
            kernel_size=1,
        )

        if self.use_cross:
            self.cond_tail_attn = CrossAttnBlock(
                d_model=d_model,
                n_heads=mcfg["n_heads"],
            )

        # This is not FiLM; it only maps the refined global context into the
        # same token feature space before residual injection.
        self.cond_inject = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
        )

        # ------------------------------------------------------------------
        # Residual trunk
        # ------------------------------------------------------------------
        self.trunk = ResidualDownStack([d_model, 16, 8])

        # ------------------------------------------------------------------
        # Multi-task heads
        # ------------------------------------------------------------------
        num_quantiles = len(mcfg["quantiles"])

        self.head_tail = nn.Linear(8, num_quantiles)

        self.cap_mlp = nn.Sequential(
            nn.Linear(cfg.TAIL_LEN * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.cap_fc_last = nn.Linear(64, num_quantiles)

    def forward(self, X_tail, S_stacked, S_batch_indices, Z_meta):
        """
        Args:
            X_tail:
                Tail input, shape (B, 2, TAIL_LEN).
            S_stacked:
                Flattened upstream fragments, shape (N_total_segments, 2, SEG_LEN).
            S_batch_indices:
                Batch index for each upstream fragment, shape (N_total_segments,).
            Z_meta:
                Metadata tensor, shape (B, d_meta). If d_meta=0, this is (B, 0).

        Returns:
            tail_out:
                Tail reconstruction quantiles, shape (B, TAIL_LEN, num_quantiles).
            cap_out:
                Maximum-capacity quantiles, shape (B, num_quantiles).
            cap_da_feat:
                Penultimate capacity representation, shape (B, 64).
        """

        B = X_tail.size(0)

        if Z_meta is None:
            Z_meta = X_tail.new_zeros(B, self.mcfg.get("d_meta", 0))

        # ------------------------------------------------------------------
        # 1. Tail feature extraction
        # ------------------------------------------------------------------
        h_tail, g_tail = self.tail_enc(X_tail)

        # ------------------------------------------------------------------
        # 2. Upstream fragment aggregation
        # ------------------------------------------------------------------
        z_opt, _ = self.seg_agg(
            S_stacked,
            S_batch_indices,
            B,
        )

        # ------------------------------------------------------------------
        # 3. Global condition token
        # ------------------------------------------------------------------
        cond_in = torch.cat([g_tail, z_opt, Z_meta], dim=1)
        cond = self.proj_cond(cond_in)
        cond_query = cond.unsqueeze(1)

        # ------------------------------------------------------------------
        # 4. Tail-token memory for Key/Value
        # ------------------------------------------------------------------
        tail_tokens = self.tail_proj(h_tail).permute(0, 2, 1).contiguous()

        # ------------------------------------------------------------------
        # 5. Cond-as-query cross-attention
        # ------------------------------------------------------------------
        if self.use_cross:
            refined_cond = self.cond_tail_attn(
                q=cond_query,
                mem=tail_tokens,
                mem_valid_mask=None,
            )
        else:
            refined_cond = cond_query

        # ------------------------------------------------------------------
        # 6. Inject refined global condition into local tail tokens
        # ------------------------------------------------------------------
        # cond_update = self.cond_inject(refined_cond)
        # q_tokens = tail_tokens * cond_update.expand(-1, tail_tokens.size(1), -1)
        # # q_tokens = cond_update.expand(-1, tail_tokens.size(1), -1)

        # refined_cond: (B, 1, d_fusion)
        cond_update = self.cond_inject(refined_cond)  # (B, 1, d_fusion)

        # Convert attention-refined condition into a bounded gate
        gate = torch.sigmoid(cond_update)  # (B, 1, d_fusion)

        # Broadcast the gate to all tail positions and modulate tail tokens
        q_tokens = tail_tokens * gate.expand(-1, tail_tokens.size(1), -1)
        # q_tokens: (B, TAIL_LEN, d_fusion)

        # ------------------------------------------------------------------
        # 7. Residual trunk
        # ------------------------------------------------------------------
        x_out = self.trunk(q_tokens)

        # ------------------------------------------------------------------
        # 8. Multi-task heads
        # ------------------------------------------------------------------
        tail_out = self.head_tail(x_out)

        expected_tail_shape = (
            B,
            cfg.TAIL_LEN,
            len(self.mcfg["quantiles"]),
        )
        if tuple(tail_out.shape) != expected_tail_shape:
            raise RuntimeError(
                f"tail_out shape mismatch: expected {expected_tail_shape}, "
                f"got {tuple(tail_out.shape)}."
            )

        x_flat = x_out.reshape(B, -1)
        cap_hidden = self.cap_mlp(x_flat)
        cap_out = self.cap_fc_last(cap_hidden)

        cap_da_feat = cap_hidden

        return tail_out, cap_out, cap_da_feat
