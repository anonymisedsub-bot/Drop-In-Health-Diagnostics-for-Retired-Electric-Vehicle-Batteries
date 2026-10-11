import math
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


# ====================== Self-attention pooling ======================

class SelfAttentionPooling(nn.Module):
    """Extract global features (N, D) from (N, T, D) with self-attention pooling."""
    def __init__(self, d_model: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, 1)
        )

    def forward(self, x, mask=None):
        """
        Args:
            x:    (B, T, D)
            mask: optional (B, T) mask; 1 keeps a position and 0 masks it out
        Returns:
            pooled: (B, D)
            attn:   (B, T)
        """
        # (B, T, D) -> (B, T, 1)
        attn_scores = self.attention(x)

        if mask is not None:
            # Align the mask shape to (B, T, 1).
            mask = mask.unsqueeze(-1)
            attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = torch.softmax(attn_scores, dim=1)  # (B, T, 1)
        pooled = (x * attn_weights).sum(dim=1)            # (B, D)
        return pooled, attn_weights.squeeze(-1)


# ====================== Positional encoding ======================

class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Args:
            x: (B, T, D)
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


class LearnedPositionalEncoding(nn.Module):
    """Learnable positional encoding."""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.normal_(self.pe, mean=0.0, std=0.02)

    def forward(self, x):
        """
        Args:
            x: (B, T, D)
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


# ====================== TCN modules ======================

class Chomp1d(nn.Module):
    """Remove excess right padding to preserve sequence length."""
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size]


class TemporalBlock(nn.Module):
    """TCN residual block: two Conv1d -> Chomp -> activation -> dropout stages plus a residual."""
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout,
                 activation_type: str = "relu"):
        super().__init__()
        padding = (kernel_size - 1) * dilation

        self.conv1 = weight_norm(
            nn.Conv1d(
                in_ch, out_ch, kernel_size,
                stride=1, padding=padding, dilation=dilation,
            )
        )
        self.chomp1 = Chomp1d(padding)
        self.activation1 = self._make_activation(activation_type)
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(
            nn.Conv1d(
                out_ch, out_ch, kernel_size,
                stride=1, padding=padding, dilation=dilation,
            )
        )
        self.chomp2 = Chomp1d(padding)
        self.activation2 = self._make_activation(activation_type)
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.final_activation = self._make_activation(activation_type)

    @staticmethod
    def _make_activation(act_type: str):
        if act_type == "relu":
            return nn.ReLU()
        elif act_type == "gelu":
            return nn.GELU()
        elif act_type == "leaky_relu":
            return nn.LeakyReLU(0.1)
        else:
            return nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.activation1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.activation2(out)
        out = self.dropout2(out)

        res = x if self.downsample is None else self.downsample(x)
        return self.final_activation(out + res)


class TemporalConvNet(nn.Module):
    """Stack multiple TCN layers."""
    def __init__(
        self,
        num_inputs: int,
        channels,
        kernel_size: int = 3,
        dropout: float = 0.1,
        activation_type: str = "relu",
    ):
        super().__init__()
        layers = []
        num_levels = len(channels)
        for i in range(num_levels):
            in_ch = num_inputs if i == 0 else channels[i - 1]
            out_ch = channels[i]
            dilation = 2 ** i  # 1, 2, 4, ...
            layers.append(
                TemporalBlock(
                    in_ch=in_ch,
                    out_ch=out_ch,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                    activation_type=activation_type,
                )
            )
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # x: (N, C, T)
        return self.network(x)  # (N, C_out, T)


# ====================== TCN + Transformer encoder-only regression ======================

class TCNTransformerEncoderRegressor(nn.Module):
    """
    Compact TCN + Transformer encoder regression model with quantile outputs.
    Architecture:
      (N, C_in, T)
        -> [optional TCN] -> (N, C_tcn, T)
        → transpose → (N, T, C_enc)
        → Linear(C_enc → d_model)
        → PositionalEncoding
        -> TransformerEncoder (L layers)
        -> Pooling (mean or self-attention)
        -> Output MLP
        -> Quantile heads
    """
    def __init__(self, config):
        super().__init__()

        # All model settings are supplied by config.py.
        self.config = dict(config)

        # Parse the configuration.
        d_model            = self.config["d_model"]
        nhead              = self.config["nhead"]
        num_encoder_layers = self.config["num_encoder_layers"]
        dim_feedforward    = self.config["dim_feedforward"]
        max_seq_len        = self.config["max_seq_len"]
        attention_dropout  = self.config["attention_dropout"]
        use_pos_encoding   = self.config["use_pos_encoding"]
        pos_encoding_type  = self.config["pos_encoding_type"]
        pooling_type       = self.config["pooling_type"]

        output_proj_layers = self.config["output_proj_layers"]
        output_activation  = self.config["output_activation"]
        output_dropout     = self.config["output_dropout"]

        in_ch              = self.config["in_ch"]
        output_dim         = self.config["output_dim"]
        quantiles          = self.config["quantiles"]

        # TCN configuration
        self.use_tcn       = self.config["use_tcn"]
        tcn_channels       = self.config["tcn_channels"]
        tcn_kernel_size    = self.config["tcn_kernel_size"]
        tcn_dropout        = self.config["tcn_dropout"]
        tcn_activation     = self.config["tcn_activation"]

        # Quantiles
        self.quantiles = list(quantiles)
        self.num_quantiles = len(self.quantiles)

        # ---------- TCN frontend ----------
        if self.use_tcn:
            self.tcn = TemporalConvNet(
                num_inputs=in_ch,
                channels=tcn_channels,
                kernel_size=tcn_kernel_size,
                dropout=tcn_dropout,
                activation_type=tcn_activation,
            )
            encoder_input_dim = tcn_channels[-1]
        else:
            self.tcn = None
            encoder_input_dim = in_ch

        # ---------- Project inputs to d_model. ----------
        self.input_proj = nn.Linear(encoder_input_dim, d_model)
        self.d_model = d_model

        # ---------- Positional encoding ----------
        if use_pos_encoding:
            if pos_encoding_type == "sinusoidal":
                self.pos_encoding = PositionalEncoding(
                    d_model=d_model,
                    dropout=attention_dropout,
                    max_len=max_seq_len,
                )
            elif pos_encoding_type == "learned":
                self.pos_encoding = LearnedPositionalEncoding(
                    d_model=d_model,
                    dropout=attention_dropout,
                    max_len=max_seq_len,
                )
            else:
                self.pos_encoding = PositionalEncoding(
                    d_model=d_model,
                    dropout=attention_dropout,
                    max_len=max_seq_len,
                )
        else:
            self.pos_encoding = nn.Identity()

        # ---------- Transformer Encoder ----------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=attention_dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)

        # ---------- Pooling layer ----------
        self.pooling_type = pooling_type
        if pooling_type == "attention":
            self.pool = SelfAttentionPooling(d_model)
        elif pooling_type == "mean":
            self.pool = None   # Use mean pooling.
        else:
            raise ValueError(f"Unsupported pooling_type: {pooling_type}")

        # ---------- Output MLP backbone ----------
        fc_dims = [d_model] + output_proj_layers
        backbone_layers = []
        for i in range(len(fc_dims) - 1):
            in_d  = fc_dims[i]
            out_d = fc_dims[i + 1]
            backbone_layers.append(nn.Linear(in_d, out_d))

            if output_activation == "relu":
                act = nn.ReLU()
            elif output_activation == "gelu":
                act = nn.GELU()
            elif output_activation == "leaky_relu":
                act = nn.LeakyReLU(0.1)
            else:
                act = nn.ReLU()
            backbone_layers.append(act)

            if output_dropout > 0:
                backbone_layers.append(nn.Dropout(output_dropout))

        self.output_backbone = nn.Sequential(*backbone_layers)
        feature_dim = fc_dims[-1]

        # ---------- Quantile output heads ----------
        self.quantile_heads = nn.ModuleList([
            nn.Linear(feature_dim, output_dim)
            for _ in range(self.num_quantiles)
        ])

        # Initialize weights.
        self._init_weights()

    def _init_weights(self):
        """Initialize Linear, Conv1d, and related weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(self, x, src_key_padding_mask=None):
        """
        Args:
            x: (N, C, T)
            src_key_padding_mask: optional (N, T) mask; True/1 masks out a position
        Returns:
            output: (N, num_quantiles * output_dim)
        """
        # ------ TCN frontend ------
        if self.use_tcn and self.tcn is not None:
            x = self.tcn(x)  # (N, C_tcn, T)

        # (N, C, T) -> (N, T, C_enc)
        x = x.transpose(1, 2)

        # Input projection and positional encoding
        x = self.input_proj(x) * math.sqrt(self.d_model)  # (N, T, D)
        x = self.pos_encoding(x)                          # (N, T, D)

        # Transformer Encoder
        encoder_memory = self.encoder(
            x,
            src_key_padding_mask=src_key_padding_mask
        )  # (N, T, D)

        # Self-attention or mean pooling
        if self.pooling_type == "attention":
            # PyTorch masks out tokens where src_key_padding_mask is True.
            # SelfAttentionPooling keeps mask=1 tokens, so convert as needed.
            if src_key_padding_mask is not None:
                attn_mask = (~src_key_padding_mask).to(encoder_memory.dtype)  # True->0, False->1
            else:
                attn_mask = None
            pooled, _ = self.pool(encoder_memory, mask=attn_mask)
        else:
            # Mean pooling without using the mask
            pooled = encoder_memory.mean(dim=1)  # (N, D)

        # Output MLP
        feat = self.output_backbone(pooled)  # (N, feature_dim)

        # Quantile heads
        outs = [head(feat) for head in self.quantile_heads]  # num_q * (N, output_dim)
        output = torch.cat(outs, dim=1)  # (N, num_quantiles * output_dim)
        return output

    def get_config(self):
        return self.config.copy()


def create_transformer_encoder(**kwargs):
    """Convenience constructor for the TCN + Transformer encoder-only model."""
    return TCNTransformerEncoderRegressor(config=kwargs)

# Backward-compatible alias used by train_torch.build_model
TinyTransformer = TCNTransformerEncoderRegressor