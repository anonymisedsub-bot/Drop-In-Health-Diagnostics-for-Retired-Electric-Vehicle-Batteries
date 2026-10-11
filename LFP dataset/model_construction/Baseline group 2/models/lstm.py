import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


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
    """TCN residual block: (Conv1d -> Chomp -> activation -> Dropout) x2 + residual."""
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout,
                 activation_type: str = "relu"):
        super().__init__()
        padding = (kernel_size - 1) * dilation  # Standard TCN padding.

        self.conv1 = weight_norm(
            nn.Conv1d(
                in_ch,
                out_ch,
                kernel_size,
                stride=1,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp1 = Chomp1d(padding)
        self.activation1 = self._make_activation(activation_type)
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(
            nn.Conv1d(
                out_ch,
                out_ch,
                kernel_size,
                stride=1,
                padding=padding,
                dilation=dilation,
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


# ====================== TCN + LSTM regression model ======================

class LSTMReg(nn.Module):
    def __init__(self, config):
        """TCN + LSTM regression model with quantile outputs.

        Architecture:
        (N, C, T) --[TCN]*--> (N, C_tcn, T) --[LSTM]*--> (N, H_last)
            --[FC backbone]--> (N, feature_dim) --[quantile heads]--> (N, num_quantiles * output_dim)
        """
        super().__init__()

        # All settings come from the adjacent public config.py.
        self.config = dict(config)

        # Read configuration.
        hidden_dims      = self.config["hidden_dims"]
        num_layers       = self.config["num_layers"]
        bidirectional    = self.config["bidirectional"]
        lstm_dropout_p   = self.config["lstm_dropout"]
        use_batchnorm    = self.config["use_batchnorm"]
        lstm_activation  = self.config["lstm_activation"]

        fc_dims          = self.config["fc_dims"]
        fc_activation    = self.config["fc_activation"]
        fc_dropout_p     = self.config["fc_dropout"]
        use_fc_dropout   = self.config["use_fc_dropout"]

        in_ch            = self.config["in_ch"]
        output_dim       = self.config["output_dim"]
        quantiles        = self.config["quantiles"]

        # TCN configuration
        self.use_tcn     = self.config["use_tcn"]
        tcn_channels     = self.config["tcn_channels"]
        tcn_kernel_size  = self.config["tcn_kernel_size"]
        tcn_dropout      = self.config["tcn_dropout"]
        tcn_activation   = self.config["tcn_activation"]

        # Quantile settings
        self.quantiles = list(quantiles)
        self.num_quantiles = len(self.quantiles)

        # Validate LSTM configuration.
        assert len(hidden_dims) == num_layers, (
            f"hidden_dims length ({len(hidden_dims)}) must equal num_layers ({num_layers})"
        )

        # ---------- TCN frontend ----------
        if self.use_tcn:
            self.tcn = TemporalConvNet(
                num_inputs=in_ch,
                channels=tcn_channels,
                kernel_size=tcn_kernel_size,
                dropout=tcn_dropout,
                activation_type=tcn_activation,
            )
            lstm_input_dim = tcn_channels[-1]
        else:
            self.tcn = None
            lstm_input_dim = in_ch

        # ---------- LSTM backbone ----------
        self.lstm_layers = nn.ModuleList()
        current_dim = lstm_input_dim

        for i in range(num_layers):
            lstm_layer = nn.LSTM(
                input_size=current_dim,
                hidden_size=hidden_dims[i],
                num_layers=1,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=0.0,  # Apply dropout between layers explicitly.
            )
            self.lstm_layers.append(lstm_layer)
            current_dim = hidden_dims[i] * (2 if bidirectional else 1)

        # Dropout between LSTM layers
        self.lstm_dropout = nn.Dropout(lstm_dropout_p) if lstm_dropout_p > 0 else nn.Identity()

        # LSTM nonlinear activation
        if lstm_activation == "relu":
            self.lstm_activation = nn.ReLU()
        elif lstm_activation == "gelu":
            self.lstm_activation = nn.GELU()
        elif lstm_activation == "leaky_relu":
            self.lstm_activation = nn.LeakyReLU(0.1)
        else:
            self.lstm_activation = nn.ReLU()

        # BatchNorm after LSTM output
        if use_batchnorm:
            self.batchnorm = nn.BatchNorm1d(current_dim)
        else:
            self.batchnorm = nn.Identity()

        # ---------- Fully connected backbone ----------
        fc_all_dims = [current_dim] + fc_dims
        fc_layers = []
        for i in range(len(fc_all_dims) - 1):
            in_d  = fc_all_dims[i]
            out_d = fc_all_dims[i + 1]
            fc_layers.append(nn.Linear(in_d, out_d))

            if fc_activation == "relu":
                act = nn.ReLU()
            elif fc_activation == "gelu":
                act = nn.GELU()
            elif fc_activation == "leaky_relu":
                act = nn.LeakyReLU(0.1)
            else:
                act = nn.ReLU()
            fc_layers.append(act)

            if use_fc_dropout and fc_dropout_p > 0:
                fc_layers.append(nn.Dropout(fc_dropout_p))

        self.fc_backbone = nn.Sequential(*fc_layers)
        feature_dim = fc_all_dims[-1]

        # ---------- Quantile output heads ----------
        self.quantile_heads = nn.ModuleList([
            nn.Linear(feature_dim, output_dim) for _ in range(self.num_quantiles)
        ])

        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # Initialize weights.
        self._init_weights()

    def _init_weights(self):
        """Initialize LSTM, Linear, Conv1d, and BatchNorm weights."""
        for module in self.modules():
            if isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if "weight_ih" in name:
                        nn.init.xavier_uniform_(param.data)
                    elif "weight_hh" in name:
                        nn.init.orthogonal_(param.data)
                    elif "bias" in name:
                        param.data.fill_(0.0)
                        # Set forget-gate bias to 1.
                        n = param.size(0)
                        param.data[n // 4:n // 2].fill_(1.0)

            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)

            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(self, x):
        """Forward pass.

        Args:
            x: (N, C, T) input tensor

        Returns:
            out: (N, num_quantiles * output_dim)
        """
        # ------ TCN frontend ------
        if self.use_tcn and self.tcn is not None:
            x = self.tcn(x)  # (N, C_tcn, T)

        # ------ LSTM backbone ------
        h = x.transpose(1, 2)  # (N, T, C_in_for_lstm)

        for i, lstm_layer in enumerate(self.lstm_layers):
            h, _ = lstm_layer(h)  # (N, T, H_i)
            if i < self.num_layers - 1:
                h = self.lstm_activation(h)
                h = self.lstm_dropout(h)

        # Apply a nonlinear activation after the final LSTM layer as well.
        h = self.lstm_activation(h)

        # Select the final time step.
        h_last = h[:, -1, :]  # (N, H_last)

        # Optional BatchNorm
        h_last = self.batchnorm(h_last)

        # ------ FC backbone and quantile heads ------
        feat = self.fc_backbone(h_last)  # (N, feature_dim)

        outs = [head(feat) for head in self.quantile_heads]  # num_q * (N, output_dim)
        out = torch.cat(outs, dim=1)  # (N, num_quantiles * output_dim)

        return out

    def get_config(self):
        return self.config.copy()


def create_lstm_model(**kwargs):
    """Convenience constructor for the TCN+LSTM model."""
    return LSTMReg(config=kwargs)

