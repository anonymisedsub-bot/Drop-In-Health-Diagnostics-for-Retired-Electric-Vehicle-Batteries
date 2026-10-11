# models/cnn.py
import torch
import torch.nn as nn

class CNN1D(nn.Module):
    def __init__(self, config):
        """
        CNN1D model with quantile regression heads.

        Args:
            config: Configuration dictionary with the following keys:
                conv_channels: list, output channels of each convolutional layer
                kernel_sizes: list, convolutional kernel sizes [7, 5, 3]
                paddings: list, convolutional padding [3, 2, 1]
                use_batchnorm: bool, whether to use BatchNorm
                activation: str, activation function: 'relu', 'gelu', or 'leaky_relu'
                use_global_pool: bool, whether to use global average pooling
                fc_dims: list, fully connected layer dimensions [256, 64, 8]
                use_dropout: bool, whether to apply dropout in the convolutions
                dropout_rate: float, dropout rate
                in_ch: int, input channels
                seq_len: int, sequence length
                output_dim: int, output dimension per quantile (usually 1)
                quantiles: list, quantiles [0.1, 0.5, 0.9]
        """
        super().__init__()

        # All model settings are supplied by config.py.
        self.config = dict(config)

        # Parse the configuration.
        conv_channels = self.config['conv_channels']
        kernel_sizes  = self.config['kernel_sizes']
        paddings      = self.config['paddings']
        use_batchnorm = self.config['use_batchnorm']
        activation    = self.config['activation']
        use_global_pool = self.config['use_global_pool']
        fc_dims       = self.config['fc_dims']
        use_dropout   = self.config['use_dropout']
        dropout_rate  = self.config['dropout_rate']
        in_ch         = self.config['in_ch']
        seq_len       = self.config['seq_len']
        output_dim    = self.config['output_dim']
        quantiles     = self.config['quantiles']

        # Quantile settings
        self.quantiles = quantiles
        self.num_quantiles = len(quantiles)

        # Validate the configuration.
        assert len(conv_channels) == len(kernel_sizes) == len(paddings), \
            "conv_channels, kernel_sizes, and paddings must have equal lengths"

        # ----- Build the convolutional backbone. -----
        conv_layers = []
        current_channels = in_ch

        for out_channels, ksize, pad in zip(conv_channels, kernel_sizes, paddings):
            conv_layers.append(
                nn.Conv1d(current_channels, out_channels,
                          kernel_size=ksize, padding=pad)
            )

            if use_batchnorm:
                conv_layers.append(nn.BatchNorm1d(out_channels))

            if activation == 'relu':
                conv_layers.append(nn.ReLU())
            elif activation == 'gelu':
                conv_layers.append(nn.GELU())
            elif activation == 'leaky_relu':
                conv_layers.append(nn.LeakyReLU(0.1))
            else:
                conv_layers.append(nn.ReLU())

            if use_dropout and dropout_rate > 0:
                conv_layers.append(nn.Dropout(dropout_rate))

            current_channels = out_channels

        if use_global_pool:
            conv_layers.append(nn.AdaptiveAvgPool1d(1))
            conv_out_dim = current_channels          # (N, C, 1) -> C
        else:
            conv_out_dim = current_channels * seq_len  # (N, C, T) -> C*T

        self.conv_net = nn.Sequential(*conv_layers)
        self.use_global_pool = use_global_pool
        self.seq_len = seq_len

        # ----- Build the FC backbone, which outputs features rather than y. -----
        # Input dimension equals the flattened convolutional output dimension.
        fc_all_dims = [conv_out_dim] + fc_dims
        fc_layers = []
        for i in range(len(fc_all_dims) - 1):
            in_d  = fc_all_dims[i]
            out_d = fc_all_dims[i + 1]
            fc_layers.append(nn.Linear(in_d, out_d))

            # The final layer produces features and can omit the activation.
            if i < len(fc_all_dims) - 1:
                if activation == 'relu':
                    fc_layers.append(nn.ReLU())
                elif activation == 'gelu':
                    fc_layers.append(nn.GELU())
                elif activation == 'leaky_relu':
                    fc_layers.append(nn.LeakyReLU(0.1))
                else:
                    fc_layers.append(nn.ReLU())

        self.fc_backbone = nn.Sequential(*fc_layers)
        feature_dim = fc_all_dims[-1]

        # ----- Quantile regression heads: one head per quantile. -----
        self.quantile_heads = nn.ModuleList([
            nn.Linear(feature_dim, output_dim) for _ in range(self.num_quantiles)
        ])

        # Initialize weights.
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        """
        x: (N, C, T)
        Returns: quantile predictions with shape (N, num_quantiles)
        """
        # Convolutional backbone
        h = self.conv_net(x)
        h = h.flatten(1)  # Flatten to (N, -1), with or without global pooling.

        # FC backbone
        feat = self.fc_backbone(h)

        # Quantile heads
        outs = [head(feat) for head in self.quantile_heads]  # Each has shape (N, 1).
        out = torch.cat(outs, dim=1)  # (N, num_quantiles)

        return out

    def get_config(self):
        return self.config.copy()

def create_cnn1d_model(**kwargs):
    return CNN1D(config=kwargs)
