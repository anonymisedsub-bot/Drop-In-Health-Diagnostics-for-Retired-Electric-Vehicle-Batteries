"""MLP with a retained q50 slot for MSE capacity point prediction."""
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, config):
        """
        MLP backbone with legacy output slots; the active q50 slot uses MSE.
        
        Args:
            config: Configuration dictionary with the following keys:
                hidden_dims: list, dimensions of the hidden layers
                activation: str, activation function: 'relu', 'gelu', or 'leaky_relu'
                dropout: float, dropout rate
                use_batchnorm: bool, whether to use BatchNorm
                quantiles: list, quantiles [0.1, 0.5, 0.9]
                in_ch: int, input channels
                seq_len: int, sequence length
                output_dim: int, output dimension per quantile (usually 1)
        """
        super().__init__()
        
        # All model settings are supplied by config.py.
        self.config = dict(config)
        
        # Parse the configuration.
        hidden_dims = self.config['hidden_dims']
        activation = self.config['activation']
        dropout = self.config['dropout']
        use_batchnorm = self.config['use_batchnorm']
        quantiles = self.config['quantiles']
        in_ch = self.config['in_ch']
        seq_len = self.config['seq_len']
        output_dim = self.config['output_dim']
        
        # Input dimension
        in_dim = in_ch * seq_len
        
        # Build the network layers.
        layers = []
        dims = [in_dim] + hidden_dims
        
        for i in range(len(dims) - 1):
            # Linear layer
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            
            # Activation function
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'gelu':
                layers.append(nn.GELU())
            elif activation == 'leaky_relu':
                layers.append(nn.LeakyReLU(0.1))
            else:
                layers.append(nn.ReLU())  # Default to ReLU.
            
            # Optional BatchNorm
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(dims[i + 1]))
            
            # Dropout, except after the final layer
            if i < len(dims) - 2 and dropout > 0:
                layers.append(nn.Dropout(dropout))
        
        self.backbone = nn.Sequential(*layers)
        
        # Quantile output heads: one output layer per quantile
        self.num_quantiles = len(quantiles)
        self.quantiles = quantiles
        
        # Create an independent output layer for each quantile.
        self.quantile_heads = nn.ModuleList([
            nn.Linear(dims[-1], output_dim) for _ in range(self.num_quantiles)
        ])
        
        # Initialize weights.
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: input tensor with shape (N, C, T)
            
        Returns:
            output: tensor with shape (N, num_quantiles), one prediction per quantile
        """
        # Flatten input: (N, C, T) -> (N, C*T).
        x = x.flatten(1)
        
        # Apply the backbone network.
        features = self.backbone(x)
        
        # Predict each quantile independently.
        quantile_outputs = []
        for head in self.quantile_heads:
            quantile_outputs.append(head(features))
        
        # Concatenate quantile outputs: (N, num_quantiles).
        output = torch.cat(quantile_outputs, dim=1)
        
        return output
    
    def get_config(self):
        """Return the model configuration."""
        return self.config.copy()
