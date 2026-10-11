# models/mlp.py
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, config):
        """
        MLP model supporting quantile regression.
        
        Args:
            config: configuration dictionary with the following keys:
                hidden_dims: list of hidden-layer widths
                activation: str, activation function: 'relu', 'gelu', or 'leaky_relu'
                dropout: float, dropout probability
                use_batchnorm: bool, whether to use BatchNorm
                quantiles: list of quantiles, e.g. [0.1, 0.5, 0.9]
                in_ch: int, number of input channels
                seq_len: int, sequence length
                output_dim: int, output width per quantile (usually 1)
        """
        super().__init__()
        
        # All settings come from the adjacent public config.py.
        self.config = dict(config)
        
        # Read configuration.
        hidden_dims = self.config['hidden_dims']
        activation = self.config['activation']
        dropout = self.config['dropout']
        use_batchnorm = self.config['use_batchnorm']
        quantiles = self.config['quantiles']
        in_ch = self.config['in_ch']
        seq_len = self.config['seq_len']
        output_dim = self.config['output_dim']
        
        # Input width
        in_dim = in_ch * seq_len
        
        # Build network layers.
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
            
            # Dropout except after the final layer
            if i < len(dims) - 2 and dropout > 0:
                layers.append(nn.Dropout(dropout))
        
        self.backbone = nn.Sequential(*layers)
        
        # Quantile heads: one output layer per quantile.
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
            x: (N, C, T) input tensor
            
        Returns:
            output: (N, num_quantiles) tensor, one prediction per quantile
        """
        # Flatten input: (N, C, T) -> (N, C*T).
        x = x.flatten(1)
        
        # Run the backbone network.
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

# Convenience constructor
def create_mlp_model(**kwargs):
    """
    Convenience constructor for an MLP model.
    
    Args:
        **kwargs: complete model configuration from config.py
        
    Returns:
        An MLP model instance.
    """
    return MLP(config=kwargs)