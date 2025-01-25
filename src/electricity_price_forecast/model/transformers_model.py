from torch import nn
import torch

class TransformersModel(nn.Module):
    """Transformers model for forecasting
    
    Attributes:
        _transformer (nn.Transformer): Transformer layer
        _out (nn.Linear): Output layer
    """
    def __init__(self, input_dim, output_dim=1, num_heads=4, n_layers=1, dropout=0.1):
        """Initialize the transformers model
        
        Args:
            input_dim (int): Input dimension
            output_dim (int): Output dimension
            num_heads (int): Number of heads
            n_layers (int): Number of layers
            dropout (float): Dropout rate
        """
        super().__init__()
        
        transformers_in_out_dim = input_dim
        # self._embedding = nn.Linear(input_dim, transformer_input_dim)
        self._transformer = nn.Transformer(
            d_model=transformers_in_out_dim,
            nhead=num_heads,
            num_encoder_layers=n_layers,
            num_decoder_layers=n_layers,
            dim_feedforward=transformers_in_out_dim,
            dropout=dropout,
        )
        self._out = nn.Linear(transformers_in_out_dim, output_dim)
        
        
    def forward(self, x):
        """Forward pass
        
        Args:
            x (torch.Tensor): Input data
        
        Returns:
            torch.Tensor: Output data
        """
        # x = self._embedding(x)
        x = x.permute(1, 0, 2) # transformer expects [seq, batch, features]
        x = self._transformer(x, x)
        x = x[-1, :, :] # last output
        x = self._out(x)
        x = x.unsqueeze(-1)
        return x