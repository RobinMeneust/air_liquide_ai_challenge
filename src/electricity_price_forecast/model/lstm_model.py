from torch import nn
import torch

from enum import Enum

class MULTI_STEP(Enum):
    """Enum for the different multi-step strategies.
    REPEAT: Repeat the model output as input (step by step). However the date is not updated (e.g. day+1) so we didn't actually used it
    ALL_AT_ONCE: Predict all the steps at once.
    ENCODER_DECODER: Use an encoder-decoder architecture.
    """
    REPEAT = 0
    ALL_AT_ONCE = 1
    ENCODER_DECODER = 2

class LSTMModel(nn.Module):
    """LSTM model for forecasting
    
    Attributes:
        _output_dim (int): Output dimension
        _multi_step (MULTI_STEP): Multi-step strategy
        _lstm (nn.LSTM): LSTM layer
        _linear (nn.Linear): Linear layer
    """
    def __init__(self, input_dim, output_dim=1, hidden_dim=32, n_layers=1, multi_step: MULTI_STEP = MULTI_STEP.ALL_AT_ONCE):
        """Initialize the LSTM model
        
        Args:
            input_dim (int): Input dimension
            output_dim (int): Output dimension
            hidden_dim (int): Hidden dimension
            n_layers (int): Number of layers
            multi_step (MULTI_STEP): Multi-step strategy
        """
        super().__init__()
        
        self._output_dim = output_dim
        self._multi_step = multi_step
        
        if output_dim < 1:
            raise ValueError("output_dim must be greater than 0")
        elif output_dim == 1:
            if self._multi_step == MULTI_STEP.REPEAT:
                self._multi_steps = MULTI_STEP.ALL_AT_ONCE
                print("INFO: output_dim is 1, multi_step REPEAT is disabled. Was set automatically to ALL_AT_ONCE")
        
        if self._multi_step == MULTI_STEP.REPEAT:
            output_dim = 1
        
        if self._multi_step == MULTI_STEP.ALL_AT_ONCE or self._multi_step == MULTI_STEP.REPEAT:
            # batch_first is True because we have the shape (batch_size, ...) https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
            if self._multi_step == MULTI_STEP.REPEAT and input_dim > 1:
                raise ValueError("input_dim must be 1 when multi_step is REPEAT")
            self._lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=n_layers, batch_first=True)
        elif self._multi_step == MULTI_STEP.ENCODER_DECODER:
            self._encoder_lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=n_layers, batch_first=True)
            self._decoder_lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=n_layers, batch_first=True)
        else:
            raise ValueError("multi_step must be one of the values in MULTI_STEP")
        
        self._linear = nn.Linear(hidden_dim, output_dim)
        
        
    def forward(self, x):
        """Forward pass
        
        Args:
            x (torch.Tensor): Input data
        
        Returns:
            torch.Tensor: Output data
        """
        if self._multi_step == MULTI_STEP.ALL_AT_ONCE:
            x, _ = self._lstm(x)
            x = self._linear(x[:, -1, :]) # x[:, -1, :] : shape (batch_size, seq_len, output_dim) -> (batch_size, output_dim)
            return x.unsqueeze(-1) # shape (batch_size, output_dim) -> (batch_size, output_dim, 1)
        elif self._multi_step == MULTI_STEP.REPEAT:        
            predictions = []
            for _ in range(self._output_dim):
                current_input = x
               
                x, _ = self._lstm(x)
                x = self._linear(x[:, -1, :]) # x[:, -1, :] : last output (shape (batch_size, seq_len, 1) -> (batch_size, 1))
                predictions.append(x)
                
                x = torch.cat([current_input, x.unsqueeze(1)], dim=1) # add the prediction to the input
                x = x[:, 1:, :] # remove the 1st item from the input
                
            return torch.stack(predictions, dim=1)
        elif self._multi_step == MULTI_STEP.ENCODER_DECODER:
            x, _ = self._encoder_lstm(x)
            x = x[:, -1, :].unsqueeze(1).repeat(1, self._output_dim, 1) # shape (batch_size, hidden_dim) -> (batch_size, output_dim, hidden_dim)
            x, _ = self._decoder_lstm(x)
            x = self._linear(x)
            return x[:, -1, :].unsqueeze(-1)
        else:
            raise ValueError("multi_step must be one of the values in MULTI_STEP")