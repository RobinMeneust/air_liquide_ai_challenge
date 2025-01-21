from torch import nn
import torch

class LSTMModel(nn.Module):
    def __init__(self, input_dim, output_dim=1, hidden_dim=32, n_layers=1, multi_steps=False):
        super().__init__()
        
        self._output_dim = output_dim
        self._multi_steps = multi_steps
        
        if output_dim < 1:
            raise ValueError("output_dim must be greater than 0")
        elif output_dim == 1:
            self._multi_steps = False
        elif self._multi_steps:
            if input_dim != 1:
                raise ValueError("input_dim must be 1 when multi_steps is True (price only as X)")
            output_dim = 1
        
        # batch_first is True because we have the shape (batch_size, ...) https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
        self._lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=n_layers, batch_first=True)
        self._linear = nn.Linear(hidden_dim, output_dim)
        
        
    def forward(self, x):
        if not self._multi_steps:
            x, _ = self._lstm(x)
            x = self._linear(x[:, -1, :]) # shape (batch_size, seq_len, output_dim) -> (batch_size, output_dim)
            return x.unsqueeze(-1) # shape (batch_size, output_dim) -> (batch_size, output_dim, 1)
        
        predictions = []
        for _ in range(self._output_dim):
            current_input = x
            x, _ = self._lstm(x)
            x = self._linear(x[:, -1, :]) # shape (batch_size, seq_len, 1) -> (batch_size, 1)
            predictions.append(x)
            
            x = torch.cat([current_input, x.unsqueeze(1)], dim=1) # add the prediction to the input
            x = x[:, 1:, :] # remove the 1st item from the input
            
        return torch.stack(predictions, dim=1)