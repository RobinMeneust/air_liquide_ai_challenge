from torch import nn

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, n_layers=1):
        super().__init__()
        # batch_first is True because we have the shape (batch_size, ...) https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
        self._lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=n_layers, batch_first=True)
        self._linear = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        x, _ = self._lstm(x)
        x = self._linear(x)
        return x