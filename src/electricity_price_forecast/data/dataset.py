import torch
from torch.utils.data import Dataset
import numpy as np

class DatasetWithWindow(Dataset):
    def __init__(self, df, window_size, window_step, horizon, x_keys, y_key, device="cuda", tensor=True):
        if tensor:
            self.X = torch.tensor(df[x_keys].values, dtype=torch.float32).to(device)
            self.y = torch.tensor(df[y_key].values, dtype=torch.float32).unsqueeze(-1).to(device)
            if type(x_keys) != type([]) or len(x_keys) == 1:
                self.X = self.X.unsqueeze(-1)
        else:
            self.X = np.array(df[x_keys].values)
            self.y = np.array(df[y_key].values).reshape(-1, 1)
        self._dates = df.index.values
        
        self.window_size = window_size
        self.window_step = window_step
        self.horizon = horizon
        self.length = (len(self.X) - self.window_size - self.horizon) // self.window_step + 1

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        start_window = index * self.window_step
        end_window = start_window + self.window_size
        end_horizon = end_window + self.horizon
        
        if end_horizon > len(self.X) or end_horizon > len(self.y):
            raise IndexError("Index out of range")
        
        window = self.X[start_window:end_window]
        target = self.y[end_window:end_horizon]
        return window, target
    
    def get_last_window_idx(self, indices):
        return len(indices) - self.window_size - self.horizon

    def get_dates(self, indices):
        return self._dates[indices]
    
    def get_X(self, indices):
        return self.X[indices]
    
    def get_y(self, indices):
        return self.y[indices]
