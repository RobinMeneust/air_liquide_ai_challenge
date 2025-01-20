import torch
from torch.utils.data import Dataset
import numpy as np

class DatasetWithWindow(Dataset):
    def __init__(self, df, window_size, window_step, horizon, x_keys, y_key, device="cuda"):
        self.X = torch.tensor(df[x_keys].values, dtype=torch.float32).to(device)
        self.y = torch.tensor(df[y_key].values, dtype=torch.float32).unsqueeze(-1).to(device)
                
        if len(x_keys) == 1:
            self.X = self.X.unsqueeze(-1)
                        
        self.window_size = window_size
        self.window_step = window_step
        self.horizon = horizon
        self.length = (len(self.X) - window_size - self.horizon) // window_step + 1
        # print(f"length: {self.length}, window_size: {window_size}, window_step: {window_step}, len(self.X): {len(self.X)}")

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
    
    def get_ground_truth(self, start, end):
        if end > len(self.X) or start < 0:
            raise IndexError("Index out of range")
        
        hours = np.arange(start, end)
        prices = self.y[start:end].cpu().numpy().flatten()
        
        return hours, prices
        
