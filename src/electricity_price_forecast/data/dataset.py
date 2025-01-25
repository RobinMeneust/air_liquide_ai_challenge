import torch
from torch.utils.data import Dataset
import numpy as np

class DatasetWithWindow(Dataset):
    """Dataset with window sliding for time series forecasting
    
    Attributes:
        X (torch.Tensor): Input data
        y (torch.Tensor): Target data
        window_size (int): Size of the window
        window_step (int): Step of the window
        horizon (int): Horizon of the forecast
        length (int): Length of the dataset
    """
    def __init__(self, df, window_size, window_step, horizon, x_keys, y_key, device="cuda", tensor=True):
        """Initialize the dataset
        
        Args:
            df (pd.DataFrame): Dataframe containing the data
            window_size (int): Size of the window
            window_step (int): Step of the window
            horizon (int): Horizon of the forecast
            x_keys (list): List of keys for the input data
            y_key (str): Key for the target data
            device (str): Device to use
            tensor (bool): Whether to convert the data to tensors
        """
        if tensor:
            self.X = torch.tensor(df[x_keys].values, dtype=torch.float32).to(device)
            self.y = torch.tensor(df[y_key].values, dtype=torch.float32).unsqueeze(-1).to(device)
            if type(x_keys) != type([]) or len(x_keys) == 1:
                self.X = self.X.unsqueeze(-1)
        else:
            self.X = np.array(df[x_keys].values)
            self.y = np.array(df[y_key].values).reshape(-1, 1)
        self._dates = df["date"].values
        
        self.window_size = window_size
        self.window_step = window_step
        self.horizon = horizon
        self.length = (len(self.X) - self.window_size - self.horizon) // self.window_step + 1

    def __len__(self):
        """Return the length of the dataset
        
        Returns:
            int: Length of the dataset
        """
        return self.length

    def __getitem__(self, index):
        """Get the item at the given index
        
        Args:
            index (int): Index of the item
        
        Returns:
            tuple: Tuple containing the window and the target
        """
        start_window = index * self.window_step
        end_window = start_window + self.window_size
        end_horizon = end_window + self.horizon
        
        if end_horizon > len(self.X) or end_horizon > len(self.y):
            raise IndexError("Index out of range")
        
        window = self.X[start_window:end_window]
        target = self.y[end_window:end_horizon]
        return window, target
    
    def get_last_window_idx(self, indices):
        """Get the start and end index of the last window
        
        Args:
            indices (list): List of indices to consider. WARNING: this is not the indices of the dataset (windows) but the indices of the points in the initial dataset
            
        Returns:
            tuple: Tuple containing the start and end index of the last window
        """
        return len(indices) - self.window_size - self.horizon, len(indices) - self.horizon - 1

    def get_dates(self, indices):
        """Get the dates corresponding to the indices
        
        Args:
            indices (list): List of indices to return. WARNING: this is not the indices of the dataset (windows) but the indices of the points in the initial dataset
        
        Returns:
            np.ndarray: Dates corresponding to the indices
        """
        return self._dates[indices]
    
    def get_X(self, indices):
        """Get the input data corresponding to the indices
        
        Args:
            indices (list): List of indices to return. WARNING: this is not the indices of the dataset (windows) but the indices of the points in the initial dataset
            
        Returns:
            np.ndarray: Input data corresponding to the indices
        """
        return self.X[indices]
    
    def get_y(self, indices):
        """Get the target data corresponding to the indices
        
        Args:
            indices (list): List of indices to return. WARNING: this is not the indices of the dataset (windows) but the indices of the points in the initial dataset
            
        Returns:
            np.ndarray: Target data corresponding to the indices
        """
        return self.y[indices]

    def windowIndicesToPointIndices(self, window_indices):
        """Convert window indices to point indices
        
        Args:
            window_indices (list): List of window indices
        
        Returns:
            np.ndarray: Point indices corresponding to the window indices
        """
        new_indices = []
        for window_idx in window_indices:
            start = window_idx * self.window_step
            new_indices += list(range(start, start + self.window_size + self.horizon))
        
        return np.unique(new_indices)