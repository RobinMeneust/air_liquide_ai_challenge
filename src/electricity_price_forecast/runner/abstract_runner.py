import os
import pandas as pd
from tqdm import tqdm
import logging
from electricity_price_forecast.data.data_processing import preprocess_true_data, preprocess_synthetic_data, DataNormalizer
from abc import ABC, abstractmethod


class AbstractRunner(ABC):
    """Abstract class for the electricity price forecast task
    
    Attributes:
        _max_synthetic_data_fetched (int): Maximum number of synthetic data fetched
        _model_name (str): Name of the model
        _tested_horizons (list): List of tested horizons
        _window_size (int): Size of the window
        _window_step (int): Step of the window
        _data_root_path (str): Path to the data root
        _save_path_root (str): Path to the save root
    """
    def __init__(self, model_name: str):
        """Initialize the AbstractRunner object
        
        Args:
            model_name (str): Name of the model
        """
        logging.getLogger("lightning.pytorch").setLevel(logging.WARNING)
        self._max_synthetic_data_fetched = 10 # a 1:10 factor of data augmentation is already a lot
        self._model_name = model_name
        
        self._tested_horizons = [6, 12, 24, 48, 72, 168]
        self._window_size = 72
        self._window_step = 24
        
        current_file_path = os.path.abspath(__file__)
        root_path = current_file_path.split("src")[0]
        current_date = pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M-%S")
        self._data_root_path = os.path.join(root_path, "data")
        self._save_path_root = os.path.join(root_path, "results", model_name, current_date)
        
        os.makedirs(self._save_path_root, exist_ok=True)

    def load_true_data(self, data_normalizer: DataNormalizer=None):
        """Load the true data
        
        Args:
            data_normalizer (DataNormalizer): DataNormalizer object to normalize the data
        
        Returns:
            pd.DataFrame: Preprocessed DataFrame
        """
        data_path = os.path.join(self._data_root_path, "donnees historiques", "prix", "hourly_day_ahead_prices_2017_2020.parquet")
        df_price = pd.read_parquet(data_path)
        preprocessed_df = preprocess_true_data(df_price)
        
        if data_normalizer is None:
            return preprocessed_df
        return data_normalizer.transform_df(preprocessed_df)

    @abstractmethod
    def get_best_params(self, datamodule, horizon, n_trials=50):
        """Get the best parameters for the model
        
        Args:
            datamodule (DataModule): DataModule object
            horizon (int): Horizon of the forecast
            n_trials (int): Number of trials for the optimization
        Returns:
            dict: Dictionary containing the best parameters
        """
        pass

    @abstractmethod
    def train_model(self, datamodule, horizon, early_stopping=True, lr=0.001, n_epochs=50, hidden_dim=32, n_layers=1, device="cuda"):
        """Train the model
        
        Args:
            datamodule (DataModule): DataModule object
            horizon (int): Horizon of the forecast (number of points to forecast)
            early_stopping (bool): Whether to use early stopping
            lr (float): Learning rate
            n_epochs (int): Number of epochs
            hidden_dim (int): Hidden dimension
            n_layers (int): Number of layers
            device (str): Device to use
        
        Returns:
            nn.Module: Trained model
        """
        pass
    
    def load_synthetic_data(self, path, max_num_fetched=None, data_normalizer=None, initial_df=None):
        """Load the synthetic data
        
        Args:
            path (str): Path to the synthetic data
            max_num_fetched (int): Maximum number of files fetched
            data_normalizer (DataNormalizer): DataNormalizer object to normalize the data
        
        Returns:
            pd.DataFrame: Preprocessed DataFrame
        """
        if max_num_fetched is not None and max_num_fetched < 0:
            max_num_fetched = None
        train_all = []
        
        i = 0
        for filename in tqdm(os.listdir(path)):
            if max_num_fetched is not None and i >= max_num_fetched:
                break
            i += 1
            if filename.endswith(".parquet"):
                current_df = pd.read_parquet(os.path.join(path, filename))
                current_df = preprocess_synthetic_data(current_df)
                                
                train_all.append(current_df)
        
        
        if initial_df is not None:
            train_all.append(initial_df) # must be at the end (test is at the end)
        

        train_all = pd.concat(train_all)
        
        if data_normalizer is not None:
            train_all = data_normalizer.transform_df(train_all)
        
        return train_all
    
    @abstractmethod
    def run_all(self, params=None):
        """Run all tests of a model
        
        Args:
            params (dict): Parameters for the model
        """
        pass
        
