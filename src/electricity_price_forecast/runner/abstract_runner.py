import os
import pandas as pd
from tqdm import tqdm
import logging
from electricity_price_forecast.data.data_processing import preprocess_true_data, preprocess_synthetic_data, DataNormalizer
from abc import ABC, abstractmethod


class AbstractRunner(ABC):
    def __init__(self, model_name: str):
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
        data_path = os.path.join(self._data_root_path, "donnees historiques", "prix", "hourly_day_ahead_prices_2017_2020.parquet")
        df_price = pd.read_parquet(data_path)
        preprocessed_df = preprocess_true_data(df_price)
        
        if data_normalizer is None:
            return preprocessed_df
        return data_normalizer.transform_df(preprocessed_df)

    @abstractmethod
    def get_best_params(self, datamodule, horizon, n_trials=50):
        pass

    @abstractmethod
    def train_model(self, datamodule, horizon, early_stopping=True, lr=0.001, n_epochs=50, hidden_dim=32, n_layers=1, device="cuda"):
        pass
    
    def load_synthetic_data(self, path, max_num_fetched=None, data_normalizer=None, initial_df=None):
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
        pass
        
