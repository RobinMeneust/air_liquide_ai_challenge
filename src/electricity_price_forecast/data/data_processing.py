import torch
from torch.utils.data import Subset
from sklearn.preprocessing import MinMaxScaler
import os
import zipfile
import pandas as pd

# Décompresser le fichier data.zip
def decompress_zip(file_path, extract_to):
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

# Charger les fichiers .parquet
def load_parquet_files(directory):
    data_frames = {}
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.parquet'):
                file_path = os.path.join(root, file)
                df = pd.read_parquet(file_path)
                data_frames[file] = df
    return data_frames

def transform_synthetic_data_optimized(df, start_date='2016-12-31 23:00:00+00:00'):
    base_time = pd.to_datetime(start_date, utc=True)
    # Calculer le nombre de semaines écoulées basé sur la position dans le DataFrame
    weeks = df.index // (7 * 24)  # 7 jours * 24 heures
    df['date'] = base_time + \
                 pd.to_timedelta(weeks, unit='W') + \
                 pd.to_timedelta(df['dayofweek'], unit='D') + \
                 pd.to_timedelta(df['hourofday'], unit='h')
    return df.set_index('date')[['price']]

def process_synthetic_files(data_frames, start_date='2016-12-31 23:00:00+00:00'):
    processed_data = {}
    for file_name, df in data_frames.items():
        if 'dayofweek' in df.columns and 'hourofday' in df.columns:
            processed_data[file_name] = transform_synthetic_data_optimized(df, start_date)
        else:
            processed_data[file_name] = df
    return processed_data

def init_extraction(path: str = 'CYTECH_AirLiquide.zip'):
    # Chemin vers le fichier ZIP et le dossier d'extraction
    zip_file_path = path
    extracted_folder = 'data'

    # Étapes principales
    decompress_zip(zip_file_path, extracted_folder)
    df_synth = load_parquet_files(extracted_folder)
    df_hist = df_synth.pop('hourly_day_ahead_prices_2017_2020.parquet')
    df_synth = process_synthetic_files(df_synth)
    return df_synth, df_hist

# Alternative versions used by some notebooks:

def preprocess_true_data(df, start_date="2016-12-31 00:00:00+00:00"):
    reference_date = pd.to_datetime(start_date)
    new_df = df.copy()
    new_df['date'] = new_df.index
    new_df['dayofweek'] = new_df.index.dayofweek
    new_df['hourofday'] = new_df.index.hour
    new_df['dayofyear'] = new_df.index.dayofyear
    new_df['dayofseries'] = (new_df.index - reference_date).days
    new_df['dayofmonth'] = new_df.index.day  
    new_df['month'] = new_df.index.month
    new_df['year'] = new_df.index.year
    new_df['hourofseries'] = new_df['dayofseries'] * 24 + new_df['hourofday'] - (new_df.index[0].hour - reference_date.hour) # starts at 0
    new_df.reset_index(drop=True, inplace=True)    
    return new_df

def preprocess_synthetic_data(df, start_date="2016-12-31 00:00:00+00:00"):
    new_df = transform_synthetic_data_optimized(df, start_date)
    return preprocess_true_data(new_df, start_date)


class DataNormalizer():
    def __init__(self):
        self._scalers = {}
       
    def transform_df(self, df, columns=None):
        new_df = df.copy()
        if columns is None:
            columns = new_df.columns
            columns = columns.drop('date') # date should not be normalized
        
        for column_name in columns:
            self._scalers[column_name] = MinMaxScaler()
            new_df[column_name] = self._scalers[column_name].fit_transform(new_df[[column_name]])
        
        return new_df
    
    def inverse_transform_numpy(self, numpy_data, column_name):
        if column_name not in self._scalers:
           raise ValueError("Unknown column name or uninitialized scaler (transform_df needs to be run before)")
        return self._scalers[column_name].inverse_transform(numpy_data)
    
    
def get_splits(dataset, test_size, val_ratio):
    train_split_size = len(dataset) - test_size

    val_split_size = int(train_split_size * val_ratio)
    train_split_size -= val_split_size

    test_start_idx = train_split_size + val_split_size

    train_indices = range(train_split_size)
    val_indices = range(train_split_size, test_start_idx)
    test_indices = range(test_start_idx, len(dataset))

    train_split = Subset(dataset, train_indices)
    
    if val_ratio == 0:
        val_split = None
    else:
        val_split = Subset(dataset, val_indices)
    
    if test_size == 0:
        test_split = None
    else:
        test_split = Subset(dataset, test_indices)
    
    return train_split, val_split, test_split


def get_predict_data(test_split: Subset):
    test_indices = test_split.indices
    test_indices = test_split.dataset.windowIndicesToPointIndices(test_indices)
    last_window_start_idx, last_window_end_idx = test_split.dataset.get_last_window_idx(test_indices)
    last_window_end_idx += 1 # include the last index but in slice it is exclusive
        
    predict_dates = test_split.dataset.get_dates(test_indices)[last_window_start_idx:]
    predict_y = test_split.dataset.get_y(test_indices)[last_window_start_idx:]
    if type(predict_y) == torch.Tensor:
        predict_y = predict_y.cpu().numpy()
    predict_x = test_split.dataset.get_X(test_indices)[last_window_start_idx:last_window_end_idx]
        
    return predict_dates, predict_y, predict_x