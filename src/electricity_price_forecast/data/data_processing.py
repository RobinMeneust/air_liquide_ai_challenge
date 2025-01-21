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
