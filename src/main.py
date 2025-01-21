import pandas as pd
import matplotlib.pyplot as plt
from src.extract_data import init_extraction

# Afficher une série temporelle graphiquement et sauvegarder en .png
def plot_time_series(data_frame, title, file_name, start_date=None, end_date=None):
    # Convertir les dates en objets Timestamp
    if start_date:
        start_date = pd.Timestamp(start_date)
    if end_date:
        end_date = pd.Timestamp(end_date)
    
    # Vérifier et ajuster les dates de début et de fin
    if start_date and end_date:
        if start_date > end_date:
            start_date, end_date = end_date, start_date
        if start_date < data_frame.index.min():
            start_date = data_frame.index.min()
        if end_date > data_frame.index.max():
            end_date = data_frame.index.max()
        data_frame = data_frame[start_date:end_date]

    plt.figure(figsize=(10, 5))
    plt.plot(data_frame.index, data_frame['price'])
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.grid(True)
    plt.savefig(f"image/{file_name}.png")  # Sauvegarder le plot en .png
    plt.show()

if __name__ == "__main__":
    # Initialiser l'extraction et charger les fichiers .parquet
    df_synth, df_hist = init_extraction()
    # afficher le nombre de dataframes
    print(f"Nombre de DataFrames synth: {len(df_synth)}")
    print("Nombre de DataFrames hist: 1")

    # Afficher les 10 premières lignes du DataFrame historique
    print("\nDataFrame historique:")
    print(df_hist.head(10))

    # Afficher les 10 premières lignes des 10 premiers DataFrames synthétiques
    for i, (file_name, df) in enumerate(df_synth.items()):
        if i < 1:
            print(f"\nnom: {file_name}")
            print(df.head(10000))

    # Afficher une série temporelle pour un DataFrame spécifique
    # Spécifier le nom du fichier ou la position
    file_name = 'hourly_day_ahead_prices_2017_2020.parquet'  # Remplacer par le nom du fichier souhaité
    file_position = None  # Remplacer par la position souhaitée (ex: 0 pour le premier fichier)

    # Spécifier les dates de début et de fin
    start_date = '2017-01-01 00:00:00+00:00'  # Remplacer par la date de début souhaitée
    end_date = '2025-01-31 23:00:00+00:00'  # Remplacer par la date de fin souhaitée

    if file_name in df_synth:
        plot_time_series(df_synth[file_name], f"Série temporelle pour {file_name}", file_name, start_date, end_date)
    elif file_name == 'hourly_day_ahead_prices_2017_2020.parquet':
        plot_time_series(df_hist, f"Série temporelle pour {file_name}", file_name, start_date, end_date)
    elif file_position is not None and 0 <= file_position < len(df_synth):
        file_key = list(df_synth.keys())[file_position]
        plot_time_series(df_synth[file_key], f"Série temporelle pour {file_key}", file_key, start_date, end_date)
    else:
        print("Fichier non trouvé ou position invalide")