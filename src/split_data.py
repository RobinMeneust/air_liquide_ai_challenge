import pandas as pd

def create_train_test_time_series(data: pd.DataFrame, horizon_hours: int) -> (pd.DataFrame, pd.DataFrame):
    """
    Sépare une série temporelle en jeu d'entraînement (train) et jeu de test (test)
    sur la base d'un horizon passé en paramètre (en heures).

    Paramètres
    ----------
    data : pd.DataFrame ou pd.Series
        Jeu de données contenant votre série temporelle (index temporel).
    horizon_hours : int
        Nombre d'heures à conserver pour le test.
        Doit être parmi [6, 12, 24, 48, 72, 168].

    Retourne
    -------
    train : pd.DataFrame ou pd.Series
        Partie d'entraînement : toutes les observations avant l'horizon spécifié.
    test : pd.DataFrame ou pd.Series
        Partie de test : les dernières `horizon_hours` observations.
    """
    # Vérification du paramètre
    if horizon_hours not in [6, 12, 24, 48, 72, 168]:
        raise ValueError("Le paramètre horizon_hours doit appartenir à [6, 12, 24, 48, 72, 168].")

    # Vérification d'avoir suffisamment de données
    if horizon_hours >= len(data):
        raise ValueError(f"Le dataset est trop court ({len(data)}) pour un horizon de {horizon_hours} heures.")

    # S'assurer que l'index est trié dans l'ordre chronologique
    data = data.sort_index()

    # Séparation
    train = data.iloc[:-horizon_hours]
    test = data.iloc[-horizon_hours:]

    return train, test
