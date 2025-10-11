import pandas as pd
import os

def charger_csv(chemin_fichier, separateur=',', low_memory=False):
    """
    Charge un fichier CSV en un DataFrame pandas.

    Paramètres :
        chemin_fichier (str) : Chemin complet ou relatif vers le fichier CSV.
        separateur (str) : Caractère séparateur des colonnes. Par défaut ','.
        low_memory (bool) : Option pour la gestion mémoire lors du chargement.

    Retourne :
        pd.DataFrame : Contenu du fichier CSV.
    """
    if not os.path.exists(chemin_fichier):
        raise FileNotFoundError(f"Le fichier '{chemin_fichier}' est introuvable.")

    return pd.read_csv(chemin_fichier, sep=separateur, low_memory=low_memory)

def fusionner_dataframes(df1, df2, on, how='left', supprimer_doublons=True, subset_df1=None, subset_df2=None):
    """
    Fusionne deux DataFrames sur une ou plusieurs colonnes.
    Permet de spécifier les colonnes pour supprimer les doublons.
    """
    if supprimer_doublons:
        df1 = df1.drop_duplicates(subset=subset_df1 or on)
        df2 = df2.drop_duplicates(subset=subset_df2 or on)

    return df1.merge(df2, on=on, how=how)
  