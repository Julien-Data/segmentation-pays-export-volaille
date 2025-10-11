# 1_preparation_donnees.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from load_data import charger_csv, fusionner_dataframes

def charger_donnees():
    dispo = charger_csv("DisponibiliteAlimentaire_2017.csv")
    population = charger_csv("Population_2000_2018.csv")
    df_fintech = charger_csv("GlobalFindexDatabase2025.csv")
    return dispo, population, df_fintech

def nettoyer_et_fusionner(dispo, population):
    """
    Nettoie et fusionne les DataFrames 'dispo' et 'population'.
    Supprime les doublons avant fusion.
    """
    cles_jointure = ['Code zone', 'Code année']
    subset_dispo = ['Code zone', 'Code Produit', 'Code Élément', 'Code année']
    subset_pop = ['Code zone', 'Code année']

    df_data = fusionner_dataframes(
        dispo, population,
        on=cles_jointure,
        how='left',
        subset_df1=subset_dispo,
        subset_df2=subset_pop
    )

    df_data = df_data.drop(columns=['Note'], errors='ignore')
    return df_data


def filtrer_poulet(df_data):
    df_poulet = df_data[df_data['Produit_x'].str.contains("poulet|volaille", case=False, na=False)]
    return df_poulet

def construire_table_pays(df_poulet):
    table_pays = df_poulet.pivot_table(
        index='Zone_x',
        columns='Élément_x',
        values='Valeur_x',
        aggfunc='sum'
    ).reset_index()

    # Remplacement des NaN
    table_pays['Exportations - Quantité'] = table_pays['Exportations - Quantité'].fillna(0)
    table_pays['Importations - Quantité'] = table_pays['Importations - Quantité'].fillna(0)

    for col in table_pays.columns[2:]:
        table_pays[col] = table_pays[col].fillna(table_pays[col].mean())

    return table_pays

def standardiser_donnees(table_pays):
    if 'Alimentation pour touristes' in table_pays.columns:
        X = table_pays.drop(columns=['Zone_x', 'Alimentation pour touristes'])
    else:
        X = table_pays.drop(columns=['Zone_x'])
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X, X_scaled











