import os
import pandas as pd

def test_processed_data_exists():
    """Vérifie si le fichier final issu de la Couche 1 est bien présent."""
    data_path = "data/processed/final.csv"
    assert os.path.exists(data_path), f"Le fichier {data_path} est introuvable. Lancez d'abord le pipeline de données !"

def test_target_column_present():
    """Vérifie que la colonne cible cruciale est bien présente dans le dataset."""
    data_path = "data/processed/final.csv"
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        expected_target = "target_readmission_risk"
        assert expected_target in df.columns, f"La colonne cible '{expected_target}' est manquante !"

def test_no_null_values_in_target():
    """S'assure qu'aucune valeur de la cible n'est nulle (leakage ou corruption)."""
    data_path = "data/processed/final.csv"
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        expected_target = "target_readmission_risk"
        if expected_target in df.columns:
            assert df[expected_target].isnull().sum() == 0, "Le dataset contient des valeurs nulles dans la colonne cible !"