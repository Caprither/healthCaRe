import os
import shutil
import pytest
import pandas as pd
from src.models.train import train_model

@pytest.fixture
def dummy_data_setup():
    """Crée un mini-dataset temporaire pour tester l'entraînement sans charger le gros fichier."""
    temp_data_path = "data/processed/temp_test_data.csv"
    os.makedirs("data/processed", exist_ok=True)
    
    # Création de données factices minimales conformes aux features attendues
    dummy_data = {
        "feature_1": [0.1, 0.5, 0.9, 0.2, 0.4] * 4,
        "feature_2": [1, 2, 3, 4, 5] * 4,
        "target_readmission_risk": [0, 1, 0, 1, 0] * 4
    }
    df = pd.DataFrame(dummy_data)
    df.to_csv(temp_data_path, index=False)
    
    yield temp_data_path
    
    # Nettoyage après le test
    if os.path.exists(temp_data_path):
        os.remove(temp_data_path)

def test_train_pipeline_execution(dummy_data_setup):
    """Vérifie que le script train_model s'exécute sans erreur et génère les bons fichiers."""
    temp_csv = dummy_data_setup
    
    # Exécution de l'entraînement avec des hyperparamètres rapides (smoke test)
    try:
        train_model(data_path=temp_csv, max_depth=2, n_estimators=5, learning_rate=0.1)
    except Exception as e:
        pytest.fail(f"Le script d'entraînement a planté lors du test : {e}")

    # Vérification stricte de l'arborescence demandée dans 'output'
    assert os.path.exists("output/xgb.pkl"), "Le fichier 'xgb.pkl' n'a pas été généré !"
    assert os.path.exists("output/lgbm.pkl"), "Le fichier 'lgbm.pkl' n'a pas été généré !"
    assert os.path.exists("output/best_model.pkl"), "Le fichier 'best_model.pkl' n'a pas été généré !"
    assert os.path.exists("output/feature_names.pkl"), "Le fichier 'feature_names.pkl' n'a pas été généré !"
    assert os.path.exists("output/experiment_history.json"), "Le registre historique global est manquant !"