"""
src/models/train.py - Registre Multi-Modèles Avancé & Comparaison (XGBoost vs LightGBM)
"""

import argparse
import logging
import os
import json
import joblib
import pandas as pd
import numpy as np

import xgboost as xgb
import lightgbm as lgb
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_recall_curve

# Configuration du logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

def append_to_experiment_history(output_dir, model_name, params, metrics, thresholds):
    """Consigne l'historique global de chaque exécution de paramètre dans un registre continu."""
    history_file = os.path.join(output_dir, "experiment_history.json")
    history = []
    
    if os.path.exists(history_file):
        with open(history_file, "r") as f:
            try:
                history = json.load(f)
            except json.JSONDecodeError:
                history = []
                
    run_data = {
        "model_type": model_name,
        "parameters": params,
        "metrics": metrics,
        "thresholds": thresholds
    }
    history.append(run_data)
    
    with open(history_file, "w") as f:
        json.dump(history, f, indent=4)
    log.info(f"-> Historique local mis à jour pour {model_name} : {history_file}")

def evaluate_and_optimize_thresholds(model, X_test, y_test):
    """Calcule les probabilités et détermine les seuils optimaux pour le F1-Score et le Recall."""
    # Gestion de la prédiction des probabilités selon le format de l'objet du modèle
    y_prob = model.predict_proba(X_test)[:, 1]
    precisions, recalls, class_thresholds = precision_recall_curve(y_test, y_prob)
    
    # Seuil Optimal pour le F1-Score
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_f1_idx = np.argmax(f1_scores)
    threshold_f1 = float(class_thresholds[best_f1_idx]) if best_f1_idx < len(class_thresholds) else 0.5
    
    # Seuil Optimal pour maximiser le Recall (Objectif minimum de 85% si possible)
    valid_recalls = np.where(recalls >= 0.85)[0]
    best_recall_idx = valid_recalls[-1] if len(valid_recalls) > 0 else 0
    threshold_recall = float(class_thresholds[best_recall_idx]) if best_recall_idx < len(class_thresholds) else 0.5
    
    # Évaluation sous le seuil F1
    preds_f1 = (y_prob >= threshold_f1).astype(int)
    metrics_f1 = {
        "accuracy": float(accuracy_score(y_test, preds_f1)),
        "recall": float(recall_score(y_test, preds_f1)),
        "f1": float(f1_score(y_test, preds_f1))
    }
    
    # Évaluation sous le seuil Recall
    preds_recall = (y_prob >= threshold_recall).astype(int)
    metrics_recall = {
        "accuracy": float(accuracy_score(y_test, preds_recall)),
        "recall": float(recall_score(y_test, preds_recall)),
        "f1": float(f1_score(y_test, preds_recall))
    }
    
    return metrics_f1, metrics_recall, {"f1": threshold_f1, "recall": threshold_recall}

def train_model(data_path, max_depth, n_estimators, learning_rate):
    if not os.path.exists(data_path):
        log.error(f"Fichier de données introuvable : {data_path}")
        return

    df = pd.read_csv(data_path)
    
    TARGET = "target_readmission_risk"
    feature_cols = [c for c in df.columns if c not in [TARGET, "target_is_high_cost", "total_cost", "DESYNPUF_ID"]]
    
    X = df[feature_cols]
    y = df[TARGET]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Sauvegarde systématique des noms de variables (requis pour l'API de déploiement)
    joblib.dump(feature_cols, os.path.join(output_dir, "feature_names.pkl"))
    
    mlflow.set_experiment("Healthcare_MultiModel_Comparison")
    
    # --------------------------------------------------------------------------
    # MODÈLE 1 : ENTRAÎNEMENT & TRACKING XGBOOST
    # --------------------------------------------------------------------------
    with mlflow.start_run(run_name="XGBoost_Run"):
        log.info("--- Entraînement de XGBoost ---")
        xgb_params = {"max_depth": max_depth, "n_estimators": n_estimators, "learning_rate": learning_rate}
        mlflow.log_params(xgb_params)
        
        xgb_model = xgb.XGBClassifier(**xgb_params, random_state=42, eval_metric="logloss")
        xgb_model.fit(X_train, y_train)
        
        xgb_metrics_f1, xgb_metrics_recall, xgb_thresholds = evaluate_and_optimize_thresholds(xgb_model, X_test, y_test)
        mlflow.log_metrics(xgb_metrics_f1)
        mlflow.xgboost.log_model(xgb_model, "model")
        
        # Sauvegarde du fichier spécifique au modèle
        joblib.dump(xgb_model, os.path.join(output_dir, "xgb.pkl"))
        append_to_experiment_history(output_dir, "XGBoost", xgb_params, xgb_metrics_f1, xgb_thresholds)

    # --------------------------------------------------------------------------
    # MODÈLE 2 : ENTRAÎNEMENT & TRACKING LIGHTGBM
    # --------------------------------------------------------------------------
    with mlflow.start_run(run_name="LightGBM_Run"):
        log.info("--- Entraînement de LightGBM ---")
        lgb_params = {"max_depth": max_depth, "n_estimators": n_estimators, "learning_rate": learning_rate, "verbose": -1}
        mlflow.log_params(lgb_params)
        
        lgb_model = lgb.LGBMClassifier(**lgb_params, random_state=42)
        lgb_model.fit(X_train, y_train)
        
        lgb_metrics_f1, lgb_metrics_recall, lgb_thresholds = evaluate_and_optimize_thresholds(lgb_model, X_test, y_test)
        mlflow.log_metrics(lgb_metrics_f1)
        mlflow.lightgbm.log_model(lgb_model, "model")
        
        # Sauvegarde du fichier spécifique au modèle
        joblib.dump(lgb_model, os.path.join(output_dir, "lgbm.pkl"))
        append_to_experiment_history(output_dir, "LightGBM", lgb_params, lgb_metrics_f1, lgb_thresholds)

    # --------------------------------------------------------------------------
    # SÉLECTION AUTOMATIQUE ET CRÉATION DU REGISTRE PRO (BEST MODEL)
    # --------------------------------------------------------------------------
    log.info("--- Sélection et Sauvegarde du Meilleur Modèle ---")
    
    # Comparaison basée sur le score F1 global
    if xgb_metrics_f1["f1"] >= lgb_metrics_f1["f1"]:
        best_model = xgb_model
        best_name = "XGBoost"
        best_params = xgb_params
        best_metrics_f1 = xgb_metrics_f1
        best_metrics_recall = xgb_metrics_recall
        best_thresholds = xgb_thresholds
    else:
        best_model = lgb_model
        best_name = "LightGBM"
        best_params = lgb_params
        best_metrics_f1 = lgb_metrics_f1
        best_metrics_recall = lgb_metrics_recall
        best_thresholds = lgb_thresholds

    log.info(f"🏆 Le meilleur modèle identifié est : {best_name} (F1-Score: {best_metrics_f1['f1']:.4f})")

    # Génération exacte de l'arborescence demandée
    joblib.dump(best_model, os.path.join(output_dir, "best_model.pkl"))
    joblib.dump(best_model, os.path.join(output_dir, "best_recall_model.pkl"))
    
    with open(os.path.join(output_dir, "best_model_info.json"), "w") as f:
        json.dump({"selected_architecture": best_name, "metrics_at_optimal_f1": best_metrics_f1}, f, indent=4)
        
    with open(os.path.join(output_dir, "best_recall_model_info.json"), "w") as f:
        json.dump({"selected_architecture": best_name, "metrics_at_optimal_recall": best_metrics_recall}, f, indent=4)
        
    with open(os.path.join(output_dir, "best_params.json"), "w") as f:
        json.dump(best_params, f, indent=4)
        
    with open(os.path.join(output_dir, "best_threshold.json"), "w") as f:
        json.dump({"optimal_f1_threshold": best_thresholds["f1"]}, f, indent=4)
        
    with open(os.path.join(output_dir, "threshold.json"), "w") as f:
        json.dump({"optimal_recall_threshold": best_thresholds["recall"]}, f, indent=4)

    log.info(f"Registre de production entièrement mis à jour avec succès dans le dossier '{output_dir}/' !")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/processed/final.csv", help="Path to processed data")
    parser.add_argument("--depth", type=int, default=5, help="Tree maximum depth")
    parser.add_argument("--estimators", type=int, default=100, help="Number of estimators")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate")
    args = parser.parse_args()

    train_model(args.input, args.depth, args.estimators, args.lr)