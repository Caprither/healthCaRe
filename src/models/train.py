"""
src/models/train.py - Registre Multi-Modèles Avancé & Comparaison (XGBoost vs LightGBM)
                    + Simulation Opérationnelle 5 Hôpitaux (Couche Métier)
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
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score, precision_score,
    roc_auc_score, precision_recall_curve
)

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

def evaluate_and_optimize_thresholds(
    model,
    X_test,
    y_test,
    recall_target: float = 0.85,
    min_precision_for_recall: float = 0.10,
):
    """
    Calcule les probabilités et détermine les seuils optimaux pour le F1-Score et le Recall.

    Améliorations vs version originale :
    ─────────────────────────────────────
    • F1 optimal : recherche exhaustive sur TOUS les seuils candidats de la courbe
      PR (sklearn) plutôt que de s'appuyer sur np.argmax qui peut souffrir de
      ties ou d'instabilité aux extrémités de la courbe.
    • Recall optimal : parmi tous les seuils atteignant `recall_target` (défaut 85 %),
      on retient celui qui maximise le F1 **sous contrainte de Recall**, au lieu de
      prendre aveuglément le dernier index. Si aucun seuil n'atteint la cible,
      on bascule gracieusement sur le seuil qui maximise le Recall brut.
    • Garde-fou `min_precision_for_recall` : évite de retenir un seuil qui classe
      presque tout le monde comme positif (precision proche de 0), ce qui serait
      cliniquement inutilisable.
    • Métriques enrichies : ajout de `precision` et `roc_auc` pour chaque régime,
      utiles pour le suivi MLflow et la sélection du meilleur modèle.
    • Logging détaillé pour chaque seuil retenu.

    Args:
        model                     : Modèle entraîné exposant predict_proba().
        X_test                    : Features du jeu de test.
        y_test                    : Labels vrais du jeu de test.
        recall_target (float)     : Recall minimum visé (défaut 0.85 / 85 %).
        min_precision_for_recall  : Precision minimale acceptée quand on optimise
                                    le Recall, pour éviter les seuils dégénérés.

    Returns:
        metrics_f1     (dict) : Métriques au seuil optimal F1.
        metrics_recall (dict) : Métriques au seuil optimal Recall.
        thresholds     (dict) : Valeurs numériques des deux seuils retenus.
    """
    # ── 1. Probabilités & courbe Precision-Recall ────────────────────────────
    y_prob = model.predict_proba(X_test)[:, 1]
    precisions, recalls, pr_thresholds = precision_recall_curve(y_test, y_prob)
    # Note : sklearn retourne len(thresholds) = len(precisions) - 1
    # Le dernier point (threshold → 0) n'a pas de seuil associé ; on le tronque.
    precisions_c = precisions[:-1]
    recalls_c    = recalls[:-1]

    # ── 2. Seuil optimal F1 ──────────────────────────────────────────────────
    # Calcul vectorisé du F1 sur toute la courbe PR, stable même si P+R = 0.
    f1_curve = np.where(
        (precisions_c + recalls_c) > 0,
        2 * (precisions_c * recalls_c) / (precisions_c + recalls_c),
        0.0
    )
    best_f1_idx   = int(np.argmax(f1_curve))
    threshold_f1  = float(pr_thresholds[best_f1_idx])

    log.info(
        f"  [Seuil F1]     threshold={threshold_f1:.4f} | "
        f"F1={f1_curve[best_f1_idx]:.4f} | "
        f"Recall={recalls_c[best_f1_idx]:.4f} | "
        f"Precision={precisions_c[best_f1_idx]:.4f}"
    )

    # ── 3. Seuil optimal Recall (sous contrainte médicale) ───────────────────
    # Candidats : tous les points de la courbe qui atteignent recall_target
    # ET qui respectent la contrainte minimale de precision.
    candidate_mask = (recalls_c >= recall_target) & (precisions_c >= min_precision_for_recall)
    valid_indices  = np.where(candidate_mask)[0]

    if len(valid_indices) > 0:
        # Parmi les candidats valides, on maximise le F1 (recall contraint).
        best_constrained_idx = valid_indices[np.argmax(f1_curve[valid_indices])]
        threshold_recall     = float(pr_thresholds[best_constrained_idx])
        log.info(
            f"  [Seuil Recall] threshold={threshold_recall:.4f} | "
            f"Recall={recalls_c[best_constrained_idx]:.4f} ≥ {recall_target:.0%} ✓ | "
            f"F1={f1_curve[best_constrained_idx]:.4f} | "
            f"Precision={precisions_c[best_constrained_idx]:.4f}"
        )
    else:
        # Fallback : aucun seuil n'atteint la cible → on maximise le Recall brut.
        best_recall_idx  = int(np.argmax(recalls_c))
        threshold_recall = float(pr_thresholds[best_recall_idx])
        log.warning(
            f"  [Seuil Recall] Cible {recall_target:.0%} non atteignable avec "
            f"precision ≥ {min_precision_for_recall:.0%}. "
            f"Fallback → Recall max = {recalls_c[best_recall_idx]:.4f} "
            f"au seuil {threshold_recall:.4f}"
        )

    # ── 4. Évaluation complète au seuil F1 ──────────────────────────────────
    preds_f1   = (y_prob >= threshold_f1).astype(int)
    metrics_f1 = {
        "accuracy"  : float(accuracy_score(y_test, preds_f1)),
        "precision" : float(precision_score(y_test, preds_f1, zero_division=0)),
        "recall"    : float(recall_score(y_test, preds_f1, zero_division=0)),
        "f1"        : float(f1_score(y_test, preds_f1, zero_division=0)),
        "roc_auc"   : float(roc_auc_score(y_test, y_prob)),
    }

    # ── 5. Évaluation complète au seuil Recall ───────────────────────────────
    preds_recall   = (y_prob >= threshold_recall).astype(int)
    metrics_recall = {
        "accuracy"  : float(accuracy_score(y_test, preds_recall)),
        "precision" : float(precision_score(y_test, preds_recall, zero_division=0)),
        "recall"    : float(recall_score(y_test, preds_recall, zero_division=0)),
        "f1"        : float(f1_score(y_test, preds_recall, zero_division=0)),
        "roc_auc"   : float(roc_auc_score(y_test, y_prob)),
    }

    thresholds = {"f1": threshold_f1, "recall": threshold_recall}
    return metrics_f1, metrics_recall, thresholds

# ==============================================================================
# NOUVELLE FONCTIONNALITÉ : SIMULATION OPÉRATIONNELLE 5 HÔPITAUX
# ==============================================================================

def generate_hospital_report(best_model, X_test, best_thresholds, output_dir, random_seed=42):
    """
    Simule la distribution des patients du jeu de test dans 5 hôpitaux fictifs
    et génère un rapport de capacité métier basé sur le seuil optimal de Recall.

    Le seuil de Recall est privilégié par prudence médicale :
    mieux vaut réserver un lit inutile que d'en manquer un pour un patient à risque.

    Args:
        best_model   : Le modèle gagnant (XGBoost ou LightGBM) déjà entraîné.
        X_test       : DataFrame des features du jeu de test.
        best_thresholds (dict) : Dictionnaire contenant les clés 'f1' et 'recall'.
        output_dir   : Dossier de sortie pour le fichier JSON généré.
        random_seed  : Graine aléatoire pour la reproductibilité de la répartition.

    Output:
        Crée `output/hospital_capacity_report.json` avec les KPIs par hôpital.
    """
    log.info("--- Génération du Rapport de Simulation Opérationnelle (5 Hôpitaux) ---")

    # ---- 1. Récupération du seuil de Recall (approche médicalement prudente) ----
    recall_threshold = best_thresholds["recall"]
    log.info(f"Seuil de Recall utilisé pour la simulation : {recall_threshold:.4f}")

    # ---- 2. Prédiction des probabilités de réadmission sur X_test ----
    y_prob = best_model.predict_proba(X_test)[:, 1]
    y_pred_recall = (y_prob >= recall_threshold).astype(int)

    # ---- 3. Répartition aléatoire reproductible des patients dans 5 hôpitaux ----
    hospital_names = ["Hôpital A", "Hôpital B", "Hôpital C", "Hôpital D", "Hôpital E"]
    rng = np.random.default_rng(seed=random_seed)
    n_patients = len(X_test)

    # Attribution d'un hôpital à chaque patient (distribution uniforme)
    hospital_assignments = rng.choice(hospital_names, size=n_patients, replace=True)

    # ---- 4. Construction du DataFrame de simulation ----
    sim_df = pd.DataFrame({
        "hospital": hospital_assignments,
        "readmission_predicted": y_pred_recall
    })

    # ---- 5. Agrégation des KPIs par hôpital ----
    hospital_report = {}

    for hospital in hospital_names:
        hospital_data = sim_df[sim_df["hospital"] == hospital]

        total_patients = int(len(hospital_data))
        predicted_readmissions = int(hospital_data["readmission_predicted"].sum())

        # Taux de réadmission : arrondi à 4 décimales pour la lisibilité dashboard
        readmission_rate = round(predicted_readmissions / total_patients, 4) if total_patients > 0 else 0.0

        # Lits préventifs à réserver : 1 lit par réadmission prédite (règle métier)
        preventive_beds_required = predicted_readmissions

        hospital_report[hospital] = {
            "total_patients_received": total_patients,
            "predicted_readmissions": predicted_readmissions,
            "estimated_readmission_rate": readmission_rate,
            "preventive_beds_to_reserve": preventive_beds_required
        }

        log.info(
            f"  {hospital} | Patients: {total_patients} | "
            f"Réadmissions prédites: {predicted_readmissions} | "
            f"Taux: {readmission_rate:.2%} | "
            f"Lits à réserver: {preventive_beds_required}"
        )

    # ---- 6. Ajout des métadonnées globales de la simulation ----
    simulation_metadata = {
        "simulation_parameters": {
            "total_test_patients": n_patients,
            "number_of_hospitals": len(hospital_names),
            "threshold_type_used": "optimal_recall",
            "recall_threshold_value": recall_threshold,
            "random_seed": random_seed
        },
        "hospital_capacity_report": hospital_report
    }

    # ---- 7. Sauvegarde du rapport JSON ----
    report_path = os.path.join(output_dir, "hospital_capacity_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(simulation_metadata, f, indent=4, ensure_ascii=False)

    log.info(f"✅ Rapport de capacité hospitalière généré avec succès : {report_path}")
    return simulation_metadata

# ==============================================================================
# PIPELINE PRINCIPAL (INCHANGÉ + APPEL À LA SIMULATION)
# ==============================================================================

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

    # --------------------------------------------------------------------------
    # SIMULATION OPÉRATIONNELLE : RAPPORT DE CAPACITÉ HOSPITALIÈRE
    # --------------------------------------------------------------------------
    generate_hospital_report(
        best_model=best_model,
        X_test=X_test,
        best_thresholds=best_thresholds,
        output_dir=output_dir,
        random_seed=42
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/processed/final.csv", help="Path to processed data")
    parser.add_argument("--depth", type=int, default=5, help="Tree maximum depth")
    parser.add_argument("--estimators", type=int, default=100, help="Number of estimators")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate")
    args = parser.parse_args()

    train_model(args.input, args.depth, args.estimators, args.lr)