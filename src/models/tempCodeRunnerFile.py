"""
src/models/train.py

Layer 2 — Experimentation & Tracking (MLflow)
Trains an XGBoost Classifier to predict patient readmission risk,
logs parameters and evaluation metrics to MLflow, and saves the model artifact.
"""

import argparse
import logging
import os
import sys
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, classification_report
import xgboost as xgb
import mlflow
import mlflow.xgboost

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

def train_model(data_path: str, max_depth: int, n_estimators: int, learning_rate: float):
    """Loads engineered data, trains an XGBoost model, and logs everything to MLflow."""
    
    # 1. Load the preprocessed dataset
    if not os.path.exists(data_path):
        log.error(f"Processed data not found at {data_path}. Please run your pipeline first!")
        sys.exit(1)
        
    log.info(f"Loading preprocessed features from {data_path}...")
    df = pd.read_csv(data_path)

    # 2. DYNAMICALLY separate target from features
    TARGET_COLUMN = "target_readmission_risk"
    
    if TARGET_COLUMN not in df.columns:
        log.error(f"Target column '{TARGET_COLUMN}' missing from dataset!")
        sys.exit(1)
        
    # Drop alternative targets to avoid data leakage if present
    drop_cols = [TARGET_COLUMN, "target_is_high_cost"]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    
    X = df[feature_cols]
    y = df[TARGET_COLUMN]

    log.info(f"Features shape: {X.shape} | Target distribution: \n{y.value_counts(normalize=True)}")

    # 3. Train/Test Split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. Initialize MLflow Experiment Tracking
    mlflow.set_experiment("Healthcare_Readmission_Prediction")
    
    with mlflow.start_run():
        log.info("MLflow logging initialized. Starting training...")
        
        # Log Hyperparameters to MLflow
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("features_count", X.shape[1])

        # 5. Initialize and train the XGBoost Classifier
        model = xgb.XGBClassifier(
            max_depth=max_depth,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=42,
            eval_metric="logloss"
        )
        
        model.fit(X_train, y_train)
        log.info("Model training complete. Running evaluation...")

        # 6. Evaluate Model Performance
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="binary")
        
        # Safely handle ROC-AUC evaluation if only one class is present in split
        try:
            auc = roc_auc_score(y_test, y_prob)
        except ValueError:
            auc = 0.5
            log.warning("Only one class present in y_test. ROC-AUC score defaulted to 0.5.")

        log.info(f"Results -> Accuracy: {acc:.4f} | F1-Score: {f1:.4f} | ROC-AUC: {auc:.4f}")

        # Log Metrics to MLflow
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", auc)

        # Print classification matrix overview for the logs
        print("\n--- Classification Report ---")
        print(classification_report(y_test, y_pred))

        # 7. Log the Model Artifact itself
        log.info("Saving model artifact to MLflow registry...")
        mlflow.xgboost.log_model(model, artifact_path="model")
        
        log.info("MLflow tracking run complete successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/processed/final.csv", help="Path to processed data")
    parser.add_argument("--depth", type=int, default=5, help="XGBoost maximum tree depth")
    parser.add_argument("--estimators", type=int, default=100, help="Number of gradient boosted trees")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate / shrinkage factor")
    args = parser.parse_args()

    train_model(
        data_path=os.path.normpath(args.input),
        max_depth=args.depth,
        n_estimators=args.estimators,
        learning_rate=args.lr
    )