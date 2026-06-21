import os
import json
import joblib
import pandas as pd
import time
import uuid
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field
# 🚨 FIX 1: Added 'Any' to the typing imports
from typing import Dict, List, Any 
from contextlib import asynccontextmanager

# 🚨 IMPORTATIONS PROMETHEUS
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

# ==========================================
# DÉFINITION DES MÉTRIQUES PROMETHEUS
# ==========================================
HTTP_REQUESTS_TOTAL = Counter(
    'healthcare_api_requests_total', 
    'Nombre total de requêtes HTTP reçues',
    ['method', 'endpoint', 'status_code']
)

REQUEST_LATENCY = Histogram(
    'healthcare_api_request_latency_seconds',
    'Temps de réponse de l\'API en secondes',
    ['endpoint']
)

MODEL_PREDICTIONS_TOTAL = Counter(
    'healthcare_model_predictions_total',
    'Nombre total de prédictions segmentées par décision',
    ['readmission_predite', 'recommandation_hopital']
)

# Variables globales pour stocker le modèle
MODEL = None
FEATURE_NAMES = None
THRESHOLD = 0.5

# ==========================================
# GESTION DU DÉMARRAGE
# ==========================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL, FEATURE_NAMES, THRESHOLD
    try:
        MODEL = joblib.load("output/best_model.pkl")
        FEATURE_NAMES = joblib.load("output/feature_names.pkl")
        
        with open("output/threshold.json", "r") as f:
            threshold_data = json.load(f)
            THRESHOLD = threshold_data.get("optimal_recall_threshold", 0.5)
            
        print(f"✅ Artefacts MLOps chargés avec succès ! Seuil appliqué : {THRESHOLD:.3f}")
    except Exception as e:
        print(f"❌ Erreur lors du chargement des artefacts : {e}")
        
    yield 
    print("🛑 Arrêt du serveur MLOps.")

app = FastAPI(
    title="🏥 Hôpital API - MLOps",
    description="API de prédiction du risque de réadmission et gestion des lits.",
    version="1.1.0",
    lifespan=lifespan 
)

# ==========================================
# 1. SCHÉMAS DE DONNÉES
# ==========================================
class PatientData(BaseModel):
    features: Dict[str, float] = Field(
        ..., 
        json_schema_extra={
            "example": {"age": 65.0, "feature_1": 0.5, "feature_2": 3.0}
        }
    )

# ==========================================
# 2. ENDPOINTS
# ==========================================

@app.get("/", include_in_schema=False)
def redirect_to_docs():
    return RedirectResponse(url="/docs")

@app.get("/metrics", tags=["Supervision"])
def metrics():
    HTTP_REQUESTS_TOTAL.labels(method="GET", endpoint="/metrics", status_code="200").inc()
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/health", tags=["Système"])
def health_check():
    status_code = "200" if MODEL is not None else "500"
    HTTP_REQUESTS_TOTAL.labels(method="GET", endpoint="/health", status_code=status_code).inc()
    
    if MODEL is not None:
        return {"status": "en_ligne", "modele_charge": True}
    return {"status": "hors_ligne", "modele_charge": False}

@app.post("/predict", tags=["Prédiction Individuelle"])
def predict_readmission(data: PatientData):
    """(Existant) Évalue un seul patient."""
    start_time = time.time() 
    try:
        df_patient = pd.DataFrame([data.features])
        
        for col in FEATURE_NAMES:
            if col not in df_patient.columns:
                df_patient[col] = 0.0 
                
        df_patient = df_patient[FEATURE_NAMES]
        probability = float(MODEL.predict_proba(df_patient)[0, 1])
        is_high_risk = probability >= THRESHOLD
        action = "Réserver un lit préventif" if is_high_risk else "Décharge standard (Pas de lit requis)"
        
        MODEL_PREDICTIONS_TOTAL.labels(readmission_predite=str(is_high_risk), recommandation_hopital=action).inc()
        duration = time.time() - start_time
        REQUEST_LATENCY.labels(endpoint="/predict").observe(duration)
        HTTP_REQUESTS_TOTAL.labels(method="POST", endpoint="/predict", status_code="200").inc()
        
        return {
            "patient_id": str(uuid.uuid4())[:8],
            "risque_calcule": round(probability, 4),
            "seuil_alerte": round(THRESHOLD, 4),
            "readmission_predite": is_high_risk,
            "recommandation_hopital": action
        }
        
    except Exception as e:
        HTTP_REQUESTS_TOTAL.labels(method="POST", endpoint="/predict", status_code="400").inc()
        raise HTTPException(status_code=400, detail=str(e))

# 🚨 FIX 2: Changed Dict[str, float] to Dict[str, Any]
@app.post("/predict_batch", tags=["Prédiction par Lot (Airflow)"])
def predict_batch(patients: List[Dict[str, Any]]):
    """Reçoit un lot de patients depuis Airflow, exécute le modèle en bloc et retourne les prédictions."""
    start_time = time.time()
    
    try:
        df_batch = pd.DataFrame(patients)
        
        # S'assurer que toutes les colonnes d'entraînement sont présentes pour tout le lot
        for col in FEATURE_NAMES:
            if col not in df_batch.columns:
                df_batch[col] = 0.0 
                
        df_batch = df_batch[FEATURE_NAMES]
        
        # Prédiction vectorisée (très rapide pour des centaines de lignes)
        probabilities = MODEL.predict_proba(df_batch)[:, 1]
        
        results = []
        for prob in probabilities:
            is_high_risk = bool(prob >= THRESHOLD)
            action = "Réserver un lit préventif" if is_high_risk else "Décharge standard"
            
            # Enregistrer les décisions dans Prometheus
            MODEL_PREDICTIONS_TOTAL.labels(readmission_predite=str(is_high_risk), recommandation_hopital=action).inc()
            
            results.append({
                "patient_id": str(uuid.uuid4())[:8],
                "risque_calcule": round(float(prob), 4),
                "readmission_predite": is_high_risk
            })
            
        # Enregistrer la latence
        duration = time.time() - start_time
        REQUEST_LATENCY.labels(endpoint="/predict_batch").observe(duration)
        HTTP_REQUESTS_TOTAL.labels(method="POST", endpoint="/predict_batch", status_code="200").inc()
        
        return {"batch_size": len(results), "predictions": results}
        
    except Exception as e:
        HTTP_REQUESTS_TOTAL.labels(method="POST", endpoint="/predict_batch", status_code="400").inc()
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/hospitals/capacity", tags=["Gestion des Lits (Métier)"])
def get_hospital_capacity_report():
    report_path = "output/hospital_capacity_report.json"
    if not os.path.exists(report_path):
        HTTP_REQUESTS_TOTAL.labels(method="GET", endpoint="/hospitals/capacity", status_code="404").inc()
        raise HTTPException(status_code=404, detail="Le rapport n'existe pas.")
        
    try:
        with open(report_path, "r", encoding="utf-8") as f:
            report_data = json.load(f)
            
        HTTP_REQUESTS_TOTAL.labels(method="GET", endpoint="/hospitals/capacity", status_code="200").inc()
        return {"message": "Rapport de capacité généré avec succès", "donnees_hopitaux": report_data}
    except Exception as e:
        HTTP_REQUESTS_TOTAL.labels(method="GET", endpoint="/hospitals/capacity", status_code="500").inc()
        raise HTTPException(status_code=500, detail=f"Erreur de lecture du rapport: {str(e)}")