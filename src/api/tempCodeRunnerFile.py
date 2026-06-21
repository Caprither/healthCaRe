import os
import json
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict
import uuid

# Initialisation de l'application FastAPI
app = FastAPI(
    title="🏥 Hôpital API - MLOps",
    description="API de prédiction du risque de réadmission des patients par hôpital.",
    version="2.0.0"
)

# Variables globales pour stocker le modèle en mémoire RAM
MODEL = None
FEATURE_NAMES = None
THRESHOLD = 0.5

# Liste des hôpitaux valides
VALID_HOSPITALS = ["Hôpital A", "Hôpital B", "Hôpital C", "Hôpital D", "Hôpital E"]

# ==========================================
# 1. SCHÉMAS DE DONNÉES (Le "Contrat")
# ==========================================

class PatientData(BaseModel):
    hospital: str = Field(
        ...,
        description="Nom de l'hôpital émetteur. Valeurs acceptées : Hôpital A, B, C, D ou E.",
        example="Hôpital A"
    )
    features: Dict[str, float] = Field(
        ...,
        example={"age": 65.0, "feature_1": 0.5, "feature_2": 3.0}
    )

# ==========================================
# 2. CHARGEMENT DES ARTEFACTS (Au démarrage)
# ==========================================

@app.on_event("startup")
def load_ml_artifacts():
    """Charge le modèle et le seuil générés par la Couche 2 au démarrage du serveur."""
    global MODEL, FEATURE_NAMES, THRESHOLD
    try:
        MODEL = joblib.load("output/best_model.pkl")
        FEATURE_NAMES = joblib.load("output/feature_names.pkl")

        # Chargement du seuil sécuritaire (Recall)
        with open("output/threshold.json", "r") as f:
            threshold_data = json.load(f)
            THRESHOLD = threshold_data.get("optimal_recall_threshold", 0.5)

        print(f"✅ Artefacts MLOps chargés avec succès ! Seuil appliqué : {THRESHOLD:.3f}")
    except Exception as e:
        print(f"❌ Erreur lors du chargement des artefacts: {e}")

# ==========================================
# 3. ENDPOINTS (Les URLs de l'API)
# ==========================================

@app.get("/health", tags=["Système"])
def health_check():
    """Vérifie si l'API est en ligne et si le modèle est prêt."""
    if MODEL is not None:
        return {"status": "en_ligne", "modele_charge": True}
    return {"status": "hors_ligne", "modele_charge": False}


@app.get("/hospitals", tags=["Système"])
def list_hospitals():
    """Retourne la liste des hôpitaux supportés par l'API."""
    return {"hopitaux_valides": VALID_HOSPITALS}


@app.post("/predict", tags=["Prédiction"])
def predict_readmission(data: PatientData):
    """
    Reçoit les données cliniques d'un patient avec son hôpital d'origine,
    exécute le modèle et retourne la prédiction complète sous forme de JSON string.
    """
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Le modèle n'est pas encore chargé. Réessayez dans quelques secondes.")

    # ── Validation de l'hôpital ──────────────────────────────────────────────
    if data.hospital not in VALID_HOSPITALS:
        raise HTTPException(
            status_code=422,
            detail=f"Hôpital '{data.hospital}' inconnu. Valeurs acceptées : {VALID_HOSPITALS}"
        )

    try:
        # ── 1. Conversion du JSON reçu en DataFrame Pandas ───────────────────
        df_patient = pd.DataFrame([data.features])

        # ── 2. Validation et alignement des colonnes ─────────────────────────
        for col in FEATURE_NAMES:
            if col not in df_patient.columns:
                # Si l'hôpital oublie une donnée, on met 0 par défaut
                df_patient[col] = 0.0

        df_patient = df_patient[FEATURE_NAMES]

        # ── 3. Prédiction ─────────────────────────────────────────────────────
        probability   = float(MODEL.predict_proba(df_patient)[0, 1])
        is_high_risk  = probability >= THRESHOLD

        # ── 4. Décision Métier ────────────────────────────────────────────────
        action = (
            "Réserver un lit préventif"
            if is_high_risk
            else "Décharge standard (Pas de lit requis)"
        )

        # ── 5. Construction du résultat et sérialisation en JSON string ───────
        result = {
            "patient_id"              : str(uuid.uuid4())[:8],
            "hopital"                 : data.hospital,
            "risque_calcule"          : round(probability, 4),
            "seuil_alerte"            : round(THRESHOLD, 4),
            "readmission_predite"     : is_high_risk,
            "recommandation_hopital"  : action
        }

        # Retour sous forme de JSON string (comme demandé)
        return {"prediction": json.dumps(result, ensure_ascii=False)}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))