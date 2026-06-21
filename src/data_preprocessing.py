import argparse
import logging
import os
import sys
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

COLUMNS_TO_IGNORE = [
    "DESYNPUF_ID", 
    "data_year",
    "total_cost",               
    "target_is_high_cost",      
    "target_readmission_risk",  
    "INPATIENT_CLAIM_COUNT",
    "INPATIENT_CLM_PMT_AMT_SUM",
    "INPATIENT_NCH_BENE_IP_DDCTBL_AMT_SUM",
    "INPATIENT_NCH_BENE_PTA_COINSRNC_LBLTY_AM_SUM",
    "INPATIENT_CLM_UTLZTN_DAY_CNT_SUM",
]

def build_preprocessor(num_cols: list, cat_cols: list) -> ColumnTransformer:
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_cols),
            ('cat', categorical_transformer, cat_cols)
        ],
        remainder='drop'
    )
    return preprocessor

# 🚨 NEW: The function Airflow will call
def run_preprocessing(input_path="data/processed/engineered.csv", 
                      output_data_path="data/processed/final.csv", 
                      output_model_path="src/models/preprocessor.joblib"):
    
    # When running in Docker, paths need to be absolute relative to /opt/airflow
    # We prefix with /opt/airflow/ if the path is relative to ensure Docker finds it
    if not input_path.startswith('/'):
        input_path = f"/opt/airflow/{input_path}"
    if not output_data_path.startswith('/'):
        output_data_path = f"/opt/airflow/{output_data_path}"
    if not output_model_path.startswith('/'):
        output_model_path = f"/opt/airflow/{output_model_path}"

    if not os.path.exists(input_path):
        log.error(f"File not found: {input_path}")
        raise FileNotFoundError(f"File not found: {input_path}")

    df = pd.read_csv(input_path, low_memory=False)
    
    feature_candidates = [
        col for col in df.columns 
        if col not in COLUMNS_TO_IGNORE and "INPATIENT" not in str(col).upper()
    ]
    df_features = df[feature_candidates]
    
    num_cols = df_features.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = df_features.select_dtypes(exclude=['int64', 'float64']).columns.tolist()

    if len(num_cols) == 0 and len(cat_cols) == 0:
        log.error("CRITICAL: No valid features found!")
        raise ValueError("CRITICAL: No valid features found!")

    log.info(f"Auto-detected {len(num_cols)} numeric and {len(cat_cols)} categorical features.")

    preprocessor = build_preprocessor(num_cols, cat_cols)
    X_processed = preprocessor.fit_transform(df)
    
    all_feature_names = num_cols.copy()
    if len(cat_cols) > 0:
        cat_feature_names = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(cat_cols)
        all_feature_names.extend(cat_feature_names)
    
    df_processed = pd.DataFrame(
        X_processed.toarray() if hasattr(X_processed, "toarray") else X_processed, 
        columns=all_feature_names
    )
    
    target_cols = [c for c in ["target_is_high_cost", "target_readmission_risk"] if c in df.columns]
    for tc in target_cols:
        df_processed[tc] = df[tc].values

    os.makedirs(os.path.dirname(output_data_path), exist_ok=True)
    df_processed.to_csv(output_data_path, index=False)
    
    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
    joblib.dump(preprocessor, output_model_path)
    
    log.info("Preprocessing complete! Artifact saved.")
    return "Success"

# Keep this so you can still run it manually from the terminal if you want to!
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/processed/engineered.csv")
    parser.add_argument("--output-data", type=str, default="data/processed/final.csv")
    parser.add_argument("--output-model", type=str, default="src/models/preprocessor.joblib")
    args = parser.parse_args()
    
    run_preprocessing(args.input, args.output_data, args.output_model)