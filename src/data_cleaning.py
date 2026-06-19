"""
src/data_cleaning.py

Dynamically cleans the versioned healthcare data by finding 
financial/count columns and filling missing values.
"""

import argparse
import logging
import os
import sys
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Applies dynamic cleaning transformations."""
    log.info(f"Starting data cleaning. Initial shape: {df.shape}")

    # Drop missing IDs if the ID column exists
    if "DESYNPUF_ID" in df.columns:
        df = df.dropna(subset=["DESYNPUF_ID"])
        
    # Dynamically find any financial or claim columns to fill NaNs with 0
    financial_cols = [c for c in df.columns if any(keyword in str(c) for keyword in ["MEDREIMB", "PPPYMT", "AMT", "COST"])]
    if financial_cols:
        df[financial_cols] = df[financial_cols].fillna(0)
    
    count_cols = [c for c in df.columns if "CLAIM_COUNT" in str(c)]
    if count_cols:
        df[count_cols] = df[count_cols].fillna(0)

    # Standardize demographics if present
    if "BENE_SEX_IDENT_CD" in df.columns:
        df["BENE_SEX_IDENT_CD"] = df["BENE_SEX_IDENT_CD"].fillna(-1).astype(int)
    
    log.info("Data cleaning complete.")
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/versioned/cms_2008.csv")
    parser.add_argument("--output", type=str, default="data/processed/cleaned.csv")
    args = parser.parse_args()

    input_path = os.path.normpath(args.input)
    output_path = os.path.normpath(args.output)

    if not os.path.exists(input_path):
        log.error(f"File not found: {input_path}. Run data_versioning.py first!")
        sys.exit(1)

    df = pd.read_csv(input_path, low_memory=False)
    cleaned_df = clean_data(df)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cleaned_df.to_csv(output_path, index=False)
    log.info(f"Cleaned data saved to {output_path}")