"""
src/feature_engineering.py

Generates derived features by dynamically finding chronic condition columns 
and calculating ratios and targets.
"""

import argparse
import logging
import os
import sys
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Creates new features dynamically based on available columns."""
    log.info(f"Starting feature engineering. Initial shape: {df.shape}")

    # 1. Dynamically find all chronic condition columns (Start with SP_ but are not State Code)
    chronic_cols = [c for c in df.columns if str(c).startswith("SP_") and "STATE" not in str(c).upper()]
    
    if chronic_cols:
        log.info(f"Auto-detected {len(chronic_cols)} chronic condition flags.")
        temp_chronic = df[chronic_cols].replace({2: 0, -1: 0}).fillna(0)
        df["chronic_condition_count"] = temp_chronic.sum(axis=1)
        df["is_multimorbid"] = (df["chronic_condition_count"] > 2).astype(int)

    # 2. Financial velocity
    if "total_cost" in df.columns and "PLAN_CVRG_MOS_NUM" in df.columns:
        safe_months = df["PLAN_CVRG_MOS_NUM"].replace(0, 1)
        df["cost_per_coverage_month"] = df["total_cost"] / safe_months

    # 3. Create TARGET variables dynamically
    if "total_cost" in df.columns:
        df["target_is_high_cost"] = (df["total_cost"] > 10000).astype(int)

    if "INPATIENT_CLAIM_COUNT" in df.columns:
        df["target_readmission_risk"] = (df["INPATIENT_CLAIM_COUNT"] > 1).astype(int)

    log.info(f"Feature engineering complete. Final shape: {df.shape}")
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/processed/cleaned.csv")
    parser.add_argument("--output", type=str, default="data/processed/engineered.csv")
    args = parser.parse_args()

    input_path = os.path.normpath(args.input)
    output_path = os.path.normpath(args.output)

    if not os.path.exists(input_path):
        log.error(f"File not found: {input_path}. Run data_cleaning.py first!")
        sys.exit(1)

    df = pd.read_csv(input_path, low_memory=False)
    engineered_df = engineer_features(df)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    engineered_df.to_csv(output_path, index=False)
    log.info(f"Engineered data saved to {output_path}")