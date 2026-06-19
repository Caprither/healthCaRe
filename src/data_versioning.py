"""
src/data_versioning.py

Layer 1 — Data Ingestion & Splitting
Dynamically reads United.csv, extracts a specific year's columns without hardcoding,
strips the year suffix, and calculates baseline totals.
"""

import argparse
import logging
import os
import sys
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

def process_year(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """Dynamically extracts a specific year's data and strips the year suffix."""
    log.info(f"Extracting data for year {year}...")
    
    suffix = f"_{year}"
    
    # 1. Dynamically find all columns belonging to this year
    year_cols = [c for c in df.columns if str(c).endswith(suffix)]
    
    # 2. Dynamically find shared columns (columns that don't have ANY year suffix)
    shared_cols = [
        c for c in df.columns 
        if not str(c).endswith("_2008") and not str(c).endswith("_2009") and not str(c).endswith("_2010")
    ]
    
    # 3. Slice the dataframe to just this year + shared columns
    batch = df[shared_cols + year_cols].copy()
    
    # 4. Rename columns to remove the suffix (e.g., BENE_SEX_IDENT_CD_2008 -> BENE_SEX_IDENT_CD)
    rename_map = {c: c.replace(suffix, "") for c in year_cols}
    batch.rename(columns=rename_map, inplace=True)
    
    # 5. Dynamically calculate total_cost by finding any column containing "MEDREIMB"
    cost_cols = [c for c in batch.columns if "MEDREIMB" in str(c)]
    if cost_cols:
        batch["total_cost"] = batch[cost_cols].fillna(0).sum(axis=1)
    else:
        batch["total_cost"] = 0
        
    log.info(f"Year {year} extracted dynamically. Final Shape: {batch.shape}")
    return batch

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/raw/United.csv")
    parser.add_argument("--output-dir", type=str, default="data/versioned")
    args = parser.parse_args()

    input_path = os.path.normpath(args.input)
    output_dir = os.path.normpath(args.output_dir)

    if not os.path.exists(input_path):
        log.error(f"Raw data file not found: {input_path}")
        sys.exit(1)

    df = pd.read_csv(input_path, low_memory=False)
    os.makedirs(output_dir, exist_ok=True)

    # Process and save 2008 baseline
    df_2008 = process_year(df, 2008)
    out_2008 = os.path.join(output_dir, "cms_2008.csv")
    df_2008.to_csv(out_2008, index=False)
    log.info(f"Saved clean 2008 baseline to {out_2008}")