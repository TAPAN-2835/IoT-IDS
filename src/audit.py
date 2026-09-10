import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict

from src import config as cfg
from src.data_loader import get_csv_files, scan_dataset, load_dataset_sample
from src.utils import setup_logger, save_json

logger = setup_logger(__name__)

def evaluate_leakage_candidate(feature: str, dtype: str) -> tuple:
    """
    Evaluates a single feature to determine if it is a leakage candidate.
    Returns (category, reason, recommended_action).
    """
    feature_lower = str(feature).lower()
    
    if feature == cfg.DEFAULT_TARGET_COL or feature == cfg.MULTICLASS_TARGET_COL:
        return ("Direct label/metadata feature", "Target column", "Keep for splitting, Drop in X")
    
    if any(kw in feature_lower for kw in ['ip', 'port', 'mac']):
        return ("Identifier", "Network identifier (IP/Port/MAC)", "Drop for operational policy")
    
    if any(kw in feature_lower for kw in ['time', 'timestamp']):
        return ("Potentially risky feature", "Raw timestamp can act as proxy for sequence", "Drop for operational policy")
        
    if any(kw in feature_lower for kw in ['id', 'capture', 'scenario', 'frame']):
        return ("Identifier", "Metadata or sequence identifier", "Drop for operational policy")
        
    if dtype == 'object' or str(dtype) == 'category':
        return ("Unknown", "Categorical or string type needs review", "Keep but requires encoding")
        
    return ("Safe operational feature", "Standard numerical feature", "Keep")

def audit_features(columns: List[str], dtypes_dict: Dict[str, str]) -> pd.DataFrame:
    """
    Classifies all columns into leakage policies.
    """
    records = []
    for col in columns:
        dtype = dtypes_dict.get(col, "unknown")
        cat, reason, action = evaluate_leakage_candidate(col, dtype)
        records.append({
            "feature": col,
            "dtype": str(dtype),
            "category": cat,
            "reason": reason,
            "recommended_action": action
        })
        
    return pd.DataFrame(records)

def run_audit() -> None:
    """Main audit pipeline."""
    csv_files = get_csv_files()
    if not csv_files:
        logger.error("No CSV files found in data/raw/. Please check data/raw/README.md.")
        return
        
    target_csv = csv_files[0]
    if target_csv.name != "DNN-EdgeIIoT-dataset.csv":
        logger.warning(f"Primary DNN-EdgeIIoT-dataset.csv not found! Falling back to: {target_csv.name}")
    logger.info(f"Auditing dataset: {target_csv.name}")
    
    metadata = scan_dataset(target_csv)
    if not metadata:
        logger.error("Could not read dataset metadata.")
        return
        
    save_json(metadata, cfg.AUDIT_DIR / "dataset_summary.json")
    
    # Load a sample to get dtypes and distributions
    sample_df = load_dataset_sample(target_csv, n_rows=50000)
    dtypes_dict = sample_df.dtypes.apply(str).to_dict()
    
    # Feature Audit
    audit_df = audit_features(metadata['columns'], dtypes_dict)
    audit_df.to_csv(cfg.AUDIT_DIR / "feature_audit.csv", index=False)
    
    # Operational Feature Policy
    operational_features = audit_df[
        (audit_df['recommended_action'] == 'Keep') | 
        (audit_df['recommended_action'] == 'Keep but requires encoding')
    ]
    operational_features.to_csv(cfg.AUDIT_DIR / "operational_feature_policy.csv", index=False)
    
    # Write leakage candidates txt
    leakage_candidates = audit_df[audit_df['recommended_action'].str.contains('Drop')]
    with open(cfg.AUDIT_DIR / "leakage_candidates.txt", "w") as f:
        f.write("Suspicious Columns and Leakage Candidates:\n")
        f.write("="*40 + "\n")
        for _, row in leakage_candidates.iterrows():
            f.write(f"- {row['feature']} ({row['category']}): {row['reason']}\n")
            
    # Class distribution (on sample)
    if cfg.DEFAULT_TARGET_COL in sample_df.columns:
        class_dist = sample_df[cfg.DEFAULT_TARGET_COL].value_counts().reset_index()
        class_dist.columns = ['class', 'count']
        class_dist.to_csv(cfg.AUDIT_DIR / "class_distribution.csv", index=False)
        logger.info(f"Class distribution saved based on sample of {len(sample_df)} rows.")
    else:
        logger.warning(f"Target column {cfg.DEFAULT_TARGET_COL} not found in dataset.")
        
    logger.info("Dataset audit completed.")

if __name__ == "__main__":
    run_audit()
