import os
import json
import logging
from pathlib import Path
from src.utils import setup_logger

logger = setup_logger("phase2_pipeline")

def main():
    logger.info("Skipping Phase 2A: Dataset Audit (Already completed)")
    
    # Check if audit produced the expected outputs for the REAL dataset
    audit_path = Path("results/audit/dataset_summary.json")
    if not audit_path.exists():
        logger.error("Audit failed to produce dataset_summary.json")
        return
        
    with open(audit_path, "r") as f:
        metadata = json.load(f)
        
    logger.info(f"Verified Real Dataset: {metadata.get('filename')}")
    logger.info(f"Rows: {metadata.get('num_rows')}, Columns: {metadata.get('num_columns')}")
    logger.info(f"Hash: {metadata.get('sha256')}")
    
    if metadata.get("filename") != "DNN-EdgeIIoT-dataset.csv":
        logger.error(f"Mismatch! Expected DNN-EdgeIIoT-dataset.csv but got {metadata.get('filename')}")
        return
        
    logger.info("Starting Phase 2C: Preprocessing...")
    from src.preprocessing import run_preprocessing_pipeline
    run_preprocessing_pipeline()
    
    logger.info("Starting Phase 2D: Random Forest Binary Baseline...")
    from src.train import train_and_evaluate
    # We first modify config.py dynamically or pass arguments.
    # The training script takes (experiment_id, target_col, task_type)
    
    # E01: Binary
    logger.info("Running E01...")
    train_and_evaluate("E01_random_forest_binary", "Attack_label", "binary")
    
    # We must re-run preprocessing for Attack_type because the target changes
    # So we edit config.DEFAULT_TARGET_COL inside python
    from importlib import reload
    import src.config as cfg
    import src.preprocessing as prep
    
    logger.info("Re-running preprocessing for Multiclass (Attack_type)...")
    cfg.DEFAULT_TARGET_COL = "Attack_type"
    reload(prep)  # Reload to pick up new config
    prep.run_preprocessing_pipeline()
    
    logger.info("Running E02: Random Forest Multiclass...")
    train_and_evaluate("E02_random_forest_multiclass", "Attack_type", "multiclass")
    
    logger.info("Phase 2 completely executed.")

if __name__ == "__main__":
    main()
