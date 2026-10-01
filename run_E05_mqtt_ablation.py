import os
import pandas as pd
from pathlib import Path
from importlib import reload

import src.config as cfg
import src.preprocessing as prep
from src.train import train_and_evaluate
from src.training import train_dl_model
from src.utils import setup_logger

logger = setup_logger("e05_mqtt_ablation")

def create_no_mqtt_policy():
    policy_path = cfg.AUDIT_DIR / "operational_feature_policy.csv"
    df = pd.read_csv(policy_path)
    
    # Filter out MQTT features
    no_mqtt_df = df[~df['feature'].str.startswith("mqtt.")]
    
    out_path = cfg.AUDIT_DIR / "no_mqtt_feature_policy.csv"
    no_mqtt_df.to_csv(out_path, index=False)
    logger.info(f"Created {out_path} with {len(no_mqtt_df)} features.")

def main():
    logger.info("=== E05 MQTT Ablation Study ===")
    
    # Step 1: Create the new policy file
    create_no_mqtt_policy()
    
    # Step 2: Set the feature policy and reload preprocessing
    cfg.FEATURE_POLICY = "no_mqtt"
    reload(prep)
    
    # Since we are doing binary classification ablation, ensure target is Attack_label
    cfg.DEFAULT_TARGET_COL = "Attack_label"
    
    # Step 3: Run Preprocessing
    logger.info("Running Preprocessing without MQTT features...")
    prep.run_preprocessing_pipeline()
    
    # Step 4: Train Random Forest (Binary)
    logger.info("Training Random Forest (Binary) without MQTT... (Skipping, already done)")
    # train_and_evaluate("E05_rf_no_mqtt", "Attack_label", "binary")
    
    # Step 5: Train CNN-GRU (Binary)
    logger.info("Training CNN-GRU (Binary) without MQTT...")
    train_dl_model("E05_cnn_gru_no_mqtt", "CNN-GRU", "Attack_label", "binary")

    logger.info("E05 MQTT Ablation completed.")

if __name__ == "__main__":
    main()
