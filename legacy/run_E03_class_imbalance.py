import os
from importlib import reload
import src.config as cfg
import src.preprocessing as prep
from src.training import train_dl_model
from src.utils import setup_logger

logger = setup_logger("e03_class_imbalance")

def main():
    logger.info("=== E03 Class Imbalance Study ===")
    
    # E03 is for multiclass (Attack_type)
    cfg.DEFAULT_TARGET_COL = "Attack_type"
    cfg.FEATURE_POLICY = "operational" # reset to default if changed
    reload(prep)
    
    logger.info("Ensuring dataset is preprocessed for multiclass...")
    # Actually, we should just run preprocessing to be safe
    prep.run_preprocessing_pipeline()
    
    # 1. Baseline - Normal Cross Entropy (E06, E07, E08 already did this, but let's re-run for E03)
    logger.info("1. Training CNN-GRU with Normal Cross-Entropy...")
    train_dl_model("E03_baseline_ce", "CNN-GRU", "Attack_type", "multiclass", loss_type="ce")
    
    # 2. Class Weighted Loss
    logger.info("2. Training CNN-GRU with Class Weighted Loss...")
    train_dl_model("E03_class_weighted", "CNN-GRU", "Attack_type", "multiclass", loss_type="class_weighted")
    
    # 3. Focal Loss
    logger.info("3. Training CNN-GRU with Focal Loss...")
    train_dl_model("E03_focal_loss", "CNN-GRU", "Attack_type", "multiclass", loss_type="focal")

    logger.info("E03 Class Imbalance Study completed.")

if __name__ == "__main__":
    main()
