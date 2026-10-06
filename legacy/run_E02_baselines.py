import os
import argparse
from importlib import reload
import src.config as cfg
import src.preprocessing as prep
from src.training import train_dl_model
from src.utils import setup_logger

logger = setup_logger("e02_baselines")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true", help="Run with 1 epoch on reduced dataset.")
    args = parser.parse_args()

    logger.info("=== E02 Controlled Loss-Function Baselines ===")
    
    # We must ensure we are evaluating on the ablated no_mqtt feature set for E02 tuning!
    cfg.DEFAULT_TARGET_COL = "Attack_type"
    cfg.FEATURE_POLICY = "no_mqtt"
    reload(prep)
    
    if args.smoke_test:
        logger.info("SMOKE TEST: Overriding epochs to 1.")
        cfg.EPOCHS = 1
        # In a real setup we would also subsample the dataset, 
        # but the updated `train_dl_model` requires the parquet files.
        # So we'll just let it run 1 epoch.
    
    logger.info("Running preprocessing pipeline to generate no_mqtt dataset...")
    prep.run_preprocessing_pipeline()
    
    logger.info("1. Training CNN-GRU with Normal Cross-Entropy...")
    train_dl_model("E02_baseline_ce", "CNN-GRU", "Attack_type", "multiclass", loss_type="ce")
    
    logger.info("2. Training CNN-GRU with Class Weighted Loss...")
    train_dl_model("E02_class_weighted", "CNN-GRU", "Attack_type", "multiclass", loss_type="class_weighted")
    
    logger.info("3. Training CNN-GRU with Focal Loss...")
    train_dl_model("E02_focal_loss", "CNN-GRU", "Attack_type", "multiclass", loss_type="focal")

    logger.info("E02 Baselines completed. All artifacts (CM, histories) saved to results/experiments/E02_*")

if __name__ == "__main__":
    main()
