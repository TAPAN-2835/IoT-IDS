"""
run_phase3_multiclass.py

Phase 3 (Multiclass): Train 1D-CNN, GRU, and CNN-GRU on 15-class attack classification.
Run AFTER run_phase3_binary.py completes.

NOTE: Re-runs preprocessing targeting Attack_type (multiclass) instead of Attack_label (binary).
"""
import logging
import src.config as cfg
from src.utils import setup_logger
from src.preprocessing import run_preprocessing_pipeline
from src.training import train_dl_model

logger = setup_logger("phase3_multiclass_pipeline")


def main():
    logger.info("=" * 60)
    logger.info("Phase 3 Multiclass: Preprocessing for Attack_type target")
    logger.info("=" * 60)

    # Switch target column to multiclass
    cfg.DEFAULT_TARGET_COL = "Attack_type"
    run_preprocessing_pipeline()

    logger.info("=" * 60)
    logger.info("Starting E06: 1D-CNN Multiclass")
    logger.info("=" * 60)
    train_dl_model("E06_cnn1d_multiclass", "1D-CNN", "Attack_type", "multiclass")

    logger.info("=" * 60)
    logger.info("Starting E07: GRU Multiclass")
    logger.info("=" * 60)
    train_dl_model("E07_gru_multiclass", "GRU", "Attack_type", "multiclass")

    logger.info("=" * 60)
    logger.info("Starting E08: CNN-GRU Hybrid Multiclass")
    logger.info("=" * 60)
    train_dl_model("E08_cnn_gru_multiclass", "CNN-GRU", "Attack_type", "multiclass")

    logger.info("Phase 3 Multiclass complete.")


if __name__ == "__main__":
    main()
