import logging
from src.utils import setup_logger
from src.training import train_dl_model

logger = setup_logger("phase3_binary_pipeline")

def main():
    logger.info("Starting Phase 3: Binary DL Models...")
    
    # E03: 1D-CNN
    train_dl_model("E03_cnn1d_binary", "1D-CNN", "Attack_label", "binary")
    
    # E04: GRU
    train_dl_model("E04_gru_binary", "GRU", "Attack_label", "binary")
    
    # E05: CNN-GRU
    train_dl_model("E05_cnn_gru_binary", "CNN-GRU", "Attack_label", "binary")
    
    logger.info("Phase 3 Binary Models execution finished.")

if __name__ == "__main__":
    main()
