from pathlib import Path
import os
import random
import numpy as np
import torch

# ==========================================
# REPRODUCIBILITY CONFIGURATION
# ==========================================
GLOBAL_SEED = 42
NP_SEED = 42
TORCH_SEED = 42

def set_seeds(seed=GLOBAL_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ==========================================
# DATA CONFIGURATION
# ==========================================
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR = BASE_DIR / "models"
AUDIT_DIR = RESULTS_DIR / "audit"
EXPERIMENTS_DIR = RESULTS_DIR / "experiments"
FIGURES_DIR = RESULTS_DIR / "figures"

# ==========================================
# DATA LOADING CONFIGURATION
# ==========================================
CHUNK_SIZE = 50_000  # Rows per chunk for memory-safe CSV reading

DEFAULT_TARGET_COL = "Attack_label"
MULTICLASS_TARGET_COL = "Attack_type"
FEATURE_POLICY = "operational"

for d in [INTERIM_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, AUDIT_DIR, EXPERIMENTS_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ==========================================
# SPLIT CONFIGURATION
# ==========================================
RANDOM_SEED = GLOBAL_SEED
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ==========================================
# MODELS CONFIGURATION
# ==========================================
BATCH_SIZE = 8192
LEARNING_RATE = 1e-3
EPOCHS = 5
EARLY_STOPPING_PATIENCE = 3

# ==========================================
# LEAKAGE KEYWORDS
# ==========================================
LEAKAGE_KEYWORDS = [
    'ip', 'port', 'mac', 'timestamp', 'time', 
    'capture', 'session', 'id'
]
