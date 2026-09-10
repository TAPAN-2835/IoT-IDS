import os
import hashlib
import pandas as pd
from typing import List, Tuple, Optional
from pathlib import Path
from src.config import RAW_DATA_DIR, CHUNK_SIZE
from src.utils import setup_logger

logger = setup_logger(__name__)

def get_csv_files(directory: Path = RAW_DATA_DIR) -> List[Path]:
    """Discover all CSV files in the raw data directory, prioritizing DNN-EdgeIIoT-dataset.csv."""
    files = list(directory.glob("*.csv"))
    # Prioritize DNN dataset if multiple exist
    dnn_file = next((f for f in files if f.name == "DNN-EdgeIIoT-dataset.csv"), None)
    if dnn_file:
        files.remove(dnn_file)
        files.insert(0, dnn_file)
    return files

def calculate_sha256(filepath: Path) -> str:
    """Calculate the SHA-256 hash of a file."""
    sha256_hash = hashlib.sha256()
    try:
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    except Exception as e:
        logger.error(f"Error calculating hash for {filepath}: {e}")
        return "ERROR"

def scan_dataset(filepath: Path) -> dict:
    """Scan the dataset to discover schema without loading it fully into memory."""
    logger.info(f"Scanning dataset: {filepath.name}")
    
    file_size = filepath.stat().st_size
    
    try:
        # Read just the header to get columns
        df_header = pd.read_csv(filepath, nrows=0, low_memory=False)
        columns = df_header.columns.tolist()
        
        # Count rows in chunks
        row_count = 0
        for chunk in pd.read_csv(filepath, chunksize=CHUNK_SIZE, low_memory=False):
            row_count += len(chunk)
            
        metadata = {
            "filename": filepath.name,
            "source_url": "https://www.kaggle.com/datasets/mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot",
            "file_size_bytes": file_size,
            "sha256": calculate_sha256(filepath),
            "num_rows": row_count,
            "num_columns": len(columns),
            "columns": columns,
        }
        return metadata
    except Exception as e:
        logger.error(f"Failed to scan {filepath}: {e}")
        return {}

def load_dataset_sample(filepath: Path, n_rows: int = 10000) -> pd.DataFrame:
    """Load a small sample for EDA."""
    logger.info(f"Loading sample of {n_rows} rows from {filepath.name}")
    return pd.read_csv(filepath, nrows=n_rows, low_memory=False)

def load_dataset_full(filepath: Path) -> pd.DataFrame:
    """Load the full dataset into memory. Use carefully!"""
    logger.info(f"Loading FULL dataset into memory: {filepath.name}")
    return pd.read_csv(filepath, engine="pyarrow")
