import os
import hashlib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pv
from typing import Dict, List, Tuple, Optional
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


def canonical_token(value):
    """Canonicalise a numeric-looking string so "0", "0.0" and "0.00" agree.

    Non-numeric values (e.g. "MQTT", "0x00000000") are returned unchanged.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return value
    text = str(value).strip()
    try:
        number = float(text)
    except ValueError:
        return text
    return repr(number) if np.isfinite(number) else text


def _scan_columns(filepath: Path, feature_cols: List[str], target_cols: List[str],
                  max_categories: int, canonicalize: bool) -> Tuple[List[str], Dict[str, int]]:
    """Pass 1: stream the CSV as strings to find string-typed and high-cardinality columns.

    Only small per-column sets are kept in memory, so huge payload columns
    (tcp.payload, tcp.options, ...) never get materialised.
    """
    read_opts = pv.ReadOptions(block_size=16 << 20, use_threads=False)
    conv_opts = pv.ConvertOptions(include_columns=feature_cols + target_cols,
                                  column_types={c: pa.string() for c in feature_cols},
                                  strings_can_be_null=True)
    numeric = {c: True for c in feature_cols}
    distinct = {c: set() for c in feature_cols}
    raw_cap = 5 * max_categories  # canonicalisation merges at most a few raw spellings

    with pv.open_csv(filepath, read_options=read_opts, convert_options=conv_opts) as reader:
        for batch in reader:
            for c in feature_cols:
                col = batch.column(c)
                if numeric[c]:
                    try:
                        col.cast(pa.float64())
                    except pa.ArrowInvalid:
                        numeric[c] = False
                if not numeric[c] and len(distinct[c]) <= raw_cap:
                    distinct[c].update(pc.unique(col).to_pylist())

    string_cols = [c for c in feature_cols if not numeric[c]]
    cardinality = {}
    for c in string_cols:
        values = {canonical_token(v) for v in distinct[c]} if canonicalize else set(distinct[c])
        values.discard(None)
        cardinality[c] = len(values)
    return string_cols, cardinality


def load_policy_columns(filepath: Path, feature_cols: List[str], target_col: str,
                        max_categories: int = 100, canonicalize: bool = True
                        ) -> Tuple[pd.DataFrame, List[str]]:
    """Low-RAM load of only the policy features plus the target.

    Pass 1 streams the file to detect string columns and their cardinality;
    pass 2 reads only the columns that survive, with numerics as float32 and
    strings as pandas categoricals. Returns (df, dropped_high_cardinality_cols).
    """
    logger.info(f"Scanning {filepath.name} for column types and cardinality (streaming)...")
    string_cols, cardinality = _scan_columns(filepath, feature_cols, [target_col],
                                             max_categories, canonicalize)
    dropped = [c for c in string_cols if cardinality[c] > max_categories]
    for c in dropped:
        logger.warning(f"Dropping {c} due to high cardinality ({cardinality[c]}+ unique values).")

    keep = [c for c in feature_cols if c not in dropped]
    keep_strings = [c for c in string_cols if c not in dropped]
    column_types = {c: (pa.string() if c in keep_strings else pa.float32()) for c in keep}
    conv_opts = pv.ConvertOptions(include_columns=keep + [target_col], column_types=column_types,
                                  strings_can_be_null=True)
    logger.info(f"Loading {len(keep)} feature columns + {target_col} ...")
    table = pv.read_csv(filepath, convert_options=conv_opts)

    data = {}
    for c in keep + [target_col]:
        col = table.column(c)
        if c in keep_strings or pa.types.is_string(col.type):
            data[c] = col.to_pandas().astype("category")
        else:
            data[c] = col.to_numpy()
        table = table.drop_columns([c])  # release Arrow buffers as we go
    del table, col
    df = pd.DataFrame(data, copy=False)  # no consolidation copy of ~0.3 GB of float32 columns
    del data
    pa.default_memory_pool().release_unused()
    return df, dropped
