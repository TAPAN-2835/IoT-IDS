import pandas as pd
import json
import logging
from src import config as cfg
from src.utils import setup_logger
from pathlib import Path

logger = setup_logger("audit_generalization")

def main():
    logger.info("=== Validation / Generalization Audit ===")
    
    data_path = cfg.RAW_DATA_DIR / "DNN-EdgeIIoT-dataset.csv"
    if not data_path.exists():
        logger.error(f"Dataset not found at {data_path}")
        return
        
    logger.info("Loading subset of raw dataset to inspect schemas and temporal data...")
    # Load just enough to see what's going on, or load full if needed.
    # Since we need to check timestamps, we might need full data but let's load specific columns.
    
    cols_to_check = ['frame.time', 'ip.src_host', 'ip.dst_host', 'arp.src.proto_ipv4', 'arp.dst.proto_ipv4']
    
    # Read first 100k rows to get a feel
    df_sample = pd.read_csv(data_path, nrows=500000, low_memory=False)
    
    found_cols = [c for c in cols_to_check if c in df_sample.columns]
    logger.info(f"Temporal/Grouping columns found: {found_cols}")
    
    if 'frame.time' in df_sample.columns:
        # Check if frame.time is a legitimate timestamp
        sample_times = df_sample['frame.time'].dropna().head()
        logger.info(f"Sample frame.time values:\n{sample_times}")
        
        # Check if it's sortable
        # Sometimes frame.time is a string like "2021-11-24 13:40:55.123456"
        # or it might just be relative time.
        
        # Check cardinality
        unique_times = df_sample['frame.time'].nunique()
        logger.info(f"Unique timestamps in 500k rows: {unique_times}")
        
    if 'ip.src_host' in df_sample.columns:
        unique_ips = df_sample['ip.src_host'].nunique()
        logger.info(f"Unique source IPs in 500k rows: {unique_ips}")
        sample_ips = df_sample['ip.src_host'].value_counts().head()
        logger.info(f"Top Source IPs:\n{sample_ips}")

    logger.info("Audit complete. Please inspect logs to decide if chronological split or sequence modeling is possible.")

if __name__ == "__main__":
    main()
