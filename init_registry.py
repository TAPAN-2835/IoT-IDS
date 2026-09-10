import pandas as pd
from pathlib import Path
import json
import os

from src.config import EXPERIMENTS_DIR, RESULTS_DIR

def init_registry():
    registry_path = RESULTS_DIR / "experiment_registry.csv"
    
    columns = [
        "experiment_id", "model", "dataset", "feature_policy", "split", "seed",
        "accuracy", "precision", "recall", "f1", "macro_f1", "weighted_f1",
        "fpr", "fnr", "training_time", "inference_latency", "parameters", "model_size_mb"
    ]
    
    records = []
    
    # Backfill E01 and E02
    for exp_id in ["E01_random_forest_binary", "E02_random_forest_multiclass"]:
        record_json = EXPERIMENTS_DIR / exp_id / "experiment_record.json"
        if record_json.exists():
            with open(record_json, "r") as f:
                data = json.load(f)
                
            # Random Forest doesn't log parameter count easily, set to 0 for now
            records.append({
                "experiment_id": exp_id,
                "model": data.get("model", "RandomForest"),
                "dataset": data.get("dataset", "Edge-IIoTset"),
                "feature_policy": "operational",
                "split": data.get("split_strategy", "stratified_random"),
                "seed": data.get("random_seed", 42),
                "accuracy": data.get("accuracy"),
                "precision": data.get("precision"),
                "recall": data.get("recall"),
                "f1": data.get("f1"),
                "macro_f1": data.get("macro_f1"),
                "weighted_f1": data.get("weighted_f1"),
                "fpr": data.get("fpr"),
                "fnr": data.get("fnr"),
                "training_time": data.get("training_time_s"),
                "inference_latency": data.get("inference_latency_ms"),
                "parameters": 0,
                "model_size_mb": data.get("model_size_bytes", 0) / (1024*1024)
            })
            
    df = pd.DataFrame(records, columns=columns)
    df.to_csv(registry_path, index=False)
    print(f"Registry created at {registry_path} with {len(df)} records.")

if __name__ == "__main__":
    init_registry()
