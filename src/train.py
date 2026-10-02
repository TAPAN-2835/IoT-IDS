import os
import json
import time
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score,
    average_precision_score, precision_recall_fscore_support
)

from src.config import RANDOM_SEED, MODELS_DIR, RESULTS_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR, AUDIT_DIR
from src.utils import setup_logger, save_json

logger = setup_logger(__name__)

def train_and_evaluate(experiment_id: str, target_col: str, task_type: str):
    logger.info(f"Starting {experiment_id} for target: {target_col}")
    exp_dir = RESULTS_DIR / "experiments" / experiment_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Load metadata to get the hash and exact file
    with open(AUDIT_DIR / "dataset_summary.json", "r") as f:
        metadata = json.load(f)
        
    # Load processed data
    # Preprocessing pipeline saves y as integer encoded
    # We should use y_train.parquet, y_test.parquet
    X_train = pd.read_parquet(PROCESSED_DATA_DIR / "X_train.parquet")
    y_train = pd.read_parquet(PROCESSED_DATA_DIR / "y_train.parquet").iloc[:, 0]
    X_test = pd.read_parquet(PROCESSED_DATA_DIR / "X_test.parquet")
    y_test = pd.read_parquet(PROCESSED_DATA_DIR / "y_test.parquet").iloc[:, 0]
    
    # Load label mapping
    with open(MODELS_DIR / "label_mapping.json", "r") as f:
        label_mapping = json.load(f)
        
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        class_weight="balanced"
    )
    
    # Train
    logger.info("Training Random Forest...")
    t0 = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - t0
    
    # Inference
    logger.info("Evaluating on Test Set...")
    t1 = time.time()
    y_pred = model.predict(X_test)
    inference_time = time.time() - t1
    
    try:
        y_prob = model.predict_proba(X_test)
    except:
        y_prob = None
        
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    # Multi/Binary handling
    avg_type = "binary" if task_type == "binary" else "macro"
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average=avg_type, zero_division=0)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    
    cm = confusion_matrix(y_test, y_pred)
    
    # FPR/FNR calculation for binary
    fpr, fnr = None, None
    if task_type == "binary":
        tn, fp, fn, tp = cm.ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        roc_auc = roc_auc_score(y_test, y_prob[:, 1]) if y_prob is not None else None
        pr_auc = average_precision_score(y_test, y_prob[:, 1]) if y_prob is not None else None
    else:
        # Multiclass AUC
        try:
            roc_auc = roc_auc_score(y_test, y_prob, multi_class='ovr') if y_prob is not None else None
        except Exception as e:
            logger.warning(f"Could not compute ROC-AUC: {e}")
            roc_auc = None
        pr_auc = None  # Complex for multiclass, skipping
        
    # Save Model
    model_path = MODELS_DIR / f"random_forest_{task_type}.joblib"
    joblib.dump(model, model_path)
    model_size = os.path.getsize(model_path)
    
    from datetime import datetime
    features_used = []
    feature_path = MODELS_DIR / "feature_columns.json"
    if feature_path.exists():
        import json
        with open(feature_path, "r") as f:
            features_used = json.load(f).get("features", [])
            
    # Save metrics JSON/CSV
    metrics = {
        "experiment_id": experiment_id,
        "timestamp": datetime.now().isoformat(),
        "dataset": "Edge-IIoTset",
        "dataset_file": metadata.get("filename"),
        "dataset_hash": metadata.get("sha256"),
        "target": target_col,
        "split_strategy": "stratified_random",
        "random_seed": RANDOM_SEED,
        "model": "RandomForestClassifier",
        "hyperparameters": {"n_estimators": 200, "class_weight": "balanced"},
        "train_size": len(X_train),
        "test_size": len(X_test),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "fpr": fpr,
        "fnr": fnr,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "training_time_s": training_time,
        "inference_time_s": inference_time,
        "inference_latency_ms": (inference_time / len(X_test)) * 1000,
        "throughput_req_per_s": len(X_test) / inference_time,
        "model_size_bytes": model_size,
        "features_used": features_used
    }
    
    save_json(metrics, exp_dir / "experiment_record.json")
    
    csv_metrics = {k: v for k, v in metrics.items() if k not in ["features_used", "hyperparameters"]}
    pd.DataFrame([csv_metrics]).to_csv(exp_dir / "experiment_record.csv", index=False)
    
    # Update central experiment registry
    registry_path = RESULTS_DIR / "experiment_registry.csv"
    if registry_path.exists():
        df_reg = pd.read_csv(registry_path)
    else:
        df_reg = pd.DataFrame()
    df_reg = pd.concat([df_reg, pd.DataFrame([csv_metrics])], ignore_index=True)
    df_reg.to_csv(registry_path, index=False)
    
    # Classification Report
    report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    pd.DataFrame(report_dict).transpose().to_csv(exp_dir / "classification_report.csv")
    
    # Confusion Matrix Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"{experiment_id} Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.savefig(exp_dir / "confusion_matrix.png")
    plt.close()
    
    # Feature Importance
    importances = model.feature_importances_
    features = X_train.columns
    fi_df = pd.DataFrame({"feature": features, "importance": importances})
    fi_df = fi_df.sort_values(by="importance", ascending=False)
    fi_df.to_csv(exp_dir / "feature_importance.csv", index=False)
    
    # Plot top 20
    plt.figure(figsize=(12, 8))
    sns.barplot(data=fi_df.head(20), x="importance", y="feature", palette="viridis")
    plt.title("Top 20 Operational Features")
    plt.tight_layout()
    plt.savefig(exp_dir / "feature_importance.png")
    plt.close()
    
    logger.info(f"Finished {experiment_id}. Accuracy: {accuracy:.4f}")
    return metrics


def train_xgb_gpu(experiment_id: str, target_col: str, task_type: str,
                  n_estimators: int = 500, max_depth: int = 8, learning_rate: float = 0.1):
    """Gradient-boosted tree baseline trained on the GPU.

    Replaces the 200-tree RandomForest for new experiments: the RF used every CPU
    core and several GB of RAM (and a 1.6 GB model file for multiclass), while
    XGBoost's QuantileDMatrix keeps a ~1-byte-per-value copy on the GPU.
    """
    import xgboost as xgb
    import torch
    from datetime import datetime
    from src.training import load_processed_data, preprocessing_record

    logger.info(f"Starting {experiment_id} (XGBoost, GPU) for target: {target_col}")
    exp_dir = RESULTS_DIR / "experiments" / experiment_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    data, meta = load_processed_data(target_col)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    feature_names = meta["feature_names"]

    params = {
        "tree_method": "hist",
        "device": device,
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "seed": RANDOM_SEED,
        "nthread": 2,  # keep the laptop responsive; the GPU does the heavy lifting
    }
    if task_type == "binary":
        params.update(objective="binary:logistic", eval_metric="logloss")
    else:
        params.update(objective="multi:softprob", eval_metric="mlogloss",
                      num_class=int(meta["n_classes"]))

    dtrain = xgb.QuantileDMatrix(data["X_train"], label=data["y_train"])
    dval = xgb.QuantileDMatrix(data["X_val"], label=data["y_val"], ref=dtrain)
    X_test, y_test = data["X_test"], data["y_test"]
    del data

    logger.info(f"Training XGBoost on {device}...")
    t0 = time.time()
    booster = xgb.train(params, dtrain, num_boost_round=n_estimators,
                        evals=[(dval, "val")], early_stopping_rounds=20, verbose_eval=50)
    training_time = time.time() - t0
    del dtrain, dval

    logger.info("Evaluating on Test Set...")
    dtest = xgb.DMatrix(X_test)
    t1 = time.time()
    y_prob = booster.predict(dtest, iteration_range=(0, booster.best_iteration + 1))
    inference_time = time.time() - t1
    y_pred = (y_prob > 0.5).astype(int) if task_type == "binary" else y_prob.argmax(axis=1)

    accuracy = accuracy_score(y_test, y_pred)
    avg_type = "binary" if task_type == "binary" else "macro"
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average=avg_type, zero_division=0)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    fpr, fnr = None, None
    if task_type == "binary":
        tn, fp, fn, tp = cm.ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

    model_path = exp_dir / "model.ubj"
    booster.save_model(model_path)

    record = {
        "experiment_id": experiment_id,
        "timestamp": datetime.now().isoformat(),
        "model": "XGBoost-GPU",
        "dataset": "Edge-IIoTset",
        "feature_policy": meta["feature_policy"],
        "split": "stratified_random",
        "seed": RANDOM_SEED,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "fpr": fpr,
        "fnr": fnr,
        "training_time": training_time,
        "inference_latency": (inference_time / len(X_test)) * 1000,
        "parameters": int(booster.best_iteration + 1),  # trees used
        "model_size_mb": os.path.getsize(model_path) / (1024 * 1024),
        "loss_type": params["objective"],
        "canonicalize_numeric_tokens": meta["canonicalize_numeric_tokens"],
        "device": device,
        "hyperparameters": {"n_estimators": n_estimators, "best_iteration": int(booster.best_iteration),
                            "max_depth": max_depth, "learning_rate": learning_rate},
        "preprocessing": preprocessing_record(meta),
        "features_used": feature_names,
    }
    save_json(record, exp_dir / "experiment_record.json")

    csv_record = {k: v for k, v in record.items() if k not in ["features_used", "hyperparameters", "preprocessing"]}
    registry_path = RESULTS_DIR / "experiment_registry.csv"
    df_reg = pd.read_csv(registry_path) if registry_path.exists() else pd.DataFrame()
    pd.concat([df_reg, pd.DataFrame([csv_record])], ignore_index=True).to_csv(registry_path, index=False)

    labels = [meta["label_mapping"][str(i)] for i in range(int(meta["n_classes"]))]
    report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0,
                                        labels=list(range(len(labels))), target_names=labels)
    pd.DataFrame(report_dict).transpose().to_csv(exp_dir / "classification_report.csv")

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=task_type == "binary", fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title(f"{experiment_id} Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.savefig(exp_dir / "confusion_matrix.png", bbox_inches="tight")
    plt.close()

    gain = booster.get_score(importance_type="gain")
    fi_df = pd.DataFrame({"feature": feature_names,
                          "importance": [gain.get(f"f{i}", 0.0) for i in range(len(feature_names))]})
    fi_df["importance"] /= fi_df["importance"].sum() or 1.0
    fi_df = fi_df.sort_values(by="importance", ascending=False)
    fi_df.to_csv(exp_dir / "feature_importance.csv", index=False)

    plt.figure(figsize=(12, 8))
    sns.barplot(data=fi_df.head(20), x="importance", y="feature", color="#3b82f6")
    plt.title(f"Top 20 Features (gain) - {experiment_id}")
    plt.tight_layout()
    plt.savefig(exp_dir / "feature_importance.png")
    plt.close()

    logger.info(f"Finished {experiment_id}. Accuracy: {accuracy:.4f} | Macro-F1: {macro_f1:.4f}")
    return record


if __name__ == "__main__":
    import sys
    exp_id = sys.argv[1]
    target = sys.argv[2]
    task = sys.argv[3]
    train_and_evaluate(exp_id, target, task)
