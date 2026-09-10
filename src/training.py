import time
import os
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score, confusion_matrix

from src import config as cfg
from src.utils import setup_logger, save_json
from src.models import CNN1D, GRUBaseline, CNN_GRU

logger = setup_logger(__name__)

def train_dl_model(experiment_id, model_name, target_col, task_type):
    logger.info(f"Starting {experiment_id} using {model_name} on {target_col}")
    cfg.set_seeds()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    exp_dir = cfg.EXPERIMENTS_DIR / experiment_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    logger.info("Loading parquet datasets...")
    X_train = pd.read_parquet(cfg.PROCESSED_DATA_DIR / "X_train.parquet").values.astype(np.float32)
    y_train = pd.read_parquet(cfg.PROCESSED_DATA_DIR / "y_train.parquet").values.squeeze()
    X_val = pd.read_parquet(cfg.PROCESSED_DATA_DIR / "X_val.parquet").values.astype(np.float32)
    y_val = pd.read_parquet(cfg.PROCESSED_DATA_DIR / "y_val.parquet").values.squeeze()
    X_test = pd.read_parquet(cfg.PROCESSED_DATA_DIR / "X_test.parquet").values.astype(np.float32)
    y_test = pd.read_parquet(cfg.PROCESSED_DATA_DIR / "y_test.parquet").values.squeeze()
    
    input_dim = X_train.shape[1]
    
    if task_type == "binary":
        num_classes = 1
        criterion = nn.BCEWithLogitsLoss()
        # Normal is class 7 in label_mapping.json. Everything else is Attack.
        y_train_bin = (y_train != 7).astype(np.float32)
        y_val_bin = (y_val != 7).astype(np.float32)
        y_test_bin = (y_test != 7).astype(np.float32)
        
        y_train_t = torch.tensor(y_train_bin).unsqueeze(1)
        y_val_t = torch.tensor(y_val_bin).unsqueeze(1)
        y_test_t = torch.tensor(y_test_bin).unsqueeze(1)
    else:
        num_classes = len(np.unique(y_train))
        criterion = nn.CrossEntropyLoss()
        y_train_t = torch.tensor(y_train, dtype=torch.long)
        y_val_t = torch.tensor(y_val, dtype=torch.long)
        y_test_t = torch.tensor(y_test, dtype=torch.long)
        
    train_dataset = TensorDataset(torch.tensor(X_train), y_train_t)
    val_dataset = TensorDataset(torch.tensor(X_val), y_val_t)
    test_dataset = TensorDataset(torch.tensor(X_test), y_test_t)
    
    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False)
    
    # Initialize Model
    if model_name == "1D-CNN":
        model = CNN1D(input_dim, num_classes).to(device)
    elif model_name == "GRU":
        model = GRUBaseline(input_dim, num_classes).to(device)
    elif model_name == "CNN-GRU":
        model = CNN_GRU(input_dim, num_classes).to(device)
    else:
        raise ValueError("Unknown model name")
        
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)
    
    # Training Loop with Early Stopping
    logger.info("Training Model...")
    t0 = time.time()
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_path = exp_dir / "best_model.pt"
    
    for epoch in range(cfg.EPOCHS):
        model.train()
        train_loss = 0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            outputs = model(X_b)
            loss = criterion(outputs, y_b)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                outputs = model(X_b)
                loss = criterion(outputs, y_b)
                val_loss += loss.item()
                
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        logger.info(f"Epoch {epoch+1}/{cfg.EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1
            if patience_counter >= cfg.EARLY_STOPPING_PATIENCE:
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break
                
    training_time = time.time() - t0
    
    # Load best model for inference
    model.load_state_dict(torch.load(best_model_path))
    if model_name == "CNN-GRU":
        # Save final architecture as requested
        torch.save(model.state_dict(), cfg.MODELS_DIR / "cnn_gru_final.pt")
    
    # Inference
    logger.info("Evaluating on Test Set...")
    model.eval()
    t1 = time.time()
    
    all_preds = []
    with torch.no_grad():
        for X_b, y_b in test_loader:
            X_b = X_b.to(device)
            outputs = model(X_b)
            if task_type == "binary":
                preds = (torch.sigmoid(outputs) > 0.5).int().cpu().numpy()
            else:
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            
    inference_time = time.time() - t1
    y_pred = np.array(all_preds).squeeze()
    
    if task_type == "binary":
        y_test_bin = (y_test != 7).astype(np.float32)
        accuracy = accuracy_score(y_test_bin, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test_bin, y_pred, average="binary", zero_division=0)
        macro_f1 = f1_score(y_test_bin, y_pred, average="macro", zero_division=0)
        weighted_f1 = f1_score(y_test_bin, y_pred, average="weighted", zero_division=0)
        
        cm = confusion_matrix(y_test_bin, y_pred)
        tn, fp, fn, tp = cm.ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    else:
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="macro", zero_division=0)
        macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        fpr, fnr = None, None
        
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_mb = os.path.getsize(best_model_path) / (1024 * 1024)
    
    # Update Registry
    registry_path = cfg.RESULTS_DIR / "experiment_registry.csv"
    if registry_path.exists():
        df = pd.read_csv(registry_path)
    else:
        df = pd.DataFrame()
        
    new_record = {
        "experiment_id": experiment_id,
        "model": model_name,
        "dataset": "Edge-IIoTset",
        "feature_policy": cfg.FEATURE_POLICY,
        "split": "stratified_random",
        "seed": cfg.GLOBAL_SEED,
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
        "parameters": param_count,
        "model_size_mb": model_size_mb
    }
    
    df = pd.concat([df, pd.DataFrame([new_record])], ignore_index=True)
    df.to_csv(registry_path, index=False)
    
    save_json(new_record, exp_dir / "experiment_record.json")
    logger.info(f"Finished {experiment_id}. Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    import sys
    exp_id = sys.argv[1]
    model_n = sys.argv[2]
    target = sys.argv[3]
    task = sys.argv[4]
    train_dl_model(exp_id, model_n, target, task)
