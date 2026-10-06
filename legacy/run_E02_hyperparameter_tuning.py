import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import logging
import json
import time
from pathlib import Path
from sklearn.metrics import f1_score

from src import config as cfg
from src.utils import setup_logger, save_json
from src.training import FocalLoss, load_processed_data, preprocessing_record, iterate_batches
from src.models import CNN_GRU

logger = setup_logger("E02_hyperparameter_tuning")

def get_data(smoke_test=False):
    """Multiclass train/val splits as tensors on the GPU (test split is never loaded)."""
    logger.info("Loading parquet datasets for tuning...")
    data, meta = load_processed_data("Attack_type")  # raises if data/processed/ is not multiclass
    X_train, y_train, X_val, y_val = data["X_train"], data["y_train"], data["X_val"], data["y_val"]
    del data

    if smoke_test:
        rng = np.random.default_rng(42)
        train_idx = rng.choice(len(X_train), size=min(10000, len(X_train)), replace=False)
        val_idx = rng.choice(len(X_val), size=min(2000, len(X_val)), replace=False)
        X_train, y_train, X_val, y_val = X_train[train_idx], y_train[train_idx], X_val[val_idx], y_val[val_idx]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensors = (torch.from_numpy(X_train).to(device), torch.tensor(y_train, dtype=torch.long, device=device),
               torch.from_numpy(X_val).to(device), torch.tensor(y_val, dtype=torch.long, device=device))
    return (*tensors, meta)

class Tunable_CNN_GRU(nn.Module):
    def __init__(self, input_dim, num_classes, dropout_rate, conv_filters, kernel_size, gru_units, dense_units):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, conv_filters, kernel_size=kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(conv_filters),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout_rate)
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_dim)
            self.flattened_size = self.conv(dummy).view(1, -1).shape[1]
            
        self.gru = nn.GRU(input_size=self.flattened_size, hidden_size=gru_units, batch_first=True)
        self.dense = nn.Sequential(
            nn.Linear(gru_units, dense_units),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dense_units, num_classes)
        )
        
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = x.view(x.size(0), 1, -1)
        x, _ = self.gru(x)
        x = self.dense(x[:, -1, :])
        return x

def get_objective(X_train, y_train, X_val, y_val, smoke_test):
    input_dim = X_train.shape[1]
    num_classes = int(torch.unique(y_train).numel())
    device = X_train.device
    y_val_np = y_val.cpu().numpy()
    
    def objective(trial):
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [1024, 2048, 4096, 8192])
        dropout = trial.suggest_float("dropout", 0.1, 0.5)
        focal_gamma = trial.suggest_float("focal_gamma", 1.0, 3.0)
        
        conv_filters = trial.suggest_categorical("conv_filters", [16, 32, 64])
        kernel_size = trial.suggest_categorical("kernel_size", [3, 5])
        gru_units = trial.suggest_categorical("gru_units", [32, 64, 128])
        dense_units = trial.suggest_categorical("dense_units", [16, 32, 64])
        
        model = Tunable_CNN_GRU(input_dim, num_classes, dropout, conv_filters, kernel_size, gru_units, dense_units).to(device)
        criterion = FocalLoss(gamma=focal_gamma)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        epochs = 2 if smoke_test else 15
        
        best_macro_f1 = 0.0
        patience = 5
        patience_counter = 0
        
        for epoch in range(epochs):
            model.train()
            for X_b, y_b in iterate_batches(X_train, y_train, batch_size, shuffle=True):
                optimizer.zero_grad()
                outputs = model(X_b)
                loss = criterion(outputs, y_b)
                loss.backward()
                optimizer.step()
                
            model.eval()
            all_preds = []
            with torch.no_grad():
                for X_b, _ in iterate_batches(X_val, y_val, batch_size):
                    all_preds.append(torch.argmax(model(X_b), dim=1))
            all_preds = torch.cat(all_preds).cpu().numpy()

            macro_f1 = f1_score(y_val_np, all_preds, average="macro", zero_division=0)
            
            if macro_f1 > best_macro_f1:
                best_macro_f1 = macro_f1
                patience_counter = 0
            else:
                patience_counter += 1
                
            trial.report(macro_f1, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
                
            if patience_counter >= patience:
                break
                
        return best_macro_f1
    return objective

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    
    logger.info("=== E02 Optuna Hyperparameter Tuning ===")
    if args.smoke_test:
        logger.info("SMOKE TEST MODE ACTIVATED")
        
    cfg.set_seeds()
        
    exp_dir = cfg.EXPERIMENTS_DIR / "E02_hyperparameter_tuning"
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    X_train, y_train, X_val, y_val, meta = get_data(smoke_test=args.smoke_test)
    logger.info(f"Tuning on data fingerprint {meta['fingerprint']} (policy={meta['feature_policy']}, "
                f"canonicalize={meta['canonicalize_numeric_tokens']}) on {X_train.device}")
    
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3)
    study = optuna.create_study(direction="maximize", study_name="CNN_GRU_Focal_Tuning", pruner=pruner)
    
    n_trials = 2 if args.smoke_test else 20
    
    objective = get_objective(X_train, y_train, X_val, y_val, args.smoke_test)
    study.optimize(objective, n_trials=n_trials)
    
    logger.info("Tuning complete.")
    if len(study.trials) > 0 and study.best_trial is not None:
        logger.info(f"Best Trial: {study.best_trial.number}")
        logger.info(f"Best Value (Macro-F1): {study.best_trial.value}")
        logger.info(f"Best Params: {study.best_trial.params}")
        save_json({**study.best_trial.params,
                   "best_val_macro_f1": study.best_trial.value,
                   "smoke_test": args.smoke_test,
                   "preprocessing": preprocessing_record(meta)}, exp_dir / "best_params.json")
        
    # Save trial history
    trials_df = study.trials_dataframe()
    trials_df.to_csv(exp_dir / "optuna_trials.csv", index=False)

if __name__ == "__main__":
    main()
