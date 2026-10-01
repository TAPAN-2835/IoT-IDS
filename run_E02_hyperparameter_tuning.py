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
from src.training import FocalLoss
from src.models import CNN_GRU

logger = setup_logger("E02_hyperparameter_tuning")

def get_data(smoke_test=False):
    logger.info("Loading parquet datasets for tuning...")
    X_train = pd.read_parquet(cfg.PROCESSED_DATA_DIR / "X_train.parquet").values.astype(np.float32)
    y_train = pd.read_parquet(cfg.PROCESSED_DATA_DIR / "y_train.parquet").values.squeeze()
    X_val = pd.read_parquet(cfg.PROCESSED_DATA_DIR / "X_val.parquet").values.astype(np.float32)
    y_val = pd.read_parquet(cfg.PROCESSED_DATA_DIR / "y_val.parquet").values.squeeze()
    
    if smoke_test:
        rng = np.random.default_rng(42)
        train_idx = rng.choice(len(X_train), size=min(10000, len(X_train)), replace=False)
        val_idx = rng.choice(len(X_val), size=min(2000, len(X_val)), replace=False)
        return X_train[train_idx], y_train[train_idx], X_val[val_idx], y_val[val_idx]
    
    return X_train, y_train, X_val, y_val

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
    num_classes = len(np.unique(y_train))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
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
        
        train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train, dtype=torch.long))
        val_dataset = TensorDataset(torch.tensor(X_val), torch.tensor(y_val, dtype=torch.long))
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        epochs = 2 if smoke_test else 15
        
        best_macro_f1 = 0.0
        patience = 5
        patience_counter = 0
        
        for epoch in range(epochs):
            model.train()
            for X_b, y_b in train_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                optimizer.zero_grad()
                outputs = model(X_b)
                loss = criterion(outputs, y_b)
                loss.backward()
                optimizer.step()
                
            model.eval()
            all_preds, all_targets = [], []
            with torch.no_grad():
                for X_b, y_b in val_loader:
                    X_b = X_b.to(device)
                    outputs = model(X_b)
                    preds = torch.argmax(outputs, dim=1).cpu().numpy()
                    all_preds.extend(preds)
                    all_targets.extend(y_b.numpy())
                    
            macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
            
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
    
    X_train, y_train, X_val, y_val = get_data(smoke_test=args.smoke_test)
    
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
        save_json(study.best_trial.params, exp_dir / "best_params.json")
        
    # Save trial history
    trials_df = study.trials_dataframe()
    trials_df.to_csv(exp_dir / "optuna_trials.csv", index=False)

if __name__ == "__main__":
    main()
