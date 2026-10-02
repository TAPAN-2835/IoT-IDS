"""
run_tune_cnn_gru.py

Optuna search for the CNN-GRU on whatever data/processed/ holds (binary or
multiclass), then retrains the best configuration with several seeds through
the normal training pipeline (validation-tuned threshold for binary).

Objective: validation Macro-F1 (binary: at the best validation threshold).
The test split is never touched during the search.

Progress appears in watch_dashboard.py (one step per trial, then the final runs).

Usage:
    python run_tune_cnn_gru.py --trials 20 --prefix F --seeds 42 7 2024
"""
import os

os.environ.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")

import argparse
import json
import time

import numpy as np
import optuna
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

import src.config as cfg
from run_clean_baselines import be_nice
from src.models import CNN_GRU
from src.run_status import RunStatus
from src.training import (FocalLoss, iterate_batches, load_processed_data, preprocessing_record,
                          train_dl_model, _write_progress)
from src.utils import setup_logger, save_json

logger = setup_logger("tune_cnn_gru")


def build_criterion(task, loss, y_train, num_classes, device, focal_gamma=2.0):
    if task == "binary":
        pos_weight = None
        if loss == "sqrt_weighted":
            pos_weight = torch.tensor([np.sqrt((y_train == 0).sum() / max((y_train == 1).sum(), 1))],
                                      device=device, dtype=torch.float32)
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    if loss == "focal":
        return FocalLoss(gamma=focal_gamma)
    if loss == "sqrt_weighted":
        counts = np.bincount(y_train, minlength=num_classes)
        weights = np.sqrt(len(y_train) / (num_classes * np.maximum(counts, 1)))
        return nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    return nn.CrossEntropyLoss()


def val_macro_f1(model, X_val, y_val_np, task, batch_size):
    model.eval()
    with torch.no_grad():
        out = torch.cat([model(X_b) for X_b, _ in iterate_batches(X_val, X_val, batch_size)])
    if task == "binary":
        probs = torch.sigmoid(out).squeeze(1).cpu().numpy()
        return max(f1_score(y_val_np, (probs > t).astype(int), average="macro", zero_division=0)
                   for t in np.arange(0.05, 0.96, 0.05))
    return f1_score(y_val_np, out.argmax(1).cpu().numpy(), average="macro", zero_division=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--prefix", default="F")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 2024])
    args = parser.parse_args()

    be_nice()
    meta_target = json.load(open(cfg.PROCESSED_DATA_DIR / "metadata.json"))["target_col"]
    task = "binary" if meta_target == "Attack_label" else "multiclass"
    data, meta = load_processed_data(meta_target)
    tag = f"{meta['feature_policy']}_{task}"
    final_ids = [f"{args.prefix}_{tag}_best_s{s}" for s in args.seeds]
    status = RunStatus(f"Optuna CNN-GRU | {tag} | {args.trials} trials + {len(args.seeds)} seeds",
                       [f"Trial {i + 1}" for i in range(args.trials)] + final_ids)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    y_train = data["y_train"].astype(np.int64)
    X_train = torch.from_numpy(data["X_train"]).to(device)
    X_val = torch.from_numpy(data["X_val"]).to(device)
    y_val_np = data["y_val"].astype(np.int64)
    if task == "binary":
        y_train_t = torch.tensor(y_train, dtype=torch.float32, device=device).unsqueeze(1)
        num_out = 1
    else:
        y_train_t = torch.tensor(y_train, dtype=torch.long, device=device)
        num_out = int(meta["n_classes"])
    del data
    input_dim = X_train.shape[1]

    def objective(trial):
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 3e-4, 5e-3, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [2048, 4096, 8192]),
            "conv_filters": trial.suggest_categorical("conv_filters", [16, 32, 64]),
            "kernel_size": trial.suggest_categorical("kernel_size", [3, 5]),
            "gru_units": trial.suggest_categorical("gru_units", [32, 64, 128]),
            "dense_units": trial.suggest_categorical("dense_units", [16, 32, 64]),
            "dropout": trial.suggest_float("dropout", 0.1, 0.4),
            "loss": trial.suggest_categorical("loss", ["ce", "sqrt_weighted"]),
        }
        step = f"Trial {trial.number + 1}"
        status.start(step, ", ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                                     for k, v in params.items()))
        cfg.set_seeds(cfg.GLOBAL_SEED)
        model = CNN_GRU(input_dim, num_out, conv_filters=params["conv_filters"],
                        kernel_size=params["kernel_size"], gru_units=params["gru_units"],
                        dense_units=params["dense_units"], dropout=params["dropout"]).to(device)
        criterion = build_criterion(task, params["loss"], y_train, num_out, device)
        optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])

        best, stale, t0 = 0.0, 0, time.time()
        try:
            for epoch in range(args.epochs):
                model.train()
                for X_b, y_b in iterate_batches(X_train, y_train_t, params["batch_size"], shuffle=True):
                    optimizer.zero_grad()
                    loss = criterion(model(X_b), y_b)
                    loss.backward()
                    optimizer.step()
                score = val_macro_f1(model, X_val, y_val_np, task, params["batch_size"])
                _write_progress(experiment_id=step, model="CNN-GRU (tuning)", phase="training",
                                epoch=epoch + 1, total_epochs=args.epochs, train_loss=float(loss.item()),
                                val_macro_f1=float(score), best_val_macro_f1=float(max(best, score)),
                                patience_counter=stale, patience_limit=3, elapsed_s=round(time.time() - t0, 1))
                if score > best:
                    best, stale = score, 0
                else:
                    stale += 1
                trial.report(score, epoch)
                if trial.should_prune():
                    status.done(step, f"pruned at epoch {epoch + 1} (val macro-F1 {best:.4f})")
                    raise optuna.TrialPruned()
                if stale >= 3:
                    break
        except optuna.TrialPruned:
            raise
        except Exception as e:
            status.fail(step, e)
            raise
        status.done(step, f"val macro-F1 {best:.4f}")
        return best

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=cfg.GLOBAL_SEED),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3))
    study.optimize(objective, n_trials=args.trials)

    best = study.best_trial
    logger.info(f"Best trial {best.number + 1}: val macro-F1 {best.value:.4f} | {best.params}")
    out_dir = cfg.EXPERIMENTS_DIR / f"{args.prefix}_{tag}_tuning"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json({**best.params, "best_val_macro_f1": best.value, "trial": best.number + 1,
               "n_trials": args.trials, "preprocessing": preprocessing_record(meta)}, out_dir / "best_params.json")
    study.trials_dataframe().to_csv(out_dir / "optuna_trials.csv", index=False)

    del X_train, X_val, y_train_t
    torch.cuda.empty_cache()

    p = best.params
    model_kwargs = {k: p[k] for k in ("conv_filters", "kernel_size", "gru_units", "dense_units", "dropout")}
    for seed, exp_id in zip(args.seeds, final_ids):
        status.start(exp_id, f"best config, seed {seed}, {args.epochs} epochs, validation-tuned threshold")
        record = train_dl_model(exp_id, "CNN-GRU", meta_target, task, loss_type=p["loss"], epochs=args.epochs,
                                seed=seed, tune_threshold=task == "binary", model_kwargs=model_kwargs,
                                learning_rate=p["learning_rate"], batch_size=p["batch_size"])
        status.done(exp_id, f"acc {record['accuracy']:.4f} | macro-F1 {record['macro_f1']:.4f}"
                            + (f" | FNR {record['fnr']:.3f}" if record.get("fnr") is not None else ""))
    status.finish()
    logger.info("Tuning complete.")


if __name__ == "__main__":
    main()
