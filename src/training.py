import time
import os
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score, confusion_matrix

from src import config as cfg
from src.utils import setup_logger, save_json
from src.models import CNN1D, GRUBaseline, CNN_GRU, MLP
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss.sum() if self.reduction == 'sum' else focal_loss

logger = setup_logger(__name__)

PROGRESS_PATH = cfg.BASE_DIR / "training_progress.json"


def load_processed_data(expected_target: str):
    """Load data/processed/* as float32 numpy arrays and verify what it contains.

    Raises if the processed data was built for a different target, so a binary
    model can never silently train or evaluate on a multiclass split (or vice versa).
    """
    from src.preprocessing import load_processed_metadata
    meta = load_processed_metadata()
    if meta["target_col"] != expected_target:
        raise RuntimeError(
            f"data/processed/ holds {meta['target_col']!r} data (policy={meta['feature_policy']}), "
            f"but {expected_target!r} was requested. Re-run preprocessing for this target first.")

    def _read(name):
        return pd.read_parquet(cfg.PROCESSED_DATA_DIR / f"{name}.parquet").to_numpy()

    arrays = {name: _read(name) for name in
              ("X_train", "y_train", "X_val", "y_val", "X_test", "y_test")}
    for name in ("X_train", "X_val", "X_test"):
        arrays[name] = arrays[name].astype(np.float32, copy=False)
    for name in ("y_train", "y_val", "y_test"):
        arrays[name] = arrays[name].squeeze()
    # if arrays["X_train"].shape[1] != meta["n_features"]:
    #     raise RuntimeError(f"data/processed/metadata.json does not match the parquet files (expected {meta['n_features']} but got {arrays['X_train'].shape[1]}); re-run preprocessing.")
    import pyarrow as pa
    pa.default_memory_pool().release_unused()  # hand parquet read buffers back to the OS
    return arrays, meta


def preprocessing_record(meta: dict) -> dict:
    """Subset of processed-data metadata stored with each experiment for traceability."""
    return {k: meta[k] for k in ("fingerprint", "target_col", "feature_policy",
                                 "canonicalize_numeric_tokens", "split_seed", "n_features")}


def choose_threshold(model, X_val_t, y_val_bin, batch_size):
    """Decision threshold that maximises validation macro-F1 (binary models)."""
    model.eval()
    with torch.no_grad():
        probs = torch.cat([torch.sigmoid(model(X_b)).squeeze(1)
                           for X_b, _ in iterate_batches(X_val_t, X_val_t, batch_size)]).cpu().numpy()
    grid = np.arange(0.05, 0.96, 0.01)
    scores = [f1_score(y_val_bin, (probs > t).astype(int), average="macro", zero_division=0) for t in grid]
    return float(grid[int(np.argmax(scores))])


def iterate_batches(X, y, batch_size, shuffle=False):
    """Yield (X, y) mini-batches from tensors that already live on the target device."""
    n = X.shape[0]
    order = torch.randperm(n, device=X.device) if shuffle else None
    for start in range(0, n, batch_size):
        idx = order[start:start + batch_size] if shuffle else slice(start, start + batch_size)
        yield X[idx], y[idx]


def _write_progress(**kwargs):
    """Persist live epoch/loss progress to disk so the dashboard can poll it."""
    try:
        from src.run_status import atomic_write_json
        payload = {"updated_at": time.strftime("%H:%M:%S"), **kwargs}
        atomic_write_json(PROGRESS_PATH, payload)
    except Exception as e:
        logger.warning(f"Could not write training_progress.json: {e}")


def train_dl_model(experiment_id, model_name, target_col, task_type, loss_type="ce", epochs=None, seed=None,
                   tune_threshold=False, model_kwargs=None, learning_rate=None, batch_size=None,
                   drop_features=None):
    epochs = epochs or cfg.EPOCHS
    seed = cfg.GLOBAL_SEED if seed is None else seed
    batch_size = batch_size or cfg.BATCH_SIZE
    learning_rate = learning_rate or cfg.LEARNING_RATE
    logger.info(f"Starting {experiment_id} using {model_name} on {target_col} ({epochs} epochs, loss={loss_type})")
    cfg.set_seeds(seed)
    _write_progress(experiment_id=experiment_id, model=model_name, phase="loading_data",
                     epoch=0, total_epochs=epochs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    exp_dir = cfg.EXPERIMENTS_DIR / experiment_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info("Loading parquet datasets...")
    data, meta = load_processed_data(target_col)
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]
    feature_names = list(meta["feature_names"])
    if drop_features:
        # Feature-removal experiments (e.g. SHAP fidelity): drop columns after loading.
        keep = [i for i, f in enumerate(feature_names) if f not in set(drop_features)]
        missing = set(drop_features) - set(feature_names)
        if missing:
            raise ValueError(f"drop_features not in data: {sorted(missing)}")
        X_train, X_val, X_test = X_train[:, keep], X_val[:, keep], X_test[:, keep]
        feature_names = [feature_names[i] for i in keep]
        logger.info(f"Dropped {len(drop_features)} features; training on {len(feature_names)}")
    n_test = len(X_test)
    # Benign class index ("Normal" in Edge-IIoTset, "Benign" in CICIoT2023)
    normal_idx = next((int(k) for k, v in meta["label_mapping"].items() if v in ("Normal", "Benign")), 7)

    input_dim = X_train.shape[1]

    if task_type == "binary":
        num_classes = 1
        criterion = nn.BCEWithLogitsLoss()
        if target_col == "Attack_label":
            y_train_bin = y_train.astype(np.float32)
            y_val_bin = y_val.astype(np.float32)
            y_test_bin = y_test.astype(np.float32)
        else:
            # Everything that is not the Normal class is an Attack.
            y_train_bin = (y_train != normal_idx).astype(np.float32)
            y_val_bin = (y_val != normal_idx).astype(np.float32)
            y_test_bin = (y_test != normal_idx).astype(np.float32)

        if loss_type == "sqrt_weighted":
            # Attacks are ~27% of rows; sqrt(neg/pos) up-weights them without over-correcting.
            pos_weight = float(np.sqrt((y_train_bin == 0).sum() / max((y_train_bin == 1).sum(), 1)))
            criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))

        y_train_t = torch.tensor(y_train_bin).unsqueeze(1)
        y_val_t = torch.tensor(y_val_bin).unsqueeze(1)
        y_test_t = torch.tensor(y_test_bin).unsqueeze(1)
    else:
        num_classes = len(np.unique(y_train))

        if loss_type == "ce":
            criterion = nn.CrossEntropyLoss()
        elif loss_type in ("class_weighted", "sqrt_weighted"):
            class_counts = np.bincount(y_train)
            weights = len(y_train) / (num_classes * class_counts)
            if loss_type == "sqrt_weighted":
                # Raw inverse-frequency weights span ~1600:1 on this data and made
                # training collapse (C05); the square root keeps the ratio near 40:1.
                weights = np.sqrt(weights)
            criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32).to(device))
        elif loss_type == "focal":
            criterion = FocalLoss(gamma=2.0)
        else:
            criterion = nn.CrossEntropyLoss()

        y_train_t = torch.tensor(y_train, dtype=torch.long)
        y_val_t = torch.tensor(y_val, dtype=torch.long)
        y_test_t = torch.tensor(y_test, dtype=torch.long)

    # Move every split to the device once (~0.6 GB of VRAM for the full dataset) and
    # free the host copies, so training does not hold the dataset in system RAM.
    X_train_t = torch.from_numpy(X_train).to(device)
    X_val_t = torch.from_numpy(X_val).to(device)
    X_test_t = torch.from_numpy(X_test).to(device)
    y_train_t, y_val_t, y_test_t = y_train_t.to(device), y_val_t.to(device), y_test_t.to(device)
    del X_train, X_val, X_test, data
    n_train_batches = -(-len(X_train_t) // batch_size)
    n_val_batches = -(-len(X_val_t) // batch_size)

    # Initialize Model
    if model_name == "1D-CNN":
        model = CNN1D(input_dim, num_classes).to(device)
    elif model_name == "GRU":
        model = GRUBaseline(input_dim, num_classes).to(device)
    elif model_name == "MLP":
        model = MLP(input_dim, num_classes).to(device)
    elif model_name == "CNN-GRU":
        model = CNN_GRU(input_dim, num_classes, **(model_kwargs or {})).to(device)
    else:
        raise ValueError("Unknown model name")

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training Loop with Early Stopping
    logger.info("Training Model...")
    t0 = time.time()

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_path = exp_dir / "best_model.pt"
    history = {"train_loss": [], "val_loss": []}
    _write_progress(experiment_id=experiment_id, model=model_name, phase="training",
                     epoch=0, total_epochs=epochs, elapsed_s=0)

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for X_b, y_b in iterate_batches(X_train_t, y_train_t, batch_size, shuffle=True):
            optimizer.zero_grad()
            outputs = model(X_b)
            loss = criterion(outputs, y_b)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_b, y_b in iterate_batches(X_val_t, y_val_t, batch_size):
                outputs = model(X_b)
                loss = criterion(outputs, y_b)
                val_loss += loss.item()

        train_loss /= n_train_batches
        val_loss /= n_val_batches

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        logger.info(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1

        _write_progress(experiment_id=experiment_id, model=model_name, phase="training",
                         epoch=epoch + 1, total_epochs=epochs,
                         train_loss=round(train_loss, 6), val_loss=round(val_loss, 6),
                         best_val_loss=round(best_val_loss, 6),
                         patience_counter=patience_counter, patience_limit=cfg.EARLY_STOPPING_PATIENCE,
                         elapsed_s=round(time.time() - t0, 1))

        if patience_counter >= cfg.EARLY_STOPPING_PATIENCE:
            logger.info(f"Early stopping triggered at epoch {epoch+1}")
            break

    training_time = time.time() - t0

    save_json(history, exp_dir / "training_history.json")

    _write_progress(experiment_id=experiment_id, model=model_name, phase="evaluating",
                     epoch=epoch + 1, total_epochs=epochs, elapsed_s=round(training_time, 1))

    # Load best model for inference
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    if model_name == "CNN-GRU":
        # Save final architecture as requested
        torch.save(model.state_dict(), cfg.MODELS_DIR / "cnn_gru_final.pt")

    # Inference
    logger.info("Evaluating on Test Set...")
    model.eval()
    t1 = time.time()

    all_preds = []
    with torch.no_grad():
        for X_b, _ in iterate_batches(X_test_t, y_test_t, batch_size):
            outputs = model(X_b)
            if task_type == "binary":
                preds = torch.sigmoid(outputs).squeeze(1).cpu().numpy()  # probabilities
            else:
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.append(preds)

    inference_time = time.time() - t1
    y_out = np.concatenate(all_preds)

    threshold, macro_f1_at_half = 0.5, None
    if task_type == "binary":
        if tune_threshold:
            # Chosen on the VALIDATION split only; the test split is never used for tuning.
            threshold = choose_threshold(model, X_val_t, y_val_bin, batch_size)
            macro_f1_at_half = f1_score(y_test_bin, (y_out > 0.5).astype(int), average="macro", zero_division=0)
            logger.info(f"Validation-tuned threshold {threshold:.2f} (test macro-F1 at 0.5 would be {macro_f1_at_half:.4f})")
        y_pred = (y_out > threshold).astype(int)
    else:
        y_pred = y_out

    if task_type == "binary":
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

    from sklearn.metrics import classification_report
    import matplotlib.pyplot as plt
    import seaborn as sns

    true_labels = y_test_bin if task_type == 'binary' else y_test
    report_dict = classification_report(true_labels, y_pred, output_dict=True, zero_division=0)
    pd.DataFrame(report_dict).transpose().to_csv(exp_dir / "classification_report.csv")

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=False, cmap='Blues', fmt='g')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - {experiment_id}')
    plt.savefig(exp_dir / "confusion_matrix.png", dpi=150, bbox_inches='tight')
    plt.close()

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_mb = os.path.getsize(best_model_path) / (1024 * 1024)

    # Update Registry
    from datetime import datetime
    registry_path = cfg.RESULTS_DIR / "experiment_registry.csv"
    if registry_path.exists():
        df = pd.read_csv(registry_path)
    else:
        df = pd.DataFrame()

    features_used = feature_names

    new_record = {
        "experiment_id": experiment_id,
        "timestamp": datetime.now().isoformat(),
        "model": model_name,
        "dataset": "Edge-IIoTset",
        "feature_policy": meta["feature_policy"],
        "split": "stratified_random",
        "seed": seed,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "fpr": fpr,
        "fnr": fnr,
        "training_time": training_time,
        "inference_latency": (inference_time / n_test) * 1000,
        "parameters": param_count,
        "model_size_mb": model_size_mb,
        "features_used": features_used,
        "loss_type": loss_type,
        "threshold": threshold,
        "macro_f1_at_0.5": macro_f1_at_half,
        "canonicalize_numeric_tokens": meta["canonicalize_numeric_tokens"],
        "epochs_run": epoch + 1,
        "device": str(device),
        "hyperparameters": {
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "epochs": epochs,
            **(model_kwargs or {}),
        },
        "preprocessing": preprocessing_record(meta),
        "dropped_features": list(drop_features or []),
    }

    # Save the full config as JSON
    save_json(new_record, exp_dir / "experiment_record.json")

    # Save to CSV (excluding complex objects like features_used and hyperparameters)
    csv_record = {k: v for k, v in new_record.items()
                  if k not in ["features_used", "hyperparameters", "preprocessing", "macro_f1_at_0.5"]}
    df = pd.concat([df, pd.DataFrame([csv_record])], ignore_index=True)
    df.to_csv(registry_path, index=False)

    logger.info(f"Finished {experiment_id}. Accuracy: {accuracy:.4f}")
    _write_progress(experiment_id=experiment_id, model=model_name, phase="complete",
                     epoch=epoch + 1, total_epochs=epochs, elapsed_s=round(training_time, 1),
                     accuracy=round(float(accuracy), 4))
    return new_record

if __name__ == "__main__":
    import sys
    exp_id = sys.argv[1]
    model_n = sys.argv[2]
    target = sys.argv[3]
    task = sys.argv[4]
    loss_t = sys.argv[5] if len(sys.argv) > 5 else "ce"
    train_dl_model(exp_id, model_n, target, task, loss_type=loss_t)
