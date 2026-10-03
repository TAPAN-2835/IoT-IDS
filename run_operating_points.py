"""
run_operating_points.py

The detection / false-alarm trade-off of the final binary models. For each
target false-alarm rate, the threshold is chosen on the VALIDATION split and the
resulting detection rate and false-alarm rate are measured on the TEST split.

Writes results/operating_points.csv.

Usage:
    python run_operating_points.py
"""
import os

os.environ.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")

import json

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, roc_auc_score

import src.config as cfg
from src.ciciot import CICIOT_STRICT_DIR
from src.explainability import _architecture_kwargs
from src.models import CNN_GRU
from src.training import iterate_batches, load_processed_data
from src.utils import setup_logger

logger = setup_logger("operating_points")

# (dataset label, processed data dir, CNN-GRU experiment, XGBoost experiment)
MODELS = [
    ("Edge-IIoTset (strict, no MQTT)", cfg.DATA_DIR / "processed",
     "F_strict_no_mqtt_binary_best_s42", "N01_xgb_binary_strict_nomqtt"),
    ("CICIoT2023 (strict)", CICIOT_STRICT_DIR / "binary",
     "CS02_cnn_gru_binary_strict", "CS01_xgb_binary_strict"),
]
TARGET_FPR = [0.01, 0.02, 0.05, 0.10]


def cnn_gru_probs(exp_id, X_by_split, device):
    exp_dir = cfg.EXPERIMENTS_DIR / exp_id
    model = CNN_GRU(next(iter(X_by_split.values())).shape[1], 1, **_architecture_kwargs(exp_dir)).to(device)
    model.load_state_dict(torch.load(exp_dir / "best_model.pt", map_location=device))
    model.eval()
    out = {}
    with torch.no_grad():
        for split, X in X_by_split.items():
            X_t = torch.from_numpy(X).to(device)
            out[split] = torch.cat([torch.sigmoid(model(b)).squeeze(1)
                                    for b, _ in iterate_batches(X_t, X_t, 8192)]).cpu().numpy()
    return out


def xgb_probs(exp_id, X_by_split):
    import xgboost as xgb
    booster = xgb.Booster()
    booster.load_model(cfg.EXPERIMENTS_DIR / exp_id / "model.ubj")
    booster.set_param({"device": "cuda" if torch.cuda.is_available() else "cpu"})
    return {split: booster.inplace_predict(X) for split, X in X_by_split.items()}


def threshold_for_fpr(probs_benign, target):
    """Smallest threshold whose false-alarm rate on benign rows is <= target."""
    return float(np.quantile(probs_benign, 1 - target))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for dataset, data_dir, cnn_id, xgb_id in MODELS:
        cfg.PROCESSED_DATA_DIR = data_dir
        record = json.load(open(cfg.EXPERIMENTS_DIR / cnn_id / "experiment_record.json"))
        data, meta = load_processed_data(record["preprocessing"]["target_col"])
        if meta["fingerprint"] != record["preprocessing"]["fingerprint"]:
            raise RuntimeError(f"{data_dir} does not hold the data {cnn_id} was trained on.")
        X = {"val": data["X_val"], "test": data["X_test"]}
        y = {"val": data["y_val"].astype(int), "test": data["y_test"].astype(int)}
        for model_name, probs in (("CNN-GRU", cnn_gru_probs(cnn_id, X, device)), ("XGBoost", xgb_probs(xgb_id, X))):
            auc = roc_auc_score(y["test"], probs["test"])
            points = [("default (0.5)", 0.5)] + [(f"false alarms <= {int(t * 100)}%",
                                                  threshold_for_fpr(probs["val"][y["val"] == 0], t))
                                                 for t in TARGET_FPR]
            for label, thr in points:
                pred = (probs["test"] > thr).astype(int)
                benign, attack = y["test"] == 0, y["test"] == 1
                rows.append({"dataset": dataset, "model": model_name, "operating_point": label,
                             "threshold": round(thr, 4),
                             "detection_rate": float(pred[attack].mean()),
                             "false_alarm_rate": float(pred[benign].mean()),
                             "macro_f1": f1_score(y["test"], pred, average="macro"),
                             "roc_auc": auc})
            logger.info(f"{dataset} | {model_name}: ROC-AUC {auc:.4f}")
    df = pd.DataFrame(rows)
    df.to_csv(cfg.RESULTS_DIR / "operating_points.csv", index=False)
    logger.info("\n" + df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
