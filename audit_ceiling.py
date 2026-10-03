"""
audit_ceiling.py

How well can ANY model do on a processed binary dataset? Many rows share an
identical feature vector; if a pattern is mostly Normal in training, every
classifier must call it Normal, so the attack rows with that pattern are
undetectable from these features. This measures that ceiling.

Writes results/audit/detection_ceiling.json.

Usage:
    python audit_ceiling.py
"""
import json

import numpy as np
import pandas as pd

import src.config as cfg
from src.ciciot import CICIOT_STRICT_DIR
from src.utils import save_json

DATASETS = {
    "Edge-IIoTset (strict, no MQTT)": cfg.DATA_DIR / "processed",
    "CICIoT2023 (strict)": CICIOT_STRICT_DIR / "binary",
}


def ceiling(data_dir):
    meta = json.load(open(data_dir / "metadata.json"))
    X_tr = pd.read_parquet(data_dir / "X_train.parquet")
    y_tr = pd.read_parquet(data_dir / "y_train.parquet").iloc[:, 0].to_numpy()
    X_te = pd.read_parquet(data_dir / "X_test.parquet")
    y_te = pd.read_parquet(data_dir / "y_test.parquet").iloc[:, 0].to_numpy()

    h_tr = pd.util.hash_pandas_object(X_tr, index=False).to_numpy()
    h_te = pd.util.hash_pandas_object(X_te, index=False).to_numpy()
    attack_share = pd.Series(y_tr).groupby(h_tr).mean()          # per distinct pattern
    share_te = pd.Series(h_te).map(attack_share).to_numpy()
    seen = ~np.isnan(share_te)
    attack = y_te == 1
    lookup_pred = (share_te >= 0.5)
    return {
        "feature_policy": meta["feature_policy"],
        "train_rows": int(len(y_tr)),
        "distinct_train_patterns": int(len(attack_share)),
        "test_rows_with_seen_pattern": float(seen.mean()),
        "attacks_with_mostly_normal_pattern": float(((share_te < 0.5) & seen & attack).sum() / attack.sum()),
        "lookup_detection_rate_on_seen": float(lookup_pred[seen & attack].mean()) if (seen & attack).any() else None,
        "lookup_false_alarm_rate_on_seen": float(lookup_pred[seen & ~attack].mean()) if (seen & ~attack).any() else None,
    }


def main():
    out = {name: ceiling(path) for name, path in DATASETS.items()}
    save_json(out, cfg.AUDIT_DIR / "detection_ceiling.json")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
