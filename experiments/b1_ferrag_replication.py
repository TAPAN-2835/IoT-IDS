import time
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
import sys
import os

# Add root to sys.path to allow importing from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import get_csv_files, canonical_token
from src import config as cfg

out_file = Path('results/paper/ferrag_replication.json')
out_file.parent.mkdir(parents=True, exist_ok=True)

# 1. reported
results = {
    "reported": {
        "accuracy": 0.9999,
        "macro_f1": 0.9999,
        "best_stump_feature": None,
        "best_stump_bal_acc": None,
        "top_10_features": []
    }
}

csv_files = get_csv_files()
if not csv_files:
    raise FileNotFoundError("Raw CSV not found")
raw_csv = csv_files[0]

print(f"Loading raw CSV: {raw_csv.name}")
df_raw = pd.read_csv(raw_csv, engine="pyarrow")

ferrag_drops = [
    "frame.time", "ip.src_host", "ip.dst_host", "arp.src.proto_ipv4",
    "arp.dst.proto_ipv4", "http.file_data", "http.request.full_uri",
    "icmp.transmit_timestamp", "http.request.uri.query", "tcp.options",
    "tcp.payload", "tcp.srcport", "tcp.dstport", "udp.port", "mqtt.msg"
]

def clean_and_prep(df_in, drop_cols, canonicalize=False, policy_cols=None):
    df = df_in.copy()
    if drop_cols:
        existing_drops = [c for c in drop_cols if c in df.columns]
        df.drop(columns=existing_drops, inplace=True)
    if policy_cols is not None:
        keep = policy_cols + ["Attack_label", "Attack_type"]
        drop = [c for c in df.columns if c not in keep]
        df.drop(columns=drop, inplace=True)
        
    df.drop_duplicates(inplace=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    
    target_bin = df.pop("Attack_label").astype(int)
    target_multi = df.pop("Attack_type")
    
    string_cols = df.select_dtypes(include=['object', 'string']).columns
    if canonicalize:
        for c in string_cols:
            df[c] = df[c].apply(canonical_token)
            
    # Drop high cardinality string columns to avoid OOM
    to_drop = []
    for c in string_cols:
        if df[c].nunique() > 100:
            to_drop.append(c)
    if to_drop:
        print(f"Dropping high-cardinality cols: {to_drop}")
        df.drop(columns=to_drop, inplace=True)
    
    string_cols = [c for c in string_cols if c not in to_drop]
    df = pd.get_dummies(df, columns=string_cols, dtype=np.float32)
    return df, target_bin, target_multi

def run_experiment(name, df_X, y_bin, y_multi):
    print(f"--- Running {name} ---")
    print(f"Shape: {df_X.shape}")
    
    X_train, X_test, y_train, y_test = train_test_split(df_X, y_bin, test_size=0.2, stratify=y_bin, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Train XGBoost
    t0 = time.time()
    xgb = XGBClassifier(tree_method="hist", random_state=42, n_jobs=-1)
    xgb.fit(X_train, y_train)
    print(f"XGB fit time: {time.time()-t0:.1f}s")
    y_pred = xgb.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))
    mac_f1 = float(f1_score(y_test, y_pred, average="macro"))
    
    # Feature importances
    importances = xgb.feature_importances_
    top_indices = np.argsort(importances)[::-1][:10]
    feature_names = df_X.columns
    top_10 = [feature_names[i] for i in top_indices]
    
    # Stump scan (sample 100k to save time, or use all validation?)
    # The prompt: "Run the single-feature stump scan"
    # We will do it on X_test to save time, or a subset of X_train
    sample_size = min(300_000, len(X_test))
    idx = np.random.RandomState(42).choice(len(X_test), sample_size, replace=False)
    X_samp = X_test[idx]
    y_samp = y_test.iloc[idx].values if hasattr(y_test, "iloc") else y_test[idx]
    
    best_stump_bal_acc = 0
    best_stump_feature = None
    
    # Check top 50 features from XGB to save time, or all?
    # Scanning all can be slow if there are thousands of dummies.
    # We scan all features since we need "the best single-feature".
    for i, col in enumerate(feature_names):
        stump = DecisionTreeClassifier(max_depth=1, random_state=42)
        stump.fit(X_train[:, [i]], y_train)
        pred = stump.predict(X_samp[:, [i]])
        bal_acc = balanced_accuracy_score(y_samp, pred)
        if bal_acc > best_stump_bal_acc:
            best_stump_bal_acc = bal_acc
            best_stump_feature = col

    print(f"{name} Binary XGB Accuracy: {acc:.4f}, Macro-F1: {mac_f1:.4f}")
    print(f"Best stump: {best_stump_feature} ({best_stump_bal_acc:.4f})")
    
    results[name] = {
        "accuracy": acc,
        "macro_f1": mac_f1,
        "best_stump_feature": best_stump_feature,
        "best_stump_bal_acc": float(best_stump_bal_acc),
        "top_10_features": top_10
    }

# 2. replicated
print("Preparing 'replicated' dataset...")
df_X_rep, y_bin_rep, y_multi_rep = clean_and_prep(df_raw, ferrag_drops, canonicalize=False)
run_experiment("replicated", df_X_rep, y_bin_rep, y_multi_rep)
del df_X_rep

# 3. canonicalised
print("Preparing 'canonicalised' dataset...")
df_X_can, y_bin_can, y_multi_can = clean_and_prep(df_raw, ferrag_drops, canonicalize=True)
run_experiment("canonicalised", df_X_can, y_bin_can, y_multi_can)
del df_X_can

# 4. strict
print("Preparing 'strict' dataset...")
policy_path = cfg.AUDIT_DIR / "strict_no_mqtt_feature_policy.csv"
strict_features = pd.read_csv(policy_path)["feature"].tolist()
df_X_str, y_bin_str, y_multi_str = clean_and_prep(df_raw, drop_cols=None, canonicalize=True, policy_cols=strict_features)
run_experiment("strict", df_X_str, y_bin_str, y_multi_str)
del df_X_str

with open(out_file, 'w') as f:
    json.dump(results, f, indent=2)
print("Saved B1 results to", out_file)
