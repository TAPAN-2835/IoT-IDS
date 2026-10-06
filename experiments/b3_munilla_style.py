import time
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from xgboost import XGBClassifier
import shap
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_loader import get_csv_files, canonical_token
from src import config as cfg

out_file = Path('results/paper/munilla_style.json')
out_file.parent.mkdir(parents=True, exist_ok=True)

csv_files = get_csv_files()
raw_csv = csv_files[0]
print(f"Loading raw CSV: {raw_csv.name}")
df_raw = pd.read_csv(raw_csv, engine="pyarrow")

def clean_and_prep(df_in, policy_cols=None):
    df = df_in.copy()
    
    if policy_cols is not None:
        keep = policy_cols + ["Attack_label", "Attack_type"]
        drop = [c for c in df.columns if c not in keep]
        df.drop(columns=drop, inplace=True)
        
    df.drop_duplicates(inplace=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    
    target_bin = df.pop("Attack_label").astype(int)
    if "Attack_type" in df.columns:
        df.pop("Attack_type")
    
    string_cols = df.select_dtypes(include=['object', 'string']).columns
    # Canonicalize
    for c in string_cols:
        df[c] = df[c].apply(canonical_token)
        
    # Drop high cardinality
    to_drop = [c for c in string_cols if df[c].nunique() > 100]
    if to_drop:
        df.drop(columns=to_drop, inplace=True)
        string_cols = [c for c in string_cols if c not in to_drop]
        
    df = pd.get_dummies(df, columns=string_cols, dtype=np.float32)
    return df, target_bin

def run_xgb_shap(name, df_X, y_bin):
    print(f"--- Running {name} ---")
    X_train, X_test, y_train, y_test = train_test_split(df_X, y_bin, test_size=0.2, stratify=y_bin, random_state=42)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    xgb = XGBClassifier(tree_method="hist", random_state=42, n_jobs=-1)
    xgb.fit(X_train_s, y_train)
    y_pred = xgb.predict(X_test_s)
    mac_f1 = float(f1_score(y_test, y_pred, average="macro"))
    print(f"{name} Macro-F1: {mac_f1:.4f}")
    
    # SHAP explainer
    explainer = shap.TreeExplainer(xgb)
    
    # Use a background sample to speed up SHAP values calculation if it's too large
    # Exact TreeSHAP is fast, but 380k rows x 100 cols might take a bit.
    # We'll use 100k samples
    sample_size = min(100_000, len(X_test_s))
    np.random.seed(42)
    idx = np.random.choice(len(X_test_s), sample_size, replace=False)
    X_samp = X_test_s[idx]
    
    t0 = time.time()
    shap_values = explainer.shap_values(X_samp)
    print(f"SHAP time: {time.time()-t0:.1f}s")
    
    # Global feature importance is mean absolute SHAP value
    if isinstance(shap_values, list): # For multiclass or old SHAP
        mean_abs_shap = np.abs(shap_values[1]).mean(axis=0)
    else:
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        
    top_indices = np.argsort(mean_abs_shap)[::-1][:10]
    feature_names = df_X.columns
    top_10 = [feature_names[i] for i in top_indices]
    print(f"{name} Top 10: {top_10}")
    
    return mac_f1, top_10

print("Preparing 'all features' dataset...")
df_X_all, y_bin_all = clean_and_prep(df_raw, policy_cols=None)
f1_all, top10_all = run_xgb_shap("All Features", df_X_all, y_bin_all)
del df_X_all

print("Preparing 'strict_no_mqtt' dataset...")
policy_path = cfg.AUDIT_DIR / "strict_no_mqtt_feature_policy.csv"
strict_features = pd.read_csv(policy_path)["feature"].tolist()
df_X_str, y_bin_str = clean_and_prep(df_raw, policy_cols=strict_features)
f1_str, top10_str = run_xgb_shap("Strict Features", df_X_str, y_bin_str)
del df_X_str

jaccard = len(set(top10_all) & set(top10_str)) / len(set(top10_all) | set(top10_str))

# shortcut columns in all-feature top-10
# (empty-field indicators, MQTT, IDs, ports, IPs)
# Usually port is 'tcp.srcport', 'udp.port', IPs are 'ip.src', mqtt starts with mqtt, ID starts with tcp.seq, empty fields end with _0.0 or _0
prone_keywords = ['mqtt', 'port', 'ip.', 'seq', 'ack', 'checksum', '_0.0', '_0']
def is_shortcut(f):
    for k in prone_keywords:
        if k in f: return True
    return False

shortcut_count = sum(1 for f in top10_all if is_shortcut(f))

results = {
    "macro_f1_all": f1_all,
    "macro_f1_strict": f1_str,
    "top_10_all": top10_all,
    "top_10_strict": top10_str,
    "jaccard": jaccard,
    "shortcut_count": shortcut_count
}

with open(out_file, 'w') as f:
    json.dump(results, f, indent=2)

print("Saved B3 results to", out_file)