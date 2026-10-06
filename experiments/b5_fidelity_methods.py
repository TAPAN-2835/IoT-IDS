import os
import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from xgboost import XGBClassifier

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import config as cfg
from src.preprocessing import load_processed_metadata
from src.explainability import _load_model, _architecture_kwargs
from src.training import train_dl_model

out_file = Path('results/paper/fidelity_methods.json')
out_file.parent.mkdir(parents=True, exist_ok=True)
fig_file = Path('paper/figures/fidelity_methods.png')
fig_file.parent.mkdir(parents=True, exist_ok=True)

# 1. Load feature names
import json
feature_names = json.load(open(cfg.MODELS_DIR / "feature_columns.json"))["features"]

print("Loading validation data for permutation importance...")
X_val = pd.read_parquet(cfg.PROCESSED_DATA_DIR / "X_val.parquet").values.astype(np.float32)
y_val = pd.read_parquet(cfg.PROCESSED_DATA_DIR / "y_val.parquet").values.ravel()

# Load the base model F_strict_no_mqtt_binary_best_s42
final_id = "F_strict_no_mqtt_binary_best_s42"
exp_dir = cfg.EXPERIMENTS_DIR / final_id
best_params = json.load(open(cfg.EXPERIMENTS_DIR / "F_strict_no_mqtt_binary_tuning" / "best_params.json"))

# Get baseline F1
with open(exp_dir / "experiment_record.json") as f:
    baseline_f1 = json.load(f)["macro_f1"]
print(f"Baseline F1: {baseline_f1:.4f}")

# a) SHAP top-5
try:
    with open(cfg.RESULTS_DIR / "shap_fidelity.json") as f:
        shap_fidelity = json.load(f)
        top5_shap = shap_fidelity["top5_shap"]
except Exception as e:
    print("Could not load shap_fidelity.json, using fallback names.")
    top5_shap = [feature_names[i] for i in range(5)]

print(f"SHAP top 5: {top5_shap}")

# b) Permutation importance on validation
print("Computing permutation importance...")
input_dim = X_val.shape[1]
# Let's train the baseline model fresh so we have the exact weights for permutation
common = dict(loss_type=best_params["loss"], epochs=cfg.EPOCHS, tune_threshold=True,
              learning_rate=best_params["learning_rate"], batch_size=best_params["batch_size"])
model_kwargs = {k: best_params[k] for k in ("conv_filters", "kernel_size", "gru_units", "dense_units", "dropout")}

print("Training baseline CNN-GRU for permutation importance...")
record = train_dl_model("B5_baseline", "CNN-GRU", "Attack_label", "binary", seed=42, model_kwargs=model_kwargs, **common)
baseline_f1 = record["macro_f1"]
print(f"Baseline F1: {baseline_f1:.4f}")

model = _load_model("CNN-GRU", input_dim, 1, cfg.EXPERIMENTS_DIR / "B5_baseline")

# Helper for scikit-learn to score PyTorch model
def score_func(estimator, X, y):
    model.eval()
    with torch.no_grad():
        out = model(torch.tensor(X))
        pred = (torch.sigmoid(out).numpy() > 0.5).astype(int).ravel()
    from sklearn.metrics import f1_score
    return f1_score(y, pred, average="macro")

# Subsample val for permutation to be faster (e.g. 20k rows)
np.random.seed(42)
idx = np.random.choice(len(X_val), min(20000, len(X_val)), replace=False)
X_val_samp = X_val[idx]
y_val_samp = y_val[idx]

perm_res = permutation_importance(model, X_val_samp, y_val_samp, scoring=score_func, n_repeats=3, random_state=42, n_jobs=-1)
top5_perm = [feature_names[i] for i in np.argsort(perm_res.importances_mean)[::-1][:5]]
print(f"Permutation top 5: {top5_perm}")

# c) XGBoost gain
print("Training XGBoost for gain top-5...")
X_train = pd.read_parquet(cfg.PROCESSED_DATA_DIR / "X_train.parquet").values.astype(np.float32)
y_train = pd.read_parquet(cfg.PROCESSED_DATA_DIR / "y_train.parquet").values.ravel()
xgb = XGBClassifier(tree_method="hist", n_jobs=-1, random_state=42)
xgb.fit(X_train, y_train)
top5_xgb = [feature_names[i] for i in np.argsort(xgb.feature_importances_)[::-1][:5]]
print(f"XGBoost top 5: {top5_xgb}")
del X_train, y_train # free memory

# d) 20 random 5-feature draws
others = [f for f in feature_names if f not in set(top5_shap + top5_perm + top5_xgb)]
random_draws = []
for i in range(20):
    random_draws.append(list(np.random.choice(others, 5, replace=False)))

# Now we retrain the CNN-GRU configuration
common = dict(loss_type=best_params["loss"], epochs=cfg.EPOCHS, tune_threshold=True,
              learning_rate=best_params["learning_rate"], batch_size=best_params["batch_size"])
model_kwargs = {k: best_params[k] for k in ("conv_filters", "kernel_size", "gru_units", "dense_units", "dropout")}

results = {"baseline": baseline_f1}
all_drops = {}

def evaluate_drop(name, drop_list):
    exp_id = f"B5_{name}"
    print(f"Retraining without {name}: {drop_list}")
    record = train_dl_model(exp_id, "CNN-GRU", "Attack_label", "binary", seed=42,
                            model_kwargs=model_kwargs, drop_features=drop_list, **common)
    f1 = record["macro_f1"]
    drop = baseline_f1 - f1
    print(f" -> F1: {f1:.4f} (Drop: {drop:.4f})")
    all_drops[name] = drop
    return f1

results["shap"] = evaluate_drop("shap", top5_shap)
results["permutation"] = evaluate_drop("permutation", top5_perm)
results["xgb_gain"] = evaluate_drop("xgb_gain", top5_xgb)

rand_f1s = []
for i, r_draw in enumerate(random_draws):
    f1 = evaluate_drop(f"random_{i}", r_draw)
    rand_f1s.append(f1)

results["random_mean"] = float(np.mean(rand_f1s))
results["random_sd"] = float(np.std(rand_f1s))

with open(out_file, 'w') as f:
    json.dump(results, f, indent=2)

print("Saved B5 results to", out_file)

# Plotting
plt.figure(figsize=(8, 6))
labels = ['SHAP', 'Permutation', 'XGB Gain', 'Random (Mean)']
values = [results['shap'], results['permutation'], results['xgb_gain'], results['random_mean']]
y_pos = np.arange(len(labels))
plt.barh(y_pos, values, xerr=[0, 0, 0, results['random_sd']], align='center', alpha=0.8, color='c')
plt.yticks(y_pos, labels)
plt.xlabel('Macro-F1 after removing top-5 features')
plt.title('Explainability Fidelity (Lower F1 = better explanation)')
plt.axvline(x=baseline_f1, color='r', linestyle='--', label='Baseline F1')
plt.legend()
plt.tight_layout()
plt.savefig(fig_file)
print("Saved B5 figure to", fig_file)