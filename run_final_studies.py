"""
run_final_studies.py

Final binary studies on the strict + no-MQTT data, after run_tune_cnn_gru.py:

  1. Ablation   MLP / 1D-CNN / GRU with the tuned CNN-GRU's training settings
                (loss, learning rate, batch size, 15 epochs, validation-tuned
                threshold), 3 seeds each -> A_<model>_s<seed>. The tuned CNN-GRU
                runs are F_<tag>_best_s<seed>.
  2. SHAP       explain the final CNN-GRU (seed 42).
  3. Fidelity   retrain the final CNN-GRU without its top-5 SHAP features and,
                as a control, without 5 random other features. If SHAP reflects
                what the model uses, the first drop must be clearly larger.
  4. Edge       CPU benchmark (FP32 / FP16 / INT8 vs XGBoost), run_edge_benchmark.py.

Progress appears in watch_dashboard.py.
"""
import os

os.environ.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")

import argparse
import json
import subprocess
import sys

import numpy as np
import pandas as pd
import torch

import src.config as cfg
from run_clean_baselines import be_nice
from src.explainability import run_shap_analysis
from src.preprocessing import load_processed_metadata
from src.run_status import RunStatus
from src.training import train_dl_model
from src.utils import setup_logger, save_json

logger = setup_logger("final_studies")

TAG = "strict_no_mqtt_binary"
XGB_BASELINE = "N01_xgb_binary_strict_nomqtt"
ABLATION_MODELS = {"MLP": "mlp", "1D-CNN": "cnn1d", "GRU": "gru"}


def _result(record):
    return f"acc {record['accuracy']:.4f} | macro-F1 {record['macro_f1']:.4f} | FNR {record['fnr']:.3f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default="F")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 2024])
    args = parser.parse_args()
    be_nice()

    meta = load_processed_metadata()
    if meta["feature_policy"] != "strict_no_mqtt" or meta["target_col"] != "Attack_label":
        raise RuntimeError("data/processed/ must hold the strict_no_mqtt binary data "
                           "(python run_clean_baselines.py --policy strict_no_mqtt --stage binary).")
    best = json.load(open(cfg.EXPERIMENTS_DIR / f"{args.prefix}_{TAG}_tuning" / "best_params.json"))
    final_id = f"{args.prefix}_{TAG}_best_s{args.seeds[0]}"
    model_kwargs = {k: best[k] for k in ("conv_filters", "kernel_size", "gru_units", "dense_units", "dropout")}
    common = dict(loss_type=best["loss"], epochs=cfg.EPOCHS, tune_threshold=True,
                  learning_rate=best["learning_rate"], batch_size=best["batch_size"])

    ablation = [(f"A_{slug}_s{seed}", model, seed) for model, slug in ABLATION_MODELS.items() for seed in args.seeds]
    steps = [a[0] for a in ablation] + ["SHAP on final CNN-GRU", "FID_drop_top5_shap", "FID_drop_random5",
                                        "Edge benchmark (CPU)"]
    status = RunStatus(f"Final studies | {TAG} | ablation, SHAP, fidelity, edge", steps)

    for exp_id, model, seed in ablation:
        status.start(exp_id, f"{model}, loss={best['loss']}, seed {seed}, tuned training settings")
        record = train_dl_model(exp_id, model, "Attack_label", "binary", seed=seed, **common)
        status.done(exp_id, _result(record))
        torch.cuda.empty_cache()

    step = "SHAP on final CNN-GRU"
    status.start(step, final_id)
    shap_df = run_shap_analysis(final_id, "CNN-GRU", "binary", n_background=200, n_explain=1000)
    top5 = shap_df["raw_feature"].head(5).tolist()
    status.done(step, "top-5: " + ", ".join(shap_df["feature"].head(5)))

    rng = np.random.default_rng(cfg.GLOBAL_SEED)
    others = [f for f in meta["feature_names"] if f not in top5]
    random5 = list(rng.choice(others, size=5, replace=False))
    baseline = json.load(open(cfg.EXPERIMENTS_DIR / final_id / "experiment_record.json"))
    fidelity = {"final_model": final_id, "baseline_macro_f1": baseline["macro_f1"],
                "top5_shap": top5, "random5": random5}
    for exp_id, drop in (("FID_drop_top5_shap", top5), ("FID_drop_random5", random5)):
        status.start(exp_id, "retrain final CNN-GRU without: " + ", ".join(d.split("__")[-1] for d in drop))
        record = train_dl_model(exp_id, "CNN-GRU", "Attack_label", "binary", seed=args.seeds[0],
                                model_kwargs=model_kwargs, drop_features=drop, **common)
        fidelity[exp_id] = {"macro_f1": record["macro_f1"], "drop": baseline["macro_f1"] - record["macro_f1"]}
        status.done(exp_id, _result(record) + f" | drop {fidelity[exp_id]['drop']:+.4f}")
        torch.cuda.empty_cache()
    save_json(fidelity, cfg.RESULTS_DIR / "shap_fidelity.json")

    step = "Edge benchmark (CPU)"
    status.start(step, "FP32 / FP16 / INT8 / XGBoost, 1 CPU thread")
    proc = subprocess.run([sys.executable, "run_edge_benchmark.py", "--model", final_id, "--xgb", XGB_BASELINE],
                          cwd=cfg.BASE_DIR)
    if proc.returncode != 0:
        status.fail(step, f"exit code {proc.returncode}")
        raise SystemExit(1)
    bench = pd.read_csv(cfg.RESULTS_DIR / "edge_benchmark.csv")
    status.done(step, " | ".join(f"{r.model}: {r.size_kb:.0f} KB, {r.latency_ms_p50:.2f} ms" for r in bench.itertuples()))
    status.finish()


if __name__ == "__main__":
    main()
