"""
run_edge_benchmark.py

Edge-feasibility benchmark on the CPU only (an IoT gateway has no GPU):
  - final CNN-GRU, FP32
  - same model with INT8 dynamic quantization (GRU + Linear layers)
  - XGBoost baseline on the same data
For each: test macro-F1 / FNR / FPR, file size, single-packet latency (1 thread),
and batch throughput.

Writes results/edge_benchmark.csv and .json.

Usage:
    python run_edge_benchmark.py --model F_strict_no_mqtt_binary_best_s42 --xgb N01_xgb_binary_strict_nomqtt
"""
import os

os.environ.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")

import argparse
import io
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, f1_score

import src.config as cfg
from src.explainability import _architecture_kwargs
from src.models import CNN_GRU
from src.training import load_processed_data
from src.utils import save_json, setup_logger

logger = setup_logger("edge_benchmark")


def _rates(y, pred):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"macro_f1": f1_score(y, pred, average="macro"), "fpr": fp / (fp + tn), "fnr": fn / (fn + tp)}


def _state_dict_bytes(model):
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.tell()


def _latency(predict_one, X, n=2000):
    for i in range(50):  # warm-up
        predict_one(X[i:i + 1])
    times = []
    for i in range(n):
        t0 = time.perf_counter()
        predict_one(X[i:i + 1])
        times.append(time.perf_counter() - t0)
    return 1000 * float(np.median(times)), 1000 * float(np.percentile(times, 99))


def _throughput(predict_batch, X, batch=1024, rounds=20):
    t0 = time.perf_counter()
    for r in range(rounds):
        predict_batch(X[(r * batch) % (len(X) - batch):][:batch])
    return batch * rounds / (time.perf_counter() - t0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="experiment ID of the final CNN-GRU")
    parser.add_argument("--xgb", required=True, help="experiment ID of the XGBoost baseline (same data)")
    args = parser.parse_args()

    torch.set_num_threads(1)  # a small gateway core
    exp_dir = cfg.EXPERIMENTS_DIR / args.model
    record = json.load(open(exp_dir / "experiment_record.json"))
    data, meta = load_processed_data(record["preprocessing"]["target_col"])
    if meta["fingerprint"] != record["preprocessing"]["fingerprint"]:
        raise RuntimeError("data/processed/ does not match the model's training data; re-run its preprocessing.")
    X_test, y_test = data["X_test"], data["y_test"].astype(int)
    threshold = record.get("threshold", 0.5)

    fp32 = CNN_GRU(X_test.shape[1], 1, **_architecture_kwargs(exp_dir))
    fp32.load_state_dict(torch.load(exp_dir / "best_model.pt", map_location="cpu"))
    fp32.eval()
    int8_all = torch.ao.quantization.quantize_dynamic(fp32, {nn.GRU, nn.Linear}, dtype=torch.qint8).eval()
    int8_linear = torch.ao.quantization.quantize_dynamic(fp32, {nn.Linear}, dtype=torch.qint8).eval()
    # FP16 storage: weights saved in half precision (half the file), upcast to FP32 to run on the CPU.
    fp16_state = {k: v.half() for k, v in fp32.state_dict().items()}
    fp16 = CNN_GRU(X_test.shape[1], 1, **_architecture_kwargs(exp_dir))
    fp16.load_state_dict({k: v.float() for k, v in fp16_state.items()})
    fp16.eval()
    buf = io.BytesIO()
    torch.save(fp16_state, buf)
    fp16_bytes = buf.tell()

    variants = [
        ("CNN-GRU FP32", fp32, _state_dict_bytes(fp32)),
        ("CNN-GRU FP16 weights", fp16, fp16_bytes),
        ("CNN-GRU INT8 Linear only", int8_linear, _state_dict_bytes(int8_linear)),
        ("CNN-GRU INT8 GRU+Linear", int8_all, _state_dict_bytes(int8_all)),
    ]
    rows = []
    X_t = torch.from_numpy(X_test)
    for name, model, size_bytes in variants:
        logger.info(f"Benchmarking {name} on CPU...")
        with torch.no_grad():
            probs = torch.cat([torch.sigmoid(model(X_t[i:i + 8192])).squeeze(1)
                               for i in range(0, len(X_t), 8192)]).numpy()
            one = lambda x: model(torch.from_numpy(x))
            p50, p99 = _latency(one, X_test)
            tput = _throughput(one, X_test)
        rows.append({"model": name, **_rates(y_test, (probs > threshold).astype(int)),
                     "size_kb": size_bytes / 1024,
                     "parameters": sum(p.numel() for p in fp32.parameters()),
                     "latency_ms_p50": p50, "latency_ms_p99": p99, "throughput_rows_per_s": tput})

    import xgboost as xgb
    xgb_record = json.load(open(cfg.EXPERIMENTS_DIR / args.xgb / "experiment_record.json"))
    if xgb_record["preprocessing"]["fingerprint"] != meta["fingerprint"]:
        raise RuntimeError(f"{args.xgb} was trained on different data than {args.model}.")
    xgb_path = cfg.EXPERIMENTS_DIR / args.xgb / "model.ubj"
    booster = xgb.Booster()
    booster.load_model(xgb_path)
    booster.set_param({"device": "cpu", "nthread": 1})
    logger.info("Benchmarking XGBoost on CPU...")
    probs = booster.inplace_predict(X_test)
    one = lambda x: booster.inplace_predict(x)
    p50, p99 = _latency(one, X_test)
    rows.append({"model": "XGBoost (CPU)", **_rates(y_test, (probs > 0.5).astype(int)),
                 "size_kb": os.path.getsize(xgb_path) / 1024, "parameters": None,
                 "latency_ms_p50": p50, "latency_ms_p99": p99,
                 "throughput_rows_per_s": _throughput(one, X_test)})

    df = pd.DataFrame(rows)
    df.to_csv(cfg.RESULTS_DIR / "edge_benchmark.csv", index=False)
    save_json({"model": args.model, "xgb": args.xgb, "threshold": threshold, "cpu_threads": 1,
               "rows": rows}, cfg.RESULTS_DIR / "edge_benchmark.json")
    logger.info("\n" + df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
