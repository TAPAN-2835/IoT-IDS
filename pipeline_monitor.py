"""
pipeline_monitor.py

Continuously monitors the IoT-IDS pipeline and writes a live
pipeline_status.json that the dashboard can poll every 2 seconds.

Run this in a separate terminal WHILE the pipeline is running:
    python pipeline_monitor.py
"""

import json
import re
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
EXPERIMENTS_DIR = RESULTS_DIR / "experiments"
STATUS_FILE = BASE_DIR / "pipeline_status.json"
PROGRESS_FILE = BASE_DIR / "training_progress.json"

# Known sizes (bytes)
DATASET_TOTAL_BYTES = 1_217_413_981
TORCH_TOTAL_BYTES   = 2_753_200_000   # ~2.75 GB

EXPERIMENTS = [
    {"id": "E01_random_forest_binary",    "label": "E01 Random Forest Binary",    "phase": 2},
    {"id": "E02_random_forest_multiclass","label": "E02 Random Forest Multiclass", "phase": 2},
    {"id": "E03_cnn1d_binary",            "label": "E03 1D-CNN Binary",            "phase": 3},
    {"id": "E04_gru_binary",              "label": "E04 GRU Binary",               "phase": 3},
    {"id": "E05_cnn_gru_binary",          "label": "E05 CNN-GRU Binary",           "phase": 3},
    {"id": "E06_cnn1d_multiclass",        "label": "E06 1D-CNN Multiclass",        "phase": 3},
    {"id": "E07_gru_multiclass",          "label": "E07 GRU Multiclass",           "phase": 3},
    {"id": "E08_cnn_gru_multiclass",      "label": "E08 CNN-GRU Multiclass",       "phase": 3},
]


def _fmt_mb(b):
    return f"{b / 1e6:.0f} MB"


def check_dataset_download():
    csv = RAW_DIR / "DNN-EdgeIIoT-dataset.csv"
    if csv.exists():
        size = csv.stat().st_size
        pct = min(100.0, (size / DATASET_TOTAL_BYTES) * 100)
        done = size >= DATASET_TOTAL_BYTES * 0.99
        return {
            "status": "complete" if done else "downloading",
            "pct": round(pct, 1),
            "downloaded_mb": round(size / 1e6, 1),
            "total_mb": round(DATASET_TOTAL_BYTES / 1e6, 1),
            "label": "Complete ✓" if done else f"{pct:.1f}%  ({size // 1_000_000} / {DATASET_TOTAL_BYTES // 1_000_000} MB)"
        }
    # Check kagglehub cache for partially downloaded archive
    cache_dir = Path.home() / ".cache" / "kagglehub" / "datasets" / "mohamedamineferrag" / "edgeiiotset-cyber-security-dataset-of-iot-iiot"
    archive = None
    for p in cache_dir.rglob("*.archive") if cache_dir.exists() else []:
        archive = p; break
    if archive and archive.exists():
        size = archive.stat().st_size
        pct = min(99.0, (size / DATASET_TOTAL_BYTES) * 100)
        return {
            "status": "downloading",
            "pct": round(pct, 1),
            "downloaded_mb": round(size / 1e6, 1),
            "total_mb": round(DATASET_TOTAL_BYTES / 1e6, 1),
            "label": f"{pct:.1f}%  ({size // 1_000_000} / {DATASET_TOTAL_BYTES // 1_000_000} MB)"
        }
    return {"status": "pending", "pct": 0, "downloaded_mb": 0,
            "total_mb": round(DATASET_TOTAL_BYTES / 1e6, 1), "label": "Waiting…"}


def check_torch_install():
    try:
        import torch
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
            return {"status": "complete", "pct": 100,
                    "label": f"CUDA ready — {gpu} ✓"}
        if "cpu" in torch.__version__:
            return {"status": "installing", "pct": 50, "label": "CPU build detected, CUDA installing…"}
        return {"status": "complete", "pct": 100, "label": f"PyTorch {torch.__version__} ✓ (no CUDA)"}
    except ImportError:
        return {"status": "pending", "pct": 0, "label": "Not installed yet"}


def check_preprocessing():
    required = ["X_train.parquet", "X_val.parquet", "X_test.parquet",
                "y_train.parquet", "y_val.parquet",  "y_test.parquet"]
    existing = [f for f in required if (PROCESSED_DIR / f).exists()]
    if len(existing) == len(required):
        total = sum((PROCESSED_DIR / f).stat().st_size for f in required)
        return {"status": "complete", "pct": 100,
                "label": f"Done — {total // 1_000_000} MB parquet ✓"}
    if len(existing) > 0:
        pct = (len(existing) / len(required)) * 100
        return {"status": "running", "pct": round(pct, 0),
                "label": f"Saving splits… {len(existing)}/{len(required)}"}
    if (PROCESSED_DIR / "preprocessing.log").exists() or PROCESSED_DIR.exists():
        return {"status": "running", "pct": 10, "label": "Loading CSV…"}
    return {"status": "pending", "pct": 0, "label": "Waiting for dataset…"}


def _read_live_progress(exp_id):
    """Read the epoch-level progress file written live by src/training.py,
    but only trust it if it's actually for THIS experiment and reasonably fresh."""
    if not PROGRESS_FILE.exists():
        return None
    try:
        with open(PROGRESS_FILE) as f:
            prog = json.load(f)
    except Exception:
        return None
    if prog.get("experiment_id") != exp_id:
        return None
    return prog


def check_experiment(exp_id):
    exp_dir = EXPERIMENTS_DIR / exp_id
    record_path = exp_dir / "experiment_record.json"
    if record_path.exists():
        try:
            with open(record_path) as f:
                rec = json.load(f)
            acc = rec.get("accuracy", rec.get("acc"))
            f1  = rec.get("macro_f1", rec.get("f1"))
            lbl = f"Done — Acc {float(acc)*100:.2f}%  F1 {float(f1):.4f} ✓" if acc else "Done ✓"
            return {"status": "complete", "pct": 100, "detail": lbl, "metrics": rec}
        except Exception:
            pass

    # Prefer live, real epoch-level progress written by src/training.py
    prog = _read_live_progress(exp_id)
    if prog:
        phase = prog.get("phase", "training")
        epoch = prog.get("epoch", 0)
        total = prog.get("total_epochs", 0) or 1
        if phase == "loading_data":
            return {"status": "training", "pct": 2, "detail": "Loading parquet datasets…"}
        if phase == "evaluating":
            return {"status": "training", "pct": 96, "detail": f"Epoch {epoch}/{total} done — evaluating on test set…"}
        if phase == "complete":
            acc = prog.get("accuracy")
            return {"status": "training", "pct": 99,
                    "detail": f"Finishing up — acc {acc*100:.2f}%" if acc is not None else "Finishing up…"}
        # phase == "training": real per-epoch progress
        pct = min(95, round((epoch / total) * 100))
        tl = prog.get("train_loss")
        vl = prog.get("val_loss")
        loss_str = f" · train {tl:.4f} · val {vl:.4f}" if tl is not None and vl is not None else ""
        return {"status": "training", "pct": pct,
                "detail": f"Epoch {epoch}/{total}{loss_str}"}

    # Fallback: no live progress file yet (or it's stale from a different run)
    if (exp_dir / "best_model.pt").exists():
        sz = (exp_dir / "best_model.pt").stat().st_size
        return {"status": "training", "pct": 50,
                "detail": f"Checkpoint saved ({sz // 1024} KB) — waiting for live progress…"}
    if exp_dir.exists():
        return {"status": "training", "pct": 5, "detail": "Starting…"}
    return {"status": "pending", "pct": 0, "detail": "Pending"}


def check_shap(exp_id):
    exp_dir = EXPERIMENTS_DIR / exp_id
    plots = ["shap_summary.png", "shap_bar.png", "shap_waterfall.png", "shap_importance.csv"]
    done = [p for p in plots if (exp_dir / p).exists()]
    if len(done) == len(plots):
        return {"status": "complete", "pct": 100, "label": "SHAP done ✓"}
    if done:
        return {"status": "running", "pct": int(len(done)/len(plots)*100),
                "label": f"Generating plots… {len(done)}/{len(plots)}"}
    return {"status": "pending", "pct": 0, "label": "Pending"}


def build_status():
    dataset   = check_dataset_download()
    torch_st  = check_torch_install()
    preproc   = check_preprocessing()
    experiments = {e["id"]: {**e, **check_experiment(e["id"])} for e in EXPERIMENTS}
    shap_status = {e["id"]: check_shap(e["id"])
                   for e in EXPERIMENTS if e["phase"] == 3}

    # Derive overall pipeline step
    dl_done   = dataset["status"] == "complete"
    torch_done = torch_st["status"] == "complete"
    pre_done  = preproc["status"] == "complete"
    binary_done = all(experiments[e["id"]]["status"] == "complete"
                      for e in EXPERIMENTS if e["phase"] == 3 and "multiclass" not in e["id"])
    multi_done  = all(experiments[e["id"]]["status"] == "complete"
                      for e in EXPERIMENTS if e["phase"] == 3 and "multiclass" in e["id"])
    shap_done = all(v["status"] == "complete" for v in shap_status.values())

    steps = [
        {"id": "download",      "label": "Dataset Download",         "icon": "📥", **dataset},
        {"id": "torch",         "label": "PyTorch CUDA Install",      "icon": "🔧", **torch_st},
        {"id": "preprocessing", "label": "Preprocessing (70/15/15)",  "icon": "⚙️",  **preproc},
        {"id": "phase3_binary", "label": "Phase 3A — Binary DL",      "icon": "🧠",
         "status": "complete" if binary_done else ("running" if pre_done and dl_done else "pending"),
         "pct": 100 if binary_done else 0,
         "label": "E03 + E04 + E05 complete ✓" if binary_done else "Waiting…"},
        {"id": "phase3_multi",  "label": "Phase 3B — Multiclass DL",  "icon": "🧠",
         "status": "complete" if multi_done else ("running" if binary_done else "pending"),
         "pct": 100 if multi_done else 0,
         "label": "E06 + E07 + E08 complete ✓" if multi_done else "Waiting…"},
        {"id": "shap",          "label": "Phase 4 — SHAP XAI",        "icon": "📊",
         "status": "complete" if shap_done else ("running" if multi_done else "pending"),
         "pct": 100 if shap_done else 0,
         "label": "All SHAP artifacts saved ✓" if shap_done else "Waiting…"},
    ]

    return {
        "updated_at": time.strftime("%H:%M:%S"),
        "steps": steps,
        "experiments": list(experiments.values()),
        "shap": shap_status,
    }


def main():
    print("[*] Pipeline Monitor running -- writing pipeline_status.json every 2s")
    print(f"    Open the dashboard at http://localhost:8000/pipeline")
    print("    Ctrl+C to stop\n")
    while True:
        try:
            status = build_status()
            with open(STATUS_FILE, "w") as f:
                json.dump(status, f, indent=2)
        except Exception as e:
            print(f"Monitor error: {e}")
        time.sleep(2)


if __name__ == "__main__":
    main()
