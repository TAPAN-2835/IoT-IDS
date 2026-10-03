"""
IoT-IDS Dashboard API
FastAPI backend serving experiment data, audit info, and model artifacts.
"""

import csv
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="IoT-IDS Dashboard API", version="1.0.0")


@app.on_event("startup")
def _warm_demo():
    from dashboard.demo import warm_up
    warm_up()

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR = BASE_DIR / "models"
STATIC_DIR = Path(__file__).parent / "static"

# ──────────────────────────────────────────────────────────────────────────────
# Helper
# ──────────────────────────────────────────────────────────────────────────────

def _safe_float(val):
    try:
        return float(val) if val not in (None, "", "null") else None
    except (ValueError, TypeError):
        return None


def _read_csv_as_dicts(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ──────────────────────────────────────────────────────────────────────────────
# API Routes
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/api/status")
def get_status():
    """Return the project phase roadmap and completion status."""
    return {
        "phases": [
            {
                "id": 1,
                "name": "Data Audit & Shortcut Removal",
                "status": "complete",
                "description": "First models scored a fake 100%: empty fields were written \"0\" in Normal and \"0.0\" in attack captures. Canonicalised tokens, removed per-packet identifiers and MQTT, added automatic shortcut scans.",
                "artifacts": ["empty_token_audit.csv", "strict_no_mqtt_feature_policy.csv", "detection_ceiling.json"],
            },
            {
                "id": 2,
                "name": "Honest Baselines & Tuning",
                "status": "complete",
                "description": "Edge-IIoTset Normal vs Attack: CNN-GRU 0.858 Macro-F1 (3 seeds), XGBoost 0.868. Optuna (20 trials) adds nothing: models sit at the measured detection ceiling.",
                "artifacts": ["F_strict_no_mqtt_binary_best_s42", "N01_xgb_binary_strict_nomqtt"],
            },
            {
                "id": 3,
                "name": "Ablation: Is the Hybrid Needed?",
                "status": "complete",
                "description": "MLP, 1D-CNN, GRU and CNN-GRU all reach 0.858 on per-packet features; the MLP is smallest (24 KB).",
                "artifacts": ["A_mlp_s42", "A_cnn1d_s42", "A_gru_s42"],
            },
            {
                "id": 4,
                "name": "Explainability with Fidelity Proof",
                "status": "complete",
                "description": "SHAP top features: tcp.flags, RST/ACK/FIN, tcp.len. Removing the top-5 SHAP features drops Macro-F1 0.858 to 0.605; removing 5 random ones changes nothing.",
                "artifacts": ["FID_drop_top5_shap", "FID_drop_random5"],
            },
            {
                "id": 5,
                "name": "Edge Benchmark & Second Dataset",
                "status": "complete",
                "description": "FP16 model 159 KB with no loss. CICIoT2023 (flow data, window-size shortcut removed): binary 0.917 XGBoost / 0.890 CNN-GRU, 8 categories 0.752 / 0.675.",
                "artifacts": ["CS01_xgb_binary_strict", "CS02_cnn_gru_binary_strict", "CS05_cnn_gru_category_sqrt_strict"],
            },
            {
                "id": 6,
                "name": "Next: Real Time Sequences",
                "status": "pending",
                "description": "Feed the GRU consecutive flow windows so it can learn how traffic changes over time; cross-dataset testing; Raspberry-Pi deployment.",
                "artifacts": [],
            },
        ]
    }


@app.get("/api/experiments")
def get_experiments():
    """Return all experiments: completed ones from registry + planned future ones."""
    registry_path = RESULTS_DIR / "experiment_registry.csv"
    completed = {}

    if registry_path.exists():
        for row in _read_csv_as_dicts(registry_path):
            numeric_fields = [
                "accuracy", "precision", "recall", "f1",
                "macro_f1", "weighted_f1", "fpr", "fnr",
                "training_time", "inference_latency", "parameters", "model_size_mb",
            ]
            for field in numeric_fields:
                row[field] = _safe_float(row.get(field))
            row["status"] = "complete"
            row["has_artifacts"] = (RESULTS_DIR / "experiments" / row["experiment_id"]).exists()
            completed[row["experiment_id"]] = row

    # Experiments shown on the dashboard, in story order (leaky originals first, for contrast)
    planned_all = [
        {"experiment_id": "E01_random_forest_binary",          "model": "Random Forest (leaky original)", "task": "Binary · Edge-IIoTset",      "phase": 1},
        {"experiment_id": "E05_cnn_gru_binary",                "model": "CNN-GRU (leaky original)",       "task": "Binary · Edge-IIoTset",      "phase": 1},
        {"experiment_id": "N01_xgb_binary_strict_nomqtt",      "model": "XGBoost",                        "task": "Binary · Edge-IIoTset",      "phase": 2},
        {"experiment_id": "F_strict_no_mqtt_binary_best_s42",  "model": "CNN-GRU (final, tuned)",         "task": "Binary · Edge-IIoTset",      "phase": 2},
        {"experiment_id": "A_mlp_s42",                         "model": "MLP",                            "task": "Binary · Edge-IIoTset",      "phase": 3},
        {"experiment_id": "A_cnn1d_s42",                       "model": "1D-CNN",                         "task": "Binary · Edge-IIoTset",      "phase": 3},
        {"experiment_id": "A_gru_s42",                         "model": "GRU",                            "task": "Binary · Edge-IIoTset",      "phase": 3},
        {"experiment_id": "FID_drop_top5_shap",                "model": "CNN-GRU without top-5 SHAP",     "task": "Fidelity test",              "phase": 4},
        {"experiment_id": "FID_drop_random5",                  "model": "CNN-GRU without 5 random",       "task": "Fidelity test",              "phase": 4},
        {"experiment_id": "S03_xgb_multiclass_strict",         "model": "XGBoost",                        "task": "Attack type · Edge-IIoTset", "phase": 3},
        {"experiment_id": "L_strict_sqrt_weighted_s42",        "model": "CNN-GRU (sqrt weights)",         "task": "Attack type · Edge-IIoTset", "phase": 3},
        {"experiment_id": "CS01_xgb_binary_strict",            "model": "XGBoost",                        "task": "Binary · CICIoT2023",        "phase": 5},
        {"experiment_id": "CS02_cnn_gru_binary_strict",        "model": "CNN-GRU",                        "task": "Binary · CICIoT2023",        "phase": 5},
        {"experiment_id": "CS03_xgb_category_strict",          "model": "XGBoost",                        "task": "8 categories · CICIoT2023",  "phase": 5},
        {"experiment_id": "CS05_cnn_gru_category_sqrt_strict", "model": "CNN-GRU (sqrt weights)",         "task": "8 categories · CICIoT2023",  "phase": 5},
    ]

    result = []
    for p in planned_all:
        if p["experiment_id"] in completed:
            row = completed[p["experiment_id"]]
            row["phase"] = p["phase"]
            row["task"] = p["task"]
            result.append(row)
        else:
            exp_dir = RESULTS_DIR / "experiments" / p["experiment_id"]
            status = "in_progress" if p["phase"] == 3 and p["task"] == "Binary" else "pending"
            # If results exist, mark complete
            if exp_dir.exists() and (exp_dir / "experiment_record.json").exists():
                status = "complete"
            result.append({
                **p,
                "status": status,
                "has_artifacts": exp_dir.exists(),
                "accuracy": None, "macro_f1": None,
                "inference_latency": None, "model_size_mb": None,
                "training_time": None, "parameters": None,
            })

    return {"experiments": result}


@app.get("/api/experiments/{exp_id}")
def get_experiment_detail(exp_id: str):
    """Return detailed record for one experiment including classification report and feature importance."""
    exp_dir = RESULTS_DIR / "experiments" / exp_id
    if not exp_dir.exists():
        raise HTTPException(status_code=404, detail=f"Experiment '{exp_id}' not found or not yet run.")

    record_path = exp_dir / "experiment_record.json"
    if not record_path.exists():
        raise HTTPException(status_code=404, detail="experiment_record.json missing.")

    with open(record_path, "r", encoding="utf-8") as f:
        record = json.load(f)

    # Classification report
    report_path = exp_dir / "classification_report.csv"
    if report_path.exists():
        record["classification_report"] = _read_csv_as_dicts(report_path)

    # Feature importance (top 20)
    fi_path = exp_dir / "feature_importance.csv"
    if fi_path.exists():
        rows = _read_csv_as_dicts(fi_path)
        # Normalise column names: may be 'feature'/'importance' or similar
        fi = []
        for r in rows:
            keys = list(r.keys())
            feat_key = next((k for k in keys if "feat" in k.lower()), keys[0])
            imp_key = next((k for k in keys if "imp" in k.lower()), keys[1] if len(keys) > 1 else keys[0])
            fi.append({"feature": r[feat_key], "importance": _safe_float(r.get(imp_key, 0))})
        fi.sort(key=lambda x: x["importance"] or 0, reverse=True)
        record["feature_importance"] = fi[:20]

    # Available images
    record["images"] = [f.name for f in exp_dir.iterdir() if f.suffix == ".png"]

    return record


@app.get("/api/experiments/{exp_id}/images/{filename}")
def get_experiment_image(exp_id: str, filename: str):
    """Serve a PNG artifact from an experiment directory."""
    image_path = RESULTS_DIR / "experiments" / exp_id / filename
    if not image_path.exists() or image_path.suffix.lower() != ".png":
        raise HTTPException(status_code=404, detail="Image not found.")
    return FileResponse(str(image_path), media_type="image/png")


@app.get("/api/audit")
def get_audit():
    """Return dataset audit summary and leakage candidates."""
    result: dict = {}

    summary_path = RESULTS_DIR / "audit" / "dataset_summary.json"
    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as f:
            result["dataset_summary"] = json.load(f)

    leakage_path = RESULTS_DIR / "audit" / "leakage_candidates.txt"
    if leakage_path.exists():
        with open(leakage_path, "r", encoding="utf-8") as f:
            result["leakage_candidates"] = f.read().strip().split("\n")

    policy_path = RESULTS_DIR / "audit" / "operational_feature_policy.csv"
    if policy_path.exists():
        rows = _read_csv_as_dicts(policy_path)
        result["feature_policy_count"] = len(rows)
        result["retained_features"] = [r for r in rows if r.get("decision", "").lower() == "retain"]
        result["dropped_features"] = [r for r in rows if r.get("decision", "").lower() != "retain"]

    return result


@app.get("/api/labels")
def get_labels():
    """Return integer → attack-name label mapping."""
    label_path = MODELS_DIR / "label_mapping.json"
    if not label_path.exists():
        raise HTTPException(status_code=404, detail="label_mapping.json not found.")
    with open(label_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _demo_or_503():
    from dashboard.demo import DemoUnavailable, get_demo
    try:
        return get_demo()
    except DemoUnavailable as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/api/demo/info")
def demo_info():
    """Final model used by the live demo, with its test-set metrics."""
    return _demo_or_503().info()


@app.get("/api/demo/sample")
def demo_sample(kind: str = "any"):
    """Pick a random held-out test packet: kind = attack | normal | any."""
    demo = _demo_or_503()
    sample_id = demo.random_sample(kind)
    return {"sample_id": sample_id, "actual": "ATTACK" if demo.y[sample_id] == 1 else "NORMAL"}


@app.post("/api/predict")
def predict(payload: dict):
    """Classify one held-out test packet and explain the verdict with SHAP.

    Body: {"sample_id": <int from /api/demo/sample>}
    """
    demo = _demo_or_503()
    try:
        return demo.predict(int(payload.get("sample_id", -1)))
    except (ValueError, TypeError) as e:
        raise HTTPException(status_code=400, detail=str(e))


# ──────────────────────────────────────────────────────────────────────────────
# Static file serving
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/api/pipeline-status")
def get_pipeline_status():
    """Live pipeline status — updated every 2s by pipeline_monitor.py."""
    status_path = BASE_DIR / "pipeline_status.json"
    if not status_path.exists():
        # Auto-generate a basic status if monitor isn't running
        return JSONResponse({"error": "pipeline_status.json not found. Run: python pipeline_monitor.py"})
    with open(status_path, "r", encoding="utf-8") as f:
        return json.load(f)

@app.get("/api/logs")
def get_logs():
    """Live training logs, tailed straight from pipeline.log.

    pipeline.log is appended to by every logger created via
    src.utils.setup_logger() (preprocessing, training, explainability),
    so this reflects whatever pipeline stage is currently running,
    including per-epoch training progress.
    """
    log_path = BASE_DIR / "pipeline.log"

    if not log_path.exists():
        return {"logs": ["No training logs available yet — pipeline.log has not been created."]}

    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        log_lines = [line.strip() for line in lines if line.strip()][-60:]
    except Exception as e:
        log_lines = [f"Error reading logs: {e}"]

    return {"logs": log_lines}


@app.get("/api/training-progress")
def get_training_progress():
    """Live epoch-level progress for whichever experiment is currently training,
    written incrementally by src/training.py after every epoch."""
    progress_path = BASE_DIR / "training_progress.json"
    if not progress_path.exists():
        return JSONResponse({"status": "idle", "message": "No training in progress."})
    with open(progress_path, "r", encoding="utf-8") as f:
        return json.load(f)



@app.get("/pipeline")
def serve_pipeline():
    return FileResponse(str(STATIC_DIR / "pipeline.html"))


@app.get("/")
def serve_index():
    return FileResponse(str(STATIC_DIR / "index.html"))


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
