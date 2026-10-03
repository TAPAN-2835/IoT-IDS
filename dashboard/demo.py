"""
dashboard/demo.py

Live-prediction demo for the dashboard: the final CNN-GRU classifies real
held-out test packets (never seen in training) and SHAP explains each verdict.

The model is only served with the exact processed data it was trained on
(fingerprint check); otherwise the endpoints return a clear 503 message.
"""
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

DEMO_EXPERIMENT = "F_strict_no_mqtt_binary_best_s42"
POOL_PER_CLASS = 300
BACKGROUND_ROWS = 100
PREP_HINT = "python run_clean_baselines.py --policy strict_no_mqtt --stage binary"


class DemoUnavailable(RuntimeError):
    pass


class DemoModel:
    def __init__(self):
        from src.explainability import _architecture_kwargs, _display_names
        from src.models import CNN_GRU

        exp_dir = BASE_DIR / "results" / "experiments" / DEMO_EXPERIMENT
        data_dir = BASE_DIR / "data" / "processed"
        if not (exp_dir / "best_model.pt").exists():
            raise DemoUnavailable(f"Model {DEMO_EXPERIMENT} not found. Run: python run_tune_cnn_gru.py")
        self.record = json.load(open(exp_dir / "experiment_record.json"))
        meta_path = data_dir / "metadata.json"
        if not meta_path.exists():
            raise DemoUnavailable(f"No processed data. Run: {PREP_HINT}")
        meta = json.load(open(meta_path))
        if meta["fingerprint"] != self.record["preprocessing"]["fingerprint"]:
            raise DemoUnavailable(
                f"data/processed/ holds different data ({meta['feature_policy']}, {meta['target_col']}) "
                f"than the demo model was trained on. Run: {PREP_HINT}")

        self.feature_names = meta["feature_names"]
        # "=0.0" is the canonical empty protocol field
        self.display_names = [n[:-4] + "=(empty)" if n.endswith("=0.0") else n
                              for n in _display_names(self.feature_names, meta.get("categorical_columns", []))]
        self.threshold = float(self.record.get("threshold", 0.5))

        # Pool of held-out test packets, balanced between classes.
        X_test = pd.read_parquet(data_dir / "X_test.parquet").to_numpy(dtype=np.float32)
        y_test = pd.read_parquet(data_dir / "y_test.parquet").iloc[:, 0].to_numpy().astype(int)
        rng = np.random.default_rng(0)
        idx = np.concatenate([rng.choice(np.flatnonzero(y_test == c), POOL_PER_CLASS, replace=False)
                              for c in (0, 1)])
        self.X = X_test[idx]
        self.y = y_test[idx]
        self.raw_values = self._raw_values(self.X)
        del X_test, y_test

        torch.set_num_threads(2)
        self.model = CNN_GRU(self.X.shape[1], 1, **_architecture_kwargs(exp_dir))
        self.model.load_state_dict(torch.load(exp_dir / "best_model.pt", map_location="cpu"))
        self.model.eval()

        # SHAP background: the first rows of the (already shuffled) training split.
        batch = next(pq.ParquetFile(data_dir / "X_train.parquet").iter_batches(batch_size=BACKGROUND_ROWS))
        background = torch.from_numpy(batch.to_pandas().to_numpy(dtype=np.float32))
        import shap
        self.explainer = shap.GradientExplainer(self.model, background)
        self.lock = threading.Lock()

    def _raw_values(self, X):
        """Un-scale numeric features for display (falls back to scaled values)."""
        raw = X.astype(np.float64).copy()
        try:
            import joblib
            pre = joblib.load(BASE_DIR / "models" / "preprocessor.joblib")
            if list(pre.get_feature_names_out()) == self.feature_names:
                scaler = pre.named_transformers_["num"].named_steps["scaler"]
                n = len(scaler.mean_)
                raw[:, :n] = X[:, :n] * scaler.scale_ + scaler.mean_
        except Exception:
            pass
        raw[np.abs(raw) < 1e-3] = 0.0  # float32 round-off from un-scaling (real values are >= 1e-3)
        return raw

    def info(self):
        r = self.record
        return {
            "experiment_id": DEMO_EXPERIMENT,
            "model": "CNN-GRU (tuned)",
            "dataset": "Edge-IIoTset, strict features without MQTT",
            "parameters": r["parameters"],
            "size_kb": round(r["model_size_mb"] * 1024, 1),
            "threshold": self.threshold,
            "test_macro_f1": r["macro_f1"],
            "test_accuracy": r["accuracy"],
            "test_false_alarm_rate": r["fpr"],
            "test_detection_rate": 1 - r["fnr"],
            "pool_size": int(len(self.y)),
        }

    def random_sample(self, kind):
        rng = np.random.default_rng()
        if kind == "attack":
            candidates = np.flatnonzero(self.y == 1)
        elif kind == "normal":
            candidates = np.flatnonzero(self.y == 0)
        else:
            candidates = np.arange(len(self.y))
        return int(rng.choice(candidates))

    def predict(self, sample_id, top_k=6):
        if not 0 <= sample_id < len(self.y):
            raise ValueError(f"sample_id must be between 0 and {len(self.y) - 1}")
        x = torch.from_numpy(self.X[sample_id:sample_id + 1])
        with self.lock, torch.no_grad():
            t0 = time.perf_counter()
            prob = float(torch.sigmoid(self.model(x)).item())
            latency_ms = (time.perf_counter() - t0) * 1000
        with self.lock:
            sv = np.asarray(self.explainer.shap_values(x, nsamples=100))
        sv = sv.reshape(-1)[: len(self.feature_names)]
        order = np.argsort(-np.abs(sv))[:top_k]
        reasons = []
        for j in order:
            name = self.display_names[j]
            raw = self.raw_values[sample_id, j]
            is_onehot = "=" in name
            value = ("present" if raw > 0.5 else "absent") if is_onehot else f"{raw:.4g}"
            reasons.append({"feature": name, "value": value, "contribution": float(sv[j]),
                            "pushes_towards": "attack" if sv[j] > 0 else "normal"})
        predicted_attack = prob > self.threshold
        actual_attack = bool(self.y[sample_id] == 1)
        return {
            "sample_id": sample_id,
            "probability_attack": prob,
            "threshold": self.threshold,
            "prediction": "ATTACK" if predicted_attack else "NORMAL",
            "actual": "ATTACK" if actual_attack else "NORMAL",
            "correct": predicted_attack == actual_attack,
            "latency_ms": latency_ms,
            "top_reasons": reasons,
        }


_demo = None
_demo_error = None
_init_lock = threading.Lock()


def warm_up():
    """Load the model and SHAP in the background so the first click is fast."""
    threading.Thread(target=lambda: _safe_get(), daemon=True).start()


def _safe_get():
    try:
        get_demo()
    except DemoUnavailable:
        pass


def get_demo():
    """Load once on first use; remember a failure so the page can explain it."""
    global _demo, _demo_error
    with _init_lock:
        if _demo is None and _demo_error is None:
            try:
                _demo = DemoModel()
            except DemoUnavailable as e:
                _demo_error = str(e)
        if _demo is None:
            raise DemoUnavailable(_demo_error)
        return _demo
