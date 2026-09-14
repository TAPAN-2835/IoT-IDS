"""
src/explainability.py

SHAP-based explainability for trained IoT-IDS deep learning models.
Generates:
  - shap_summary.png   (beeswarm global attribution)
  - shap_bar.png       (mean |SHAP| bar chart)
  - shap_importance.csv (ranked feature importances from SHAP)

Usage:
    python run_explainability.py
"""

import json
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')   # headless rendering
import matplotlib.pyplot as plt
import shap

from src import config as cfg
from src.utils import setup_logger
from src.models import CNN1D, GRUBaseline, CNN_GRU

logger = setup_logger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Feature name helpers
# ──────────────────────────────────────────────────────────────────────────────

def _load_feature_names(input_dim: int) -> list[str]:
    """Load post-OHE feature names from feature_columns.json."""
    path = cfg.MODELS_DIR / "feature_columns.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        names = data.get("features", [])
        if len(names) == input_dim:
            return names
    return [f"feature_{i}" for i in range(input_dim)]


def _clean_name(name: str, maxlen: int = 32) -> str:
    """Strip sklearn ColumnTransformer prefixes and OHE suffixes for readability."""
    name = name.replace("num__", "").replace("cat__", "")
    # Remove trailing _0.0 / _1.0 / _missing OHE suffixes
    import re
    name = re.sub(r'_[\d\.]+$', '', name)
    name = re.sub(r'_missing$', '', name)
    return name[:maxlen] if len(name) > maxlen else name


# ──────────────────────────────────────────────────────────────────────────────
# Model loader
# ──────────────────────────────────────────────────────────────────────────────

def _load_model(model_name: str, input_dim: int, num_classes: int,
                exp_dir) -> torch.nn.Module:
    """Load a trained model from best_model.pt (or cnn_gru_final.pt)."""
    if model_name == "1D-CNN":
        model = CNN1D(input_dim, num_classes)
    elif model_name == "GRU":
        model = GRUBaseline(input_dim, num_classes)
    elif model_name == "CNN-GRU":
        model = CNN_GRU(input_dim, num_classes)
    else:
        raise ValueError(f"Unknown model name: {model_name!r}")

    # Try experiment dir first, then global models dir
    candidates = [
        exp_dir / "best_model.pt",
        cfg.MODELS_DIR / "cnn_gru_final.pt",
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        raise FileNotFoundError(
            f"No trained model found for {model_name}. "
            "Run the corresponding training script first."
        )
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    logger.info(f"Loaded model from {path}")
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Plot helpers
# ──────────────────────────────────────────────────────────────────────────────

_DARK_BG = "#080d1a"
_TEXT    = "#94a3b8"
_CYAN    = "#00d4ff"


def _apply_dark_theme():
    plt.rcParams.update({
        "figure.facecolor":  _DARK_BG,
        "axes.facecolor":    _DARK_BG,
        "axes.edgecolor":    "#1e2d45",
        "axes.labelcolor":   _TEXT,
        "xtick.color":       _TEXT,
        "ytick.color":       _TEXT,
        "text.color":        _TEXT,
        "font.family":       "DejaVu Sans",
        "grid.color":        "#0e1c2e",
        "grid.linewidth":    0.6,
    })


def _save_and_close(path, dpi: int = 150):
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight",
                facecolor=_DARK_BG, edgecolor="none")
    plt.close()
    logger.info(f"  Saved → {path.name}")


# ──────────────────────────────────────────────────────────────────────────────
# Core SHAP analysis
# ──────────────────────────────────────────────────────────────────────────────

def run_shap_analysis(
    experiment_id: str,
    model_name: str,
    task_type: str = "binary",
    n_background: int = 150,
    n_explain: int = 300,
) -> pd.DataFrame:
    """
    Run SHAP GradientExplainer on a trained model and persist artifacts.

    Parameters
    ----------
    experiment_id : str
        e.g. "E05_cnn_gru_binary"
    model_name : str
        One of "1D-CNN", "GRU", "CNN-GRU"
    task_type : str
        "binary" or "multiclass"
    n_background : int
        Number of training samples used as SHAP background reference.
    n_explain : int
        Number of test samples to explain.

    Returns
    -------
    pd.DataFrame
        Top SHAP feature importances (feature, mean_abs_shap).
    """
    logger.info(f"═══ SHAP Analysis: {experiment_id} | {model_name} | {task_type} ═══")

    exp_dir = cfg.EXPERIMENTS_DIR / experiment_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────────────
    logger.info("Loading processed parquet datasets…")
    X_train = pd.read_parquet(cfg.PROCESSED_DATA_DIR / "X_train.parquet").values.astype(np.float32)
    X_test  = pd.read_parquet(cfg.PROCESSED_DATA_DIR / "X_test.parquet").values.astype(np.float32)
    y_test  = pd.read_parquet(cfg.PROCESSED_DATA_DIR / "y_test.parquet").values.squeeze()

    input_dim = X_train.shape[1]
    num_classes = 1 if task_type == "binary" else int(len(np.unique(y_test)))

    raw_names    = _load_feature_names(input_dim)
    display_names = [_clean_name(n) for n in raw_names]

    # ── Load model ─────────────────────────────────────────────────────────────
    model = _load_model(model_name, input_dim, num_classes, exp_dir)

    # ── Sample background & explanation sets ───────────────────────────────────
    rng = np.random.default_rng(cfg.GLOBAL_SEED)
    bg_idx = rng.choice(len(X_train), size=min(n_background, len(X_train)), replace=False)
    ex_idx = rng.choice(len(X_test),  size=min(n_explain,    len(X_test)),  replace=False)

    X_bg = torch.tensor(X_train[bg_idx])
    X_ex = torch.tensor(X_test[ex_idx])

    # ── SHAP GradientExplainer ────────────────────────────────────────────────
    logger.info(f"Computing SHAP values (background={len(bg_idx)}, explain={len(ex_idx)})…")
    explainer   = shap.GradientExplainer(model, X_bg)
    shap_values = explainer.shap_values(X_ex)   # (N, D) for binary; list for multiclass

    # Normalise to a single (N, D) importance array
    if isinstance(shap_values, list):
        # Multiclass: average absolute values across all classes
        sv = np.mean(np.abs(np.stack(shap_values, axis=0)), axis=0)
    else:
        sv = np.asarray(shap_values)  # (N, D) for most cases

    # Newer shap versions return an explicit trailing output-dimension axis
    # even for a single-logit (sigmoid/BCE) binary model, i.e. shape (N, D, 1)
    # instead of (N, D). Squeeze that away so downstream plotting (which
    # expects a plain 2D matrix) and the per-sample waterfall slice (which
    # must be a 1D vector) both work correctly.
    if sv.ndim == 3:
        sv = sv[..., 0] if sv.shape[-1] == 1 else np.mean(np.abs(sv), axis=-1)

    X_ex_np = X_ex.numpy()

    # ── Plots ──────────────────────────────────────────────────────────────────
    _apply_dark_theme()

    # 1. Beeswarm summary plot
    logger.info("Generating SHAP beeswarm summary…")
    shap.summary_plot(
        sv, X_ex_np,
        feature_names=display_names,
        max_display=20,
        show=False,
        plot_type="dot",
        color_bar=True,
    )
    _save_and_close(exp_dir / "shap_summary.png")

    # 2. Global bar chart (mean |SHAP|)
    logger.info("Generating SHAP bar chart…")
    shap.summary_plot(
        sv, X_ex_np,
        feature_names=display_names,
        max_display=20,
        show=False,
        plot_type="bar",
    )
    _save_and_close(exp_dir / "shap_bar.png")

    # 3. Waterfall for single highest-confidence prediction
    logger.info("Generating SHAP waterfall (single sample)…")
    sample_idx = int(np.abs(sv).sum(axis=1).argmax())
    exp_obj = shap.Explanation(
        values=sv[sample_idx],
        base_values=0.0,
        data=X_ex_np[sample_idx],
        feature_names=display_names,
    )
    shap.waterfall_plot(exp_obj, max_display=15, show=False)
    _save_and_close(exp_dir / "shap_waterfall.png")

    # ── Persist importance CSV ─────────────────────────────────────────────────
    mean_abs = np.abs(sv).mean(axis=0)
    shap_df = (
        pd.DataFrame({"feature": display_names, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    shap_df.to_csv(exp_dir / "shap_importance.csv", index=False)
    logger.info(f"Top SHAP feature: {shap_df.iloc[0]['feature']} "
                f"({shap_df.iloc[0]['mean_abs_shap']:.5f})")

    logger.info(f"═══ SHAP analysis complete for {experiment_id} ═══\n")
    return shap_df
