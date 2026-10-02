"""
run_clean_baselines.py

Re-runs the core baselines after fixing the "0" vs "0.0" empty-field artefact
(cfg.CANONICALIZE_NUMERIC_TOKENS = True). New experiment IDs use the "C" prefix
so they never collide with the earlier E01-E08 / Phase-2 runs.

  binary stage      C01_xgb_binary_clean          XGBoost (GPU)
                    C02_cnn_gru_binary_clean      CNN-GRU (GPU), 15 epochs
  multiclass stage  C03_xgb_multiclass_clean      XGBoost (GPU)
                    C04_cnn_gru_mc_ce_clean       CNN-GRU, cross-entropy
                    C05_cnn_gru_mc_weighted_clean CNN-GRU, class-weighted CE
                    C06_cnn_gru_mc_focal_clean    CNN-GRU, focal loss
                    (C04-C06 = the E03 loss comparison redone at equal epochs)

Laptop-friendly: heavy work runs on the GPU, the process lowers its own
priority, CPU threads are capped, and preprocessing streams the CSV.

Usage:
    python run_clean_baselines.py                    # binary then multiclass
    python run_clean_baselines.py --stage binary
    python run_clean_baselines.py --stage multiclass --skip-existing
"""
import os

# Use the system allocator for Arrow (must be set before pyarrow is imported):
# its default mimalloc pool keeps freed parquet/CSV buffers instead of returning
# them to the OS, which left GBs of RAM held during GPU training.
os.environ.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")

import argparse
import multiprocessing as mp

import psutil
import torch

import src.config as cfg
from src.train import train_xgb_gpu
from src.training import train_dl_model
from src.utils import setup_logger

logger = setup_logger("run_clean_baselines")

STAGES = {
    "binary": ("Attack_label", "binary", [
        ("C01_xgb_binary_clean", "xgb", None),
        ("C02_cnn_gru_binary_clean", "CNN-GRU", "ce"),
    ]),
    "multiclass": ("Attack_type", "multiclass", [
        ("C03_xgb_multiclass_clean", "xgb", None),
        ("C04_cnn_gru_mc_ce_clean", "CNN-GRU", "ce"),
        ("C05_cnn_gru_mc_weighted_clean", "CNN-GRU", "class_weighted"),
        ("C06_cnn_gru_mc_focal_clean", "CNN-GRU", "focal"),
    ]),
}


def keep_awake():
    """Stop Windows from sleeping mid-run (the screen may still turn off).

    Applies only while this process runs; normal sleep settings resume when it exits.
    Without it, the laptop's idle-sleep timer froze a training run halfway through.
    """
    if os.name != "nt":
        return
    import ctypes
    ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)


def be_nice():
    """Lower this process's priority and CPU thread use so the laptop stays usable."""
    try:
        proc = psutil.Process()
        proc.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if hasattr(psutil, "BELOW_NORMAL_PRIORITY_CLASS") else 10)
    except Exception as e:
        logger.warning(f"Could not lower process priority: {e}")
    torch.set_num_threads(2)
    keep_awake()
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.warning("CUDA not available: deep-learning runs will fall back to the CPU (slow).")


def run_stage(name, skip_existing):
    target_col, task_type, experiments = STAGES[name]
    logger.info(f"=== Stage: {name} ({target_col}) | policy={cfg.FEATURE_POLICY} | "
                f"canonicalize={cfg.CANONICALIZE_NUMERIC_TOKENS} ===")
    todo = [e for e in experiments
            if not (skip_existing and (cfg.EXPERIMENTS_DIR / e[0] / "experiment_record.json").exists())]
    if not todo:
        logger.info("All experiments in this stage already exist; skipping.")
        return

    if processed_data_matches(target_col):
        logger.info("data/processed/ already holds this target/policy; reusing it.")
    else:
        preprocess_in_child(target_col)

    for exp_id, model, loss in todo:
        if model == "xgb":
            train_xgb_gpu(exp_id, target_col, task_type)
        else:
            train_dl_model(exp_id, model, target_col, task_type, loss_type=loss, epochs=cfg.EPOCHS)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    log_peak_memory(name)


def _preprocess_job(target_col, feature_policy, canonicalize):
    """Runs in a child process: all of its RAM is returned to the OS when it exits."""
    be_nice()
    cfg.FEATURE_POLICY = feature_policy
    cfg.CANONICALIZE_NUMERIC_TOKENS = canonicalize
    from src.preprocessing import run_preprocessing_pipeline
    from audit_shortcuts import processed_shortcut_scan
    if not run_preprocessing_pipeline(target_col=target_col):
        raise SystemExit(1)
    processed_shortcut_scan()


def processed_data_matches(target_col):
    """True when data/processed/ was already built with the settings this stage needs."""
    from src.preprocessing import load_processed_metadata
    try:
        meta = load_processed_metadata()
    except FileNotFoundError:
        return False
    return (meta["target_col"] == target_col and meta["feature_policy"] == cfg.FEATURE_POLICY
            and meta["canonicalize_numeric_tokens"] == cfg.CANONICALIZE_NUMERIC_TOKENS)


def preprocess_in_child(target_col):
    proc = mp.get_context("spawn").Process(
        target=_preprocess_job, args=(target_col, cfg.FEATURE_POLICY, cfg.CANONICALIZE_NUMERIC_TOKENS))
    proc.start()
    proc.join()
    if proc.exitcode != 0:
        raise RuntimeError(f"Preprocessing for {target_col} failed (exit code {proc.exitcode}); see pipeline.log")


def log_peak_memory(label):
    mem = psutil.Process().memory_info()
    peak = getattr(mem, "peak_wset", None)  # Windows only
    msg = f"[{label}] RAM now {mem.rss / 1e9:.2f} GB"
    if peak:
        msg += f", peak {peak / 1e9:.2f} GB"
    if torch.cuda.is_available():
        msg += f" | GPU peak {torch.cuda.max_memory_allocated() / 1e9:.2f} GB"
    logger.info(msg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["binary", "multiclass", "all"], default="all")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    be_nice()
    cfg.FEATURE_POLICY = "operational"
    cfg.CANONICALIZE_NUMERIC_TOKENS = True
    stages = ["binary", "multiclass"] if args.stage == "all" else [args.stage]
    for stage in stages:
        run_stage(stage, args.skip_existing)
    logger.info("Clean baselines complete. See results/experiment_registry.csv (C0* rows).")


if __name__ == "__main__":
    main()
