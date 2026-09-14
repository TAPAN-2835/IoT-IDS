"""
run_remaining_pipeline.py

Completes everything left in the IoT-IDS pipeline in one sequential run:
  1. E05 CNN-GRU (binary)              -- was interrupted mid-training previously
  2. Multiclass preprocessing (Attack_type target)
  3. E06 1D-CNN, E07 GRU, E08 CNN-GRU  (multiclass)
  4. Phase 4 SHAP explainability for E03/E04/E05 (binary models)

E01-E04 are already complete (see results/experiment_registry.csv) and are
deliberately NOT re-run here to avoid wasting ~10 minutes of GPU time re-training
things that already finished successfully.

Progress is visible live at http://localhost:8000/pipeline while this runs
(make sure `python pipeline_monitor.py` and the dashboard server are running).
All log output is appended to pipeline.log via src.utils.setup_logger.
"""
import sys
import traceback

from src.utils import setup_logger
import src.config as cfg
from src.training import train_dl_model
from src.preprocessing import run_preprocessing_pipeline
from src.explainability import run_shap_analysis

logger = setup_logger("run_remaining_pipeline")


def main():
    logger.info("#" * 70)
    logger.info("RESUMING PIPELINE: completing everything after E04")
    logger.info("#" * 70)

    # ── Step 1: E05 CNN-GRU Hybrid (Binary) ────────────────────────────────
    logger.info("=" * 60)
    logger.info("Step 1/4: E05 CNN-GRU Hybrid (Binary) — resuming interrupted run")
    logger.info("=" * 60)
    train_dl_model("E05_cnn_gru_binary", "CNN-GRU", "Attack_label", "binary")

    # ── Step 2: Multiclass preprocessing ───────────────────────────────────
    logger.info("=" * 60)
    logger.info("Step 2/4: Re-preprocessing dataset for Attack_type (multiclass) target")
    logger.info("=" * 60)
    cfg.DEFAULT_TARGET_COL = "Attack_type"
    run_preprocessing_pipeline()

    # ── Step 3: Multiclass DL models ───────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Step 3/4: E06 1D-CNN Multiclass")
    logger.info("=" * 60)
    train_dl_model("E06_cnn1d_multiclass", "1D-CNN", "Attack_type", "multiclass")

    logger.info("=" * 60)
    logger.info("Step 3/4: E07 GRU Multiclass")
    logger.info("=" * 60)
    train_dl_model("E07_gru_multiclass", "GRU", "Attack_type", "multiclass")

    logger.info("=" * 60)
    logger.info("Step 3/4: E08 CNN-GRU Hybrid Multiclass")
    logger.info("=" * 60)
    train_dl_model("E08_cnn_gru_multiclass", "CNN-GRU", "Attack_type", "multiclass")

    # ── Step 4: SHAP explainability on binary models ───────────────────────
    # NOTE: explainability reads whatever is currently in data/processed/, which
    # at this point holds the MULTICLASS-encoded y (from Step 2). That's fine —
    # for task_type="binary" the SHAP code only uses X_test (feature values,
    # target-independent) and hardcodes num_classes=1, so y's encoding doesn't
    # affect these three explanations.
    logger.info("=" * 60)
    logger.info("Step 4/4: Phase 4 SHAP Explainability (E05, E03, E04)")
    logger.info("=" * 60)
    shap_targets = [
        ("E05_cnn_gru_binary", "CNN-GRU", "binary"),
        ("E03_cnn1d_binary",   "1D-CNN",  "binary"),
        ("E04_gru_binary",     "GRU",     "binary"),
    ]
    for exp_id, model_name, task in shap_targets:
        try:
            logger.info(f"Running SHAP for {exp_id}...")
            df = run_shap_analysis(exp_id, model_name, task, n_background=150, n_explain=300)
            logger.info(f"Top 5 SHAP features for {exp_id}:")
            for _, row in df.head(5).iterrows():
                logger.info(f"  {row['feature']:30s}  mean|SHAP| = {row['mean_abs_shap']:.5f}")
        except Exception as e:
            logger.error(f"SHAP failed for {exp_id}: {e}", exc_info=True)

    logger.info("#" * 70)
    logger.info("PIPELINE COMPLETE: all experiments (E01-E08) + SHAP explainability done.")
    logger.info("#" * 70)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.error("run_remaining_pipeline.py CRASHED:\n" + traceback.format_exc())
        sys.exit(1)
