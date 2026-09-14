"""
run_shap_fix.py

Fix-up for Phase 4 SHAP explainability.

What went wrong: run_remaining_pipeline.py ran Phase 4 SHAP right after the
multiclass (E06-E08) training, at which point data/processed/*.parquet held
data preprocessed for the Attack_type (multiclass) target -- a DIFFERENT
stratified train/val/test split than the one E03/E04/E05 were originally
trained on. Because src/preprocessing.py refits a fresh OneHotEncoder and
StandardScaler on whatever split is active, that produced 91 post-encoding
feature columns instead of the 90 the binary models were trained with:
  - E04 (GRU) failed outright: nn.GRU's weight_ih_l0 is shape-locked to
    input_dim, so loading the checkpoint raised a hard size-mismatch error.
  - E03 (1D-CNN) and E05 (CNN-GRU) did NOT crash -- 90 // 2 == 91 // 2 == 45,
    so their Conv1d/Linear layers happened to tolerate the off-by-one -- but
    their already-saved shap_summary.png / shap_bar.png were computed against
    the wrong scaler and wrong one-hot encoding. Those are invalid, not just
    E04's missing artifacts.

Fix: regenerate BINARY-target preprocessing (deterministic given
GLOBAL_SEED=42, so it reconstructs the exact same 90-feature schema E03/E04/E05
were trained on), then rerun SHAP for all three from scratch.
"""
from src.utils import setup_logger
import src.config as cfg
from src.preprocessing import run_preprocessing_pipeline
from src.explainability import run_shap_analysis

logger = setup_logger("run_shap_fix")


def main():
    logger.info("=" * 60)
    logger.info("SHAP FIX-UP: restoring binary (Attack_label) preprocessing")
    logger.info("so E03/E04/E05 see the exact data schema they were trained on")
    logger.info("=" * 60)
    cfg.DEFAULT_TARGET_COL = "Attack_label"
    run_preprocessing_pipeline()

    targets = [
        ("E05_cnn_gru_binary", "CNN-GRU", "binary"),
        ("E03_cnn1d_binary",   "1D-CNN",  "binary"),
        ("E04_gru_binary",     "GRU",     "binary"),
    ]
    failures = []
    for exp_id, model_name, task in targets:
        try:
            logger.info(f"Running SHAP for {exp_id} (redo, binary-consistent data)...")
            df = run_shap_analysis(exp_id, model_name, task, n_background=150, n_explain=300)
            logger.info(f"Top 5 SHAP features for {exp_id}:")
            for _, row in df.head(5).iterrows():
                logger.info(f"  {row['feature']:30s}  mean|SHAP| = {row['mean_abs_shap']:.5f}")
        except Exception as e:
            failures.append(exp_id)
            logger.error(f"SHAP failed for {exp_id}: {e}", exc_info=True)

    if failures:
        logger.error(f"SHAP FIX-UP FINISHED WITH FAILURES: {failures}")
    else:
        logger.info("SHAP FIX-UP COMPLETE: E05, E03, E04 all regenerated successfully.")


if __name__ == "__main__":
    main()
