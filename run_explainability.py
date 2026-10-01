"""
run_explainability.py

Phase 4: SHAP Explainability for the trained CNN-GRU Hybrid (E05).
Also runs SHAP on 1D-CNN (E03) and GRU (E04) for ablation comparison.

Run AFTER phase 3 training has completed:
    python run_phase3_binary.py
    python run_explainability.py
"""
import logging
from src.utils import setup_logger
from src.explainability import run_shap_analysis

logger = setup_logger("phase4_explainability")


def main():
    logger.info("=" * 60)
    logger.info("Phase 4: SHAP Explainability Analysis")
    logger.info("=" * 60)

    experiments = [
        ("E05_cnn_gru_no_mqtt", "CNN-GRU", "binary"),   # Primary: proposed model (ablated)
        ("E03_cnn1d_binary",   "1D-CNN",  "binary"),   # Ablation: CNN only
        ("E04_gru_binary",     "GRU",     "binary"),   # Ablation: GRU only
    ]

    for exp_id, model_name, task in experiments:
        try:
            logger.info(f"\nRunning SHAP for {exp_id}...")
            df = run_shap_analysis(
                experiment_id=exp_id,
                model_name=model_name,
                task_type=task,
                n_background=150,
                n_explain=300,
            )
            logger.info(f"Top 5 SHAP features for {exp_id}:")
            for _, row in df.head(5).iterrows():
                logger.info(f"  {row['feature']:30s}  mean|SHAP| = {row['mean_abs_shap']:.5f}")
        except FileNotFoundError as e:
            logger.warning(f"Skipping {exp_id} — model not found: {e}")
        except Exception as e:
            logger.error(f"Error in {exp_id}: {e}", exc_info=True)

    logger.info("\nPhase 4 explainability complete.")
    logger.info("Artifacts saved to results/experiments/<exp_id>/:")
    logger.info("  shap_summary.png   (beeswarm global attribution)")
    logger.info("  shap_bar.png       (mean |SHAP| bar chart)")
    logger.info("  shap_waterfall.png (single sample waterfall)")
    logger.info("  shap_importance.csv (ranked feature importances)")


if __name__ == "__main__":
    main()
