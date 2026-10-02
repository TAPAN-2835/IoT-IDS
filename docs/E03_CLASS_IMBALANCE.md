> **Correction (2026-10-02):** These runs used EPOCHS = 5, so the comparison is between undertrained models; the 15-epoch cross-entropy CNN-GRU (E08) already reached Macro-F1 0.644. The focal-loss conclusion is not supported; see the equal-epoch redo (C04-C06). See [LEAKAGE_FIX_AND_CLEAN_BASELINES.md](LEAKAGE_FIX_AND_CLEAN_BASELINES.md).

# E03 - CLASS IMBALANCE EXPERIMENTS

## 1. Objective
The multiclass CNN-GRU achieves ~95% accuracy but a significantly lower Macro-F1 (~0.64), indicating that minority attack classes are poorly detected. This experiment evaluates loss functions designed to handle class imbalance.

## 2. Methodology
- **Target:** Multiclass (`Attack_type`).
- **Feature Set:** `operational` policy.
- **Model:** CNN-GRU.
- **Conditions:**
  1. Standard Cross-Entropy Loss (Baseline).
  2. Class-Weighted Cross-Entropy Loss.
  3. Focal Loss ($\gamma = 2.0$).

## 3. Results

| Loss Function | Accuracy | Macro-F1 | Weighted-F1 |
| ------------- | -------- | -------- | ----------- |
| Cross-Entropy | 0.8802 | 0.2634 | 0.8563 |
| Class-Weighted | 0.8411 | 0.3299 | 0.8120 |
| Focal Loss | 0.9163 | 0.4084 | 0.8962 |

## 4. Conclusion
The dataset suffers from extreme class imbalance in the `Attack_type` target. Standard cross-entropy completely ignores minority attacks (Macro-F1 0.263). Applying class weights slightly improves minority detection but hurts overall accuracy. **Focal Loss** is highly effective, yielding a massive improvement in Macro-F1 (0.408) and boosting overall accuracy to 91.6%. Focal Loss should be adopted for all future multiclass experiments.
