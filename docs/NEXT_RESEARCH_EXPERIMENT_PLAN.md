# NEXT RESEARCH EXPERIMENT PLAN

## 1. Current Implementation Status

**What is verified (actually implemented and working):**
* **Dataset & Splitting:** Loading chunks via PyArrow, stratified random split (70/15/15) before One-Hot Encoding and Scaling.
* **Leakage-Aware Preprocessing:** High-cardinality/payload string columns (>100 unique values) are dropped dynamically. `Attack_label` and `Attack_type` are appropriately isolated. Scaler/Encoder are fitted strictly on `X_train`.
* **Baselines (E01, E02):** Random Forest (Binary & Multiclass) are fully trained and evaluated.
* **DL Models (E03-E08):** 1D-CNN, GRU, and CNN-GRU models exist and train correctly for both binary and multiclass tasks.
* **Explainability:** SHAP GradientExplainer runs successfully on the three binary models. MQTT protocol features consistently emerge as the top influencers.
* **Experiment Framework:** Basic registry in `results/experiment_registry.csv` captures training time, inference latency, parameters, model size, and baseline metrics. Dashboard serves live monitoring.

**What is only documented / planned (not yet implemented):**
* **Chronological Split:** The current data split is random stratified. Chronological or device-group splitting is purely theoretical right now.
* **Temporal Sequences for GRU:** The current GRU model explicitly uses sequence length = 1. Genuine temporal sequence construction (grouping by device/time) is not implemented.
* **Data Augmentation / Imbalance Handling:** SMOTE, Class Weighting, Focal Loss are documented in the deep research report as ideas but not present in the codebase.
* **Hyperparameter Tuning:** Current configurations are hardcoded baselines. No automated search framework (like Optuna) exists yet.
* **Model Compression:** INT8 Quantization, Pruning, Knowledge Distillation are documented but not yet coded.
* **Adversarial / Robustness Checks:** Not implemented.

## 2. Potential Methodological Issues

* **Suspicious Binary Accuracy (100%):** A 100% accuracy and 1.000 Macro-F1 score in the binary setting across multiple models strongly suggests a remaining leakage, a trivial shortcut, or strong reliance on a protocol-specific feature (e.g., `mqtt.topic`). The model might not be detecting "attacks," but rather the presence of "MQTT" traffic which happens to be perfectly correlated with attacks.
* **Temporal Modeling Validity:** A GRU processing `seq_len=1` is essentially a dense network with recurrent weights. Presenting it as a sequential model is misleading unless sequences are actually constructed from grouped flows.
* **Multiclass Imbalance Dropoff:** The difference between ~95% accuracy and ~0.64 Macro-F1 highlights that the majority class dominates training, and rare attack types are being ignored or misclassified. The current cross-entropy loss doesn't mitigate this.
* **Experiment Tracking Weakness:** While `experiment_registry.csv` exists, we need a robust framework (saving `config.json`, `confusion_matrix.png`, etc.) per run without overwriting previous runs' artifacts.

## 3. Recommended Experiment Order (Roadmap)

### Phase 2: Scientific Validation & Framework Upgrade (P0 - Must Do)
* **E01_validation_leakage_audit:** Deep dive into the 100% binary accuracy. Analyze label distribution across protocols, check for duplicated samples across splits, and verify whether preprocessing accidentally left a direct proxy for the label.
* **Experiment Framework Enhancement:** Improve the codebase to automatically output `config.json`, `metrics.json`, `classification_report.csv`, `confusion_matrix.png`, `model_summary.txt`, etc., into isolated `results/<exp_id>/` directories.
* **E05_mqtt_ablation:** Ablation study training without MQTT-related features to scientifically evaluate if the 100% accuracy drops, revealing the model's dependency.

### Phase 3: Enhancing Generalization & Explainability (P1 - High Value)
* **E03_class_imbalance:** Compare standard loss, class-weighted loss, and focal loss for CNN-GRU on multiclass tasks to address the ~0.64 Macro-F1.
* **E04_shap_feature_ablation:** Train models using only the top 10/20 SHAP features versus dropping the top SHAP features to validate if SHAP explanations accurately reflect causal model dependency.
* **E07_generalization_validation:** Attempt chronological splitting (if `frame.time` is available and logically orderable) to evaluate real-world drift performance compared to the optimistic random split.
* **E06_temporal_sequence_modeling:** Only if a valid grouping key (Device ID, Session) exists in the raw data, construct sequences for GRU comparison. If not, explicitly document the limitation.

### Phase 4: Optimization & Edge Feasibility (P2 - Optional/Tuning)
* **E02_cnn_gru_hyperparameter_tuning:** Optimize CNN-GRU architecture using an Optuna-like framework on the validation set, focusing entirely on Macro-F1.
* **Architecture Experiments:** Test Depthwise-separable CNN or Residual Blocks variants if tuning proves insufficient.
* **E08_model_compression:** Apply INT8 quantization and moderate pruning on the final tuned CNN-GRU.
* **E09_edge_efficiency:** Comprehensive benchmarking on CPU (Latency, Throughput, Memory, Disk Size).

## 4. Expected Outputs, Exact Files & Compute

| Experiment | Files Created/Modified | Expected Outputs | Compute Need |
| ---------- | ---------------------- | ---------------- | ------------ |
| **E01 (Leakage Audit)** | `notebooks/E01_leakage_audit.ipynb`, `results/E01_validation_leakage_audit/report.md` | Report detailing correlation between protocols, splits, and labels. | Low (CPU) |
| **Framework Update** | `src/utils.py`, `src/training.py`, `configs/` | Improved experiment structure `results/<EXP_ID>/` with JSON/CSV/PNG metrics. | None |
| **E05 (MQTT Ablation)** | `run_E05_mqtt_ablation.py` | Metric comparison with/without MQTT features. | Med (GPU) |
| **E03 (Imbalance)** | `src/losses.py`, `run_E03_class_imbalance.py` | Macro-F1 improvements on rare classes. | Med (GPU) |
| **E04 (SHAP Ablation)** | `run_E04_shap_ablation.py` | Performance table: top-k vs missing top-k features. | Med (GPU) |
| **E02 (Tuning)** | `run_E02_hyperparameter_tuning.py` | Best hyperparameter JSON and tuning history plots. | High (GPU) |

## 5. Risks
* **Risk 1:** Uncovering that the Edge-IIoTset is fundamentally flawed for intrusion detection (if all attacks are perfectly correlated with MQTT fields). *Mitigation:* Focus the research narrative on "Leakage-Aware Evaluation" and robust methodology.
* **Risk 2:** Class-imbalance techniques like SMOTE could blow up RAM or take excessive time on a 2.2M dataset. *Mitigation:* Favor Focal Loss and Class Weighting.
* **Risk 3:** Hyperparameter tuning explodes compute budgets. *Mitigation:* Start with a small search space and limit max epochs.
