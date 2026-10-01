# PHASE 2 IMPLEMENTATION AUDIT

This document maps the current implementation state of the project based on the existing source code (`src/`), configuration (`config.py`), and artifacts in the `results/` directory. 

## 1. What is Currently Implemented
* **Data Loading:** Efficient loading using `pandas` with the `pyarrow` engine, avoiding OOM on the ~1.6GB CSV.
* **Leakage-Aware Preprocessing:** A policy-driven feature selection (`audit/operational_feature_policy.csv`) drops basic identifiers (IPs, ports, timestamps). A dynamic cardinality filter drops text columns with >100 unique values.
* **Train/Test Splitting:** Stratified random splitting (70% Train, 15% Validation, 15% Test).
* **Models:** Random Forest, 1D-CNN, GRU, and CNN-GRU models are fully implemented for both binary (`Attack_label`) and multiclass (`Attack_type`) objectives.
* **Evaluation:** Computation of accuracy, precision, recall, F1 (macro & weighted), confusion matrices, inference latency, and model size.
* **Explainability (SHAP):** `GradientExplainer` is implemented and generates summary, bar, and waterfall plots for the binary deep learning models.
* **Logging/Monitoring:** `experiment_registry.csv` logs core metrics. A FastAPI dashboard tails `pipeline.log` and `training_progress.json` for live monitoring.

## 2. What is Only Documented (Not Implemented)
* **Chronological/Device Splitting:** The data is only split randomly. GroupKFold or chronological splitting does not exist in `preprocessing.py`.
* **True Temporal Sequences:** The GRU models explicitly reshape the 1D feature vector into `(batch, 1, features)`—there is no sliding window or session grouping implemented.
* **Imbalance Handling:** No class-weighting, Focal Loss, or SMOTE is applied during training (multiclass uses standard `CrossEntropyLoss`).
* **Hyperparameter Tuning:** Models use hardcoded layer sizes, dropout rates, and learning rates. No tuning framework (e.g., Optuna) is present.
* **Model Compression/Quantization:** Not present in the codebase.
* **Robustness/Adversarial Tests:** Not implemented.

## 3. Current Data Flow
1. Load `DNN-EdgeIIoT-dataset.csv` via PyArrow.
2. `clean_data()` filters features against `operational_feature_policy.csv`.
3. High-cardinality string columns (>100 unique values) are dynamically dropped to prevent One-Hot Encoder memory explosion.
4. Dataset is randomly stratified and split (70/15/15).
5. Scikit-learn `ColumnTransformer` (Imputer + StandardScaler for numerical; Imputer + OneHotEncoder for categorical) is fitted **strictly on the training set**.
6. Transformed data is saved to `data/processed/X_train.parquet`, etc.
7. Model trains using PyTorch `TensorDataset` and `DataLoader`.

## 4. Current Feature List
* **Raw Dataset:** 63 columns.
* **Operational Policy (Safe list):** 49 columns.
* **Dropped by Policy (Leakage):** `frame.time`, `ip.src_host`, `ip.dst_host`, `arp.dst.proto_ipv4`, `arp.src.proto_ipv4`, `icmp.transmit_timestamp`, `tcp.dstport`, `tcp.srcport`, `udp.port`, `udp.time_delta`, `mbtcp.trans_id`, `mbtcp.unit_id` etc.
* **Dynamic Dropping:** Columns like `tcp.payload`, `tcp.options`, `mqtt.msg` are marked safe in the policy but are likely dropped at runtime by `clean_data()` if they exceed 100 unique values in the dataset chunk.

## 5. Current Split Strategy
* **Method:** `sklearn.model_selection.train_test_split`
* **Type:** Random Stratified Split.
* **Ratio:** 70% Train, 15% Validation, 15% Test.

## 6. Current Model Architectures
* **1D-CNN:** `Conv1d(32)` -> `BatchNorm` -> `ReLU` -> `MaxPool1d(2)` -> `Dropout(0.2)` -> `Dense(32)` -> `Output`
* **GRU:** `GRU(hidden=64, seq_len=1)` -> `Dropout(0.2)` -> `Dense(32)` -> `Output`
* **CNN-GRU:** `Conv1d(32)` -> `BatchNorm` -> `ReLU` -> `MaxPool1d(2)` -> `Dropout(0.2)` -> Transpose -> `GRU(hidden=64)` -> `Dense(32)` -> `Dropout(0.2)` -> `Output`

## 7. Current Evaluation
Metrics logged in `experiment_registry.csv`:
* Accuracy, Precision, Recall, Macro-F1, Weighted-F1
* False Positive Rate (FPR), False Negative Rate (FNR)
* Training Time (seconds), Inference Latency (ms/sample)
* Parameter Count, Model Size (MB)

## 8. Current SHAP Implementation
* **Method:** `shap.GradientExplainer`
* **Scope:** Applied only to binary PyTorch models (E03, E04, E05).
* **Sample Size:** 150 background samples (from training data), 300 test samples for explanation.
* **Outputs:** Beeswarm summary (`shap_summary.png`), global bar chart (`shap_bar.png`), and single-sample waterfall (`shap_waterfall.png`).

## 9. Current Experiment Logging
* **State:** Simple append to `results/experiment_registry.csv`.
* **Weakness:** Does not save the actual hyperparameter configurations, preprocessing state, or specific feature lists used for a given experiment. Confusion matrices and classification reports are not permanently archived for the DL models in the codebase (except printed to logs).

## 10. Discrepancies Discovered
1. **Dynamic Feature Drop:** Documentation implies a strict operational feature list, but `clean_data()` dynamically drops high-cardinality categorical features. This means the actual feature set varies depending on the raw data contents, which is slightly non-deterministic without saving the final feature list per experiment.
2. **GRU Sequence Length:** The GRU model receives `seq_len=1` and `input_size=features` (e.g. 60+). A GRU cell expects a sequence of steps; here, it processes the entire feature vector as a single "timestep". While documented as "exploratory", presenting this as a GRU model is scientifically misleading.
3. **100% Binary Accuracy:** The binary models achieve perfect scores, which suggests a severe shortcut. Since protocol-specific features (like `mqtt.topic`) remain, the model is likely learning protocol signatures (e.g., all MQTT traffic in the dataset happens to be malicious) rather than generalizing intrusion detection.
