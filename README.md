# Explainable Hybrid Deep Learning-Based Intrusion Detection System for IoT Networks

This repository implements the research project: **Leakage-Aware and Explainable CNN-GRU Intrusion Detection for IoT Networks**.

## Project Status

### Phase 1: Data Acquisition & Audit (✅ Complete)
- **Dataset Acquired:** Downloaded Kaggle Edge-IIoTset (`DNN-EdgeIIoT-dataset.csv`) (Hash: `1d3ef6c7cc22784a528a117f1158f5cad750273679437e18a39c360ee6b79fd7`).
- **Audit & Leakage Policy:** Analyzed schema. Dropped raw IPs, timestamps, and metadata to prevent train-test shortcuts and enforce a strict **Operational Feature Policy**.
- **Preprocessing & Memory Handling:** Implemented chunk-based and PyArrow fast-loading. Handled 4+ TB array memory explosion by eliminating >100 high-cardinality payload fields dynamically before One-Hot Encoding.
- **Artifacts:** `X_train.parquet`, `X_val.parquet`, `X_test.parquet` splits saved efficiently to disk.

### Phase 2: Baseline Machine Learning (✅ Complete)
- **E01 - Random Forest Binary:** Evaluated `Normal` vs `Attack`. Resulted in trivial 1.0 F1 score.
- **E02 - Random Forest Multiclass:** Evaluated 15 attack classes. Resulted in 0.9825 accuracy, but highlighted struggle with minority application-layer attacks (like XSS and Fingerprinting) due to payload truncation.
- **Experiment Tracking:** Auto-logged configurations, inference latency, and parameter sizes to `results/experiment_registry.csv`.

### Phase 3: Deep Learning & Dashboard (⏳ In Progress)
- **Framework:** Transitioned repository to `PyTorch`.
- **Architectural Refactoring:** Split monolithic scripts into `src/models.py`, `src/training.py`, `src/config.py`. 
- **1D-CNN (E03):** Currently training (baseline spatial).
- **GRU (E04):** Currently training (baseline temporal). *Note: treated strictly as exploratory (sequence length=1) to prevent faking chronological flows since timestamps/IPs were strictly removed as leakage.*
- **CNN-GRU Hybrid (E05):** Currently training.

---

## Setup Instructions

### 1. Python Virtual Environment
**Windows**:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux/macOS**:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
# Additional PyTorch reqs
pip install torch torchvision
```

### 3. Collaborator Setup (Dataset Download)
Since the `1.6GB` dataset and large `.parquet` intermediate files are not pushed to GitHub, collaborators must download the original dataset locally.
We have written an automation script that handles this safely:
```bash
python download_dataset.py
```
This script uses `kagglehub` to download the exact dataset version and places `DNN-EdgeIIoT-dataset.csv` into `data/raw/` automatically.

### 4. Resuming the Pipeline
Since the raw dataset is downloaded, you must reconstruct the intermediate splits (`X_train.parquet`, etc.) before running neural networks:
```bash
# Regenerates data/processed/ parquet files
python -m src.preprocessing
```
You can then run the Phase 3 deep learning scripts. All the previous experiment metrics (CSVs, JSONs) are already stored in `results/experiments/` and `results/experiment_registry.csv`!

## Reproducibility
- Every metric reported must come from an actual execution on the full dataset (no arbitrary downsampling).
- Global seed (`42`) controls random splits and layer initialization for replicability.
- See `results/experiment_registry.csv` for tracked metric outputs across all tested models.
