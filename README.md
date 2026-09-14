# Explainable Hybrid Deep Learning-Based Intrusion Detection System for IoT Networks

This repository implements the research project: **Leakage-Aware and Explainable CNN-GRU Intrusion Detection for IoT Networks**.

## Project Status — Phases 1-4 Complete ✅

All planned experiments (E01–E08) and SHAP explainability have been trained/run to completion on the full Edge-IIoTset dataset. What remains is analysis, write-up, and optional extensions — see [Next Steps](#next-steps) below.

### Phase 1: Data Acquisition & Audit (✅ Complete)
- **Dataset Acquired:** Downloaded Kaggle Edge-IIoTset (`DNN-EdgeIIoT-dataset.csv`) (Hash: `1d3ef6c7cc22784a528a117f1158f5cad750273679437e18a39c360ee6b79fd7`).
- **Audit & Leakage Policy:** Analyzed schema. Dropped raw IPs, timestamps, and metadata to prevent train-test shortcuts and enforce a strict **Operational Feature Policy**. Both `Attack_label` (binary target) and `Attack_type` (multiclass target) are excluded from the operational feature list, so training one target's model never leaks the other.
- **Preprocessing & Memory Handling:** Implemented chunk-based and PyArrow fast-loading. Handled 4+ TB array memory explosion by eliminating >100 high-cardinality payload fields dynamically before One-Hot Encoding (e.g. `tcp.payload`, `tcp.options`, `http.request.full_uri` — anything with >100 unique values is dropped).
- **Artifacts:** `X_train.parquet`, `X_val.parquet`, `X_test.parquet` splits saved efficiently to disk.

### Phase 2: Baseline Machine Learning (✅ Complete)
- **E01 — Random Forest Binary:** `Normal` vs `Attack`. F1 = 1.0000.
- **E02 — Random Forest Multiclass:** 15 attack classes. Accuracy = 98.25%, Macro F1 = 0.8813 (struggles most on minority application-layer attacks like XSS and Fingerprinting, due to payload truncation).
- **Experiment Tracking:** Auto-logged configurations, inference latency, and parameter sizes to `results/experiment_registry.csv`.

### Phase 3: Deep Learning — Binary (✅ Complete)
- **Framework:** PyTorch 2.11 (CUDA 12.8, tested on an NVIDIA RTX 4060 Laptop GPU).
- **E03 — 1D-CNN:** Accuracy = 1.0000, F1 = 1.0000. 46,337 params, 0.18 MB.
- **E04 — GRU:** Accuracy = 1.0000, F1 = 1.0000. 32,065 params, 0.13 MB. *Treated strictly as exploratory (sequence length=1) since timestamps/IPs were removed as leakage — this experiment does not claim real temporal structure.*
- **E05 — CNN-GRU Hybrid (proposed architecture):** Accuracy = 1.0000, F1 = 1.0000. 21,121 params, 0.09 MB — the smallest and highest-accuracy model of the three.
- **Run:** `python run_phase3_binary.py`

### Phase 3: Deep Learning — Multiclass (✅ Complete)
- **E06 — 1D-CNN Multiclass:** Accuracy = 95.11%, Macro F1 = 0.7036.
- **E07 — GRU Multiclass:** Accuracy = 94.99%, Macro F1 = 0.6904.
- **E08 — CNN-GRU Multiclass (proposed):** Accuracy = 94.74%, Macro F1 = 0.6445.
- **Run:** `python run_phase3_multiclass.py` (re-preprocesses the dataset targeting `Attack_type`, then trains E06–E08)

> Multiclass macro-F1 is noticeably lower than binary despite similar accuracy — this is the expected effect of severe class imbalance across the 15 attack types (a handful of rare classes drag macro-F1 down even when overall accuracy stays high). Worth digging into per-class metrics (see `classification_report.csv` per experiment) for the write-up.

### Phase 4: Explainability — SHAP (✅ Complete)
- **SHAP GradientExplainer** run on all three binary models (E03, E04, E05) — 150 background samples, 300 explained samples.
- **Artifacts per experiment**, in `results/experiments/<id>/`: `shap_summary.png` (beeswarm), `shap_bar.png` (mean |SHAP| ranking), `shap_waterfall.png` (single high-confidence sample), `shap_importance.csv` (full ranked feature list).
- **Top feature by mean |SHAP| per model:**
  | Experiment | Model | Top SHAP feature |
  |---|---|---|
  | E05 | CNN-GRU (proposed) | `mqtt.topic` |
  | E03 | 1D-CNN | `mqtt.protoname` |
  | E04 | GRU | `mqtt.conack.flags` |

  All three models converge on MQTT protocol fields as the dominant signal — worth a sentence or two in the discussion section, and a candidate ablation (see [Next Steps](#next-steps)).
- **Run:** `python run_explainability.py` (see [⚠️ Known Gotcha](#️-known-gotcha-shap-must-run-against-binary-preprocessed-data) below before running this after a multiclass run)

---

## Full Results Table

| ID | Model | Task | Accuracy | Macro F1 | Params | Model Size | Train Time |
|---|---|---|---|---|---|---|---|
| E01 | Random Forest | Binary | 100.00% | 1.0000 | — | 12.0 MB | 63s |
| E02 | Random Forest | Multiclass | 98.25% | 0.8813 | — | 1.59 GB | 56s |
| E03 | 1D-CNN | Binary | 100.00% | 1.0000 | 46,337 | 0.18 MB | 363s |
| E04 | GRU | Binary | 100.00% | 1.0000 | 32,065 | 0.13 MB | 384s |
| **E05** | **CNN-GRU (proposed)** | **Binary** | **100.00%** | **1.0000** | **21,121** | **0.09 MB** | **402s** |
| E06 | 1D-CNN | Multiclass | 95.11% | 0.7036 | 46,799 | 0.18 MB | 426s |
| E07 | GRU | Multiclass | 94.99% | 0.6904 | 32,719 | 0.13 MB | 380s |
| **E08** | **CNN-GRU (proposed)** | **Multiclass** | **94.74%** | **0.6445** | **21,583** | **0.09 MB** | **470s** |

Full precision/recall/FPR/FNR/inference-latency numbers for every row: `results/experiment_registry.csv`. Per-experiment classification reports and feature importances: `results/experiments/<id>/`.

---

## Live Training Dashboard

A FastAPI dashboard serves live pipeline status, per-epoch training progress, and a tailing activity log.

```bash
# Terminal 1 — the dashboard server
uvicorn dashboard.api:app --reload --port 8000

# Terminal 2 — the status poller (writes pipeline_status.json every 2s)
python pipeline_monitor.py
```

Then open **http://localhost:8000/pipeline**. It shows:
- Pipeline step cards (dataset download → CUDA → preprocessing → binary DL → multiclass DL → SHAP), each with live percentage and detail text.
- An experiment table with live per-epoch status (`Epoch 7/15 · train 0.12 · val 0.11`) for whatever is currently training.
- An Activity Log panel tailing `pipeline.log` in real time, with a **Maximize** button (top-right of the panel) to expand it full-screen — click the backdrop or press `Esc` to close it.

`http://localhost:8000/` serves a simpler project-status/experiments-summary page. `/api/status`, `/api/experiments`, `/api/experiments/{id}`, `/api/audit`, `/api/labels`, `/api/logs`, `/api/pipeline-status`, and `/api/training-progress` are the underlying JSON endpoints.

Every logger created via `src.utils.setup_logger()` (preprocessing, training, explainability, and any pipeline-runner script) appends to `pipeline.log` at the repo root — this is what `/api/logs` tails, and what makes the dashboard's live log panel work regardless of which terminal or process actually ran the training.

`src/training.py` also writes `training_progress.json` after every epoch (current experiment, epoch, train/val loss) — that's what makes the experiment table's live epoch detail possible without parsing the log text.

---

## ⚠️ Known Gotcha: SHAP must run against BINARY-preprocessed data

`src/preprocessing.py` **refits a fresh `OneHotEncoder` and `StandardScaler`** every time it runs, based on whichever `cfg.DEFAULT_TARGET_COL` is currently set (`Attack_label` for binary, `Attack_type` for multiclass). Because the two targets stratify the train/val/test split differently, the fitted encoder can end up with a **different number of output columns** between a binary run and a multiclass run (a rare categorical value may or may not land in the training fold depending on the split).

**Concretely:** if you run `run_phase3_multiclass.py` (which overwrites `data/processed/*.parquet` for the `Attack_type` target) and *then* run `run_explainability.py` without re-preprocessing for `Attack_label` first, SHAP will load E03/E04/E05 (binary models) against multiclass-preprocessed data:
- **GRU (E04) will hard-crash** on `model.load_state_dict(...)` with a `size mismatch` error, because `nn.GRU`'s weight matrix shape is locked to `input_dim`.
- **1D-CNN (E03) and CNN-GRU (E05) will *not* crash**, but silently compute SHAP values against the wrong scaler/encoding — because their `Conv1d`/`Linear` layer sizes only depend on `input_dim // 2`, and a difference of exactly one feature can floor-divide to the same value. This is worse than a crash: it produces plausible-looking but invalid explainability artifacts with no error at all.

**The fix, if you ever need to redo this:** before running SHAP for the binary experiments, force-regenerate binary-consistent data:
```python
import src.config as cfg
from src.preprocessing import run_preprocessing_pipeline
cfg.DEFAULT_TARGET_COL = "Attack_label"
run_preprocessing_pipeline()
```
Then run `run_explainability.py` (or call `run_shap_analysis(...)` directly). This is exactly what `run_shap_fix.py` in this repo does — keep it around as a template, or as a one-liner to rerun if you ever need to regenerate the binary SHAP artifacts again after another multiclass run.

There's also a second, independent SHAP bug already patched in `src/explainability.py`: newer versions of the `shap` library return an extra trailing output-dimension axis (`(N, D, 1)`) for single-logit binary models instead of `(N, D)`. The code now squeezes that away before slicing a single sample for the waterfall plot — if you upgrade `shap` and see a `waterfall plot can currently only plot a single explanation` error again, that's the same class of bug resurfacing with a different array shape.

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
pip install torch torchvision
pip install -r dashboard/requirements_dashboard.txt   # only needed for the live dashboard
```

### 3. Collaborator Setup (Dataset Download)
Since the `1.6GB` dataset and large `.parquet` intermediate files are not pushed to GitHub, collaborators must download the original dataset locally.
We have written an automation script that handles this safely:
```bash
python download_dataset.py
```
This script uses `kagglehub` to download the exact dataset version and places `DNN-EdgeIIoT-dataset.csv` into `data/raw/` automatically.

### 4. Full Pipeline Execution Order
```bash
# Step 1: Download dataset (once)
python download_dataset.py

# Step 2: Preprocessing (binary target)
python -m src.preprocessing

# Step 3: Phase 3A — Binary DL experiments (E03, E04, E05)
python run_phase3_binary.py

# Step 4: Phase 3B — Multiclass DL experiments (E06, E07, E08)
#         (re-preprocesses data/processed/* for the Attack_type target)
python run_phase3_multiclass.py

# Step 5: Phase 4 — SHAP Explainability (binary models only)
#         ⚠️ See "Known Gotcha" above — data/processed/* is currently
#         multiclass-preprocessed after Step 4. Re-run binary preprocessing
#         first, or just run run_shap_fix.py which does this for you.
python run_explainability.py

# Step 6: Launch Dashboard
uvicorn dashboard.api:app --reload
python pipeline_monitor.py   # in a second terminal
```

Alternatively, `run_remaining_pipeline.py` chains Steps 3–5 (skipping whatever's already complete) into a single script — see [Next Steps](#next-steps) for how to adapt it for further work.

## Reproducibility
- Every metric reported must come from an actual execution on the full dataset (no arbitrary downsampling).
- Global seed (`42`) controls random splits, numpy, and PyTorch layer initialization for replicability — `src.config.set_seeds()` seeds `random`, `numpy`, and `torch` (CPU and CUDA) consistently, so re-running preprocessing with the same target column reproduces the exact same train/val/test split and encoder fit.
- See `results/experiment_registry.csv` for tracked metric outputs across all tested models.
- Dataset SHA-256 hash is verified on every run: `1d3ef6c7cc22784a528a117f1158f5cad750273679437e18a39c360ee6b79fd7`

---

## Next Steps

Training and explainability are done. What's left is analysis, write-up, and optional deepening:

1. **Comparative analysis & ablation write-up.** All 8 experiments plus SHAP are in `results/`. Write the comparative section: binary models are all near-ceiling (1.0 F1) so the interesting story is in the multiclass numbers (E06 vs E07 vs E08) and per-class breakdowns in each `classification_report.csv` — which attack types drag macro-F1 down, and does the CNN-GRU hybrid actually help there or just match the baselines?
2. **Per-class multiclass diagnostics.** Macro F1 sits around 0.64–0.70 for E06–E08 despite ~95% accuracy — pull the per-class rows out of `results/experiments/E06_cnn1d_multiclass/classification_report.csv` (and E07/E08) to identify exactly which minority attack types are underperforming, and consider whether class-weighted loss or oversampling is worth a follow-up experiment.
3. **SHAP on the multiclass models (E06–E08).** Phase 4 as originally scoped only covers the binary models. `run_shap_analysis()` already supports `task_type="multiclass"` (it averages `|SHAP|` across the softmax outputs) — extending `run_explainability.py`'s experiment list to include E06/E07/E08 is a small addition, mindful of the same binary-vs-multiclass preprocessing gotcha above (in this direction it's simpler: multiclass SHAP needs multiclass-preprocessed data, so don't run it back-to-back with the binary SHAP step without re-preprocessing in between).
4. **MQTT-field ablation.** All three binary models rank an MQTT protocol field as their top SHAP feature (`mqtt.topic`, `mqtt.protoname`, `mqtt.conack.flags`). Worth an ablation experiment: retrain without MQTT-prefixed features and see how much accuracy degrades — this would meaningfully strengthen the explainability section by showing the SHAP ranking is causally, not just correlationally, important.
5. **Explore GradCAM / attention-based explainability** as a second XAI method to compare against SHAP, since the README's original XAI scope mentioned GradCAM as a possibility alongside SHAP.
6. **Report & Publication (Phase 5, not started).** Final research paper write-up: methodology, leakage-audit rationale, the full comparative table above, SHAP figures, discussion, and limitations (particularly the GRU's explicitly-exploratory seq_len=1 framing, and the class-imbalance effect on multiclass macro-F1).
7. **Dashboard/inference hardening (optional, not required for the paper).** `POST /api/predict` in `dashboard/api.py` is currently a stub that just checks the model file exists — wiring it up to actually run the saved `cnn_gru_final.pt` on a submitted feature vector would make the dashboard demo-able end-to-end, not just a results viewer.

### For another contributor/agent picking this up

If you're starting fresh (or resuming after this session), read `pipeline.log` and `pipeline_status.json` first — they capture exactly what was run and in what order, more reliably than trying to infer it from file timestamps.

- **To check what's actually done vs. still pending:** `results/experiment_registry.csv` has one row per completed experiment (E01–E08 as of this writing) with real metrics — if an experiment ID isn't a row there, it hasn't finished. `results/experiments/<id>/experiment_record.json` existing is the per-experiment equivalent of "this one is done."
- **To resume/extend training:** don't just rerun `run_phase3_binary.py` as-is — E03 is commented out in its `main()` because it's already done; re-enable it only if you intend to retrain it. Prefer writing a small one-off runner script (see `run_remaining_pipeline.py` as a template) that only calls `train_dl_model(...)` for the specific experiment IDs you actually need, so you don't waste 5-7 minutes of GPU time re-running something that already succeeded.
- **Before touching `data/processed/*.parquet`:** check which target column it currently reflects. There's no marker file for this — the safest check is `pd.read_parquet('data/processed/y_train.parquet').nunique()`: `2` means binary, `15` means multiclass. Re-preprocessing overwrites this data for whichever `cfg.DEFAULT_TARGET_COL` is active, and any binary model you try to load/explain afterward needs it back in binary form (see the Known Gotcha above).
- **Live progress while a long run is going:** start `pipeline_monitor.py` and `uvicorn dashboard.api:app --reload` (see Live Training Dashboard above) *before* kicking off training, then watch `http://localhost:8000/pipeline`. `pipeline.log` (repo root) is the ground truth for exactly what happened and when — every logger in this codebase appends to it, so `tail -f pipeline.log` (or `grep` it for `Error|Traceback|Finished|Epoch`) is the fastest way to see if something's stuck, crashed, or progressing normally, even without the dashboard running.
- **If GPU/CUDA isn't available:** `src/training.py` auto-falls-back to CPU (`torch.device("cuda" if torch.cuda.is_available() else "cpu")`), but expect binary experiments to take much longer than the ~6-7 minutes each took here on an RTX 4060 Laptop GPU.
