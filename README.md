# Explainable Hybrid Deep Learning-Based Intrusion Detection System for IoT Networks

A lightweight, explainable intrusion detection system (IDS) for IoT: a hybrid **CNN-GRU** with
**SHAP** explanations, evaluated honestly on **Edge-IIoTset** (2,219,201 packets, Normal + 14 attack
types) and **CICIoT2023** (flow-level, 33 attacks in 7 categories + Benign).

**Full results and the presentation source of truth: [docs/FINAL_RESULTS.md](docs/FINAL_RESULTS.md).**
Details of the leakage fix: [docs/LEAKAGE_FIX_AND_CLEAN_BASELINES.md](docs/LEAKAGE_FIX_AND_CLEAN_BASELINES.md).

---

## Status (October 2026)

Done: shortcut audit and fix, strict feature policy, clean baselines, CNN-GRU tuning (3 seeds),
ablation against simpler models, SHAP with a fidelity test, edge (CPU) benchmark, measured
detection ceiling, and the same pipeline on a second dataset (CICIoT2023). Remaining work is listed
under [Next steps](#limitations-and-next-steps).

### The first 100% accuracy was fake

All first models (Random Forest, CNN, GRU, CNN-GRU) scored 100% on Normal vs Attack. The cause was
a formatting artifact, not attack behaviour: Edge-IIoTset writes an empty protocol field as `"0"` in
the Normal captures and as `"0.0"` in the attack captures, so after one-hot encoding a single column
gives away the label.

| Empty `dns.qry.name.len` written as | Normal rows | Attack rows |
|---|---|---|
| `"0"` | 1,613,798 | 0 |
| `"0.0"` | 0 | 603,331 |

The same holds for `mqtt.topic`, `mqtt.protoname`, `mqtt.conack.flags`; the HTTP columns use `"0"`
only for web attacks. Removing MQTT features alone did not help (the DNS column carries the same
artifact), and the old SHAP plots hid it by stripping the `_0` / `_0.0` suffix from feature names.
Evidence: `results/audit/empty_token_audit.csv` (`python audit_shortcuts.py --raw`).

**Shortcuts found and removed**

| Dataset | Shortcut | Fix |
|---|---|---|
| Edge-IIoTset | `"0"` vs `"0.0"` spelling of empty fields | Canonicalise numeric-looking text before encoding |
| Edge-IIoTset | 7 per-packet identifiers (`tcp.seq`, `tcp.ack`, `tcp.ack_raw`, `tcp.checksum`, `icmp.checksum`, `icmp.seq_le`, `udp.stream`) | Removed (`strict` policy) |
| Edge-IIoTset | Every MQTT packet is Normal | All `mqtt.*` removed as a check (`strict_no_mqtt` policy) |
| CICIoT2023 | `source_file` names the capture (the label) | Removed |
| CICIoT2023 | `packet_count` is the extraction window size (10 for Benign/Recon, 100 for DDoS/DoS/Mirai); `total_sum = packet_count × mean_packet_size` | Both dropped; flag counts turned into fractions of the window |

---

## Results

### Edge-IIoTset, binary (Normal vs Attack), strict features, no MQTT (final setting)

Test set, `results/experiment_registry.csv`:

| Model | Macro-F1 | Accuracy | False alarms | Attacks detected | ROC-AUC |
|---|---|---|---|---|---|
| Original pipeline (leaky) | 1.000 | 100% | 0% | 100% | - |
| **CNN-GRU (tuned, ours)** | **0.858** | **90.0%** | **0.7%** | **65.3%** | 0.904 |
| XGBoost (GPU) | 0.868 | 90.7% | 0.6% | 67.3% | 0.917 |

* **Robust:** without vs with MQTT 0.8575 vs 0.8579; 3 seeds 0.8584 / 0.8583 / 0.8582; Optuna (20
  trials) best validation 0.8596, test 0.858, same as untuned.
* **Measured ceiling** (`results/audit/detection_ceiling.json`): the 1,553,440 training packets
  contain only 11,122 distinct feature patterns; 31.9% of attack packets look exactly like patterns
  that are mostly Normal. A perfect "memorise every pattern" classifier detects only 67.5% of
  attacks (at 0.5% false alarms), so both models are at the ceiling.
* **Trade-off** (`results/operating_points.csv`, thresholds chosen on validation): lowering the
  threshold raises CNN-GRU detection to 73.3% at 6.6% false alarms.

### Ablation: is the hybrid needed? (3 seeds each)

| Model | Macro-F1 (mean) | Parameters | File size |
|---|---|---|---|
| MLP | 0.858 | 5,441 | 24 KB |
| 1D-CNN | 0.858 | 25,857 | 106 KB |
| GRU | 0.858 | 24,577 | 99 KB |
| CNN-GRU (tuned) | 0.858 | 79,169 | 315 KB |
| XGBoost | 0.868 | 226 trees | 492 KB |

On per-packet features the hybrid gives no accuracy gain; all networks hit the same ceiling.

### Explainability with a fidelity test (`results/shap_fidelity.json`)

* SHAP top features (final CNN-GRU): `tcp.flags`, `tcp.connection.rst`, `tcp.flags.ack`, `tcp.len`,
  `tcp.connection.fin` (real TCP behaviour).
* Retrain without the top-5 SHAP features: Macro-F1 0.858 → **0.605**. Without 5 random features:
  0.858 → 0.858. SHAP identifies what the model really depends on.

### Edge benchmark (1 CPU thread, `results/edge_benchmark.csv`)

| Version | Macro-F1 | Size | Latency per packet | Throughput |
|---|---|---|---|---|
| CNN-GRU FP32 | 0.858 | 315 KB | 6.4 ms | 1,500 rows/s |
| **CNN-GRU FP16 weights** | **0.858** | **159 KB** | 6.5 ms | 1,660 rows/s |
| CNN-GRU INT8 (Linear only) | 0.858 | 304 KB | 8.1 ms | 1,620 rows/s |
| CNN-GRU INT8 (GRU + Linear) | 0.782 | 88 KB | 13.3 ms | 2,050 rows/s |
| XGBoost | 0.868 | 492 KB | 0.8 ms | 30,700 rows/s |

### Attack type (14 attacks + Normal), Edge-IIoTset strict

Weak on packet data: XGBoost 0.429 Macro-F1, CNN-GRU 0.337 ± 0.033. Loss study over 3 seeds: sqrt
class weights 0.337 > focal 0.254 > cross-entropy 0.238. Floods and scans are defined by packet
rate, which a single packet cannot show (`udp.time_delta` is 0 for every attack row).

### CICIoT2023 (flow-level, with rate and timing): `python run_ciciot.py --policy strict`

1.41M-row capped sample, 87,417 duplicates removed before splitting; shortcut scan after the fix:
best single feature 0.85, depth-3 tree 0.85 (no giveaway).

| Task | Model | Macro-F1 | Notes |
|---|---|---|---|
| Benign vs Attack | XGBoost | **0.917** | detects 93.4%, 8.4% false alarms; ROC-AUC 0.979 |
| Benign vs Attack | CNN-GRU | **0.890** | detects 90.4%, 9.6% false alarms; ROC-AUC 0.966 |
| 8 categories | XGBoost | **0.752** | |
| 8 categories | CNN-GRU, sqrt weights | **0.675** | cross-entropy: 0.620 |

At ≤ 1% false alarms: XGBoost detects 86.6%, CNN-GRU 82.3%. SHAP (CNN-GRU): `header_length`,
`ack_count_frac`, `rate`, `https`, `arp`.

### Figures (`results/figures/`, built by `make_figures.py`)

* [fig1_fake_vs_honest.png](results/figures/fig1_fake_vs_honest.png): leaky vs honest results
* [fig2_shap_fidelity.png](results/figures/fig2_shap_fidelity.png): SHAP removal test
* [fig3_model_comparison.png](results/figures/fig3_model_comparison.png): ablation
* [fig4_datasets.png](results/figures/fig4_datasets.png): Edge-IIoTset vs CICIoT2023
* [fig5_detection_ceiling.png](results/figures/fig5_detection_ceiling.png): detection ceiling

---

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt       # includes torch, optuna, xgboost, psutil
pip install rich                      # used by watch_dashboard.py
pip install -r dashboard/requirements_dashboard.txt   # only for the FastAPI web dashboard
```

For GPU training install the CUDA build of PyTorch (tested on an RTX 4060 Laptop GPU); everything
also runs on CPU, more slowly.

**Datasets** (not in the repository):

* **Edge-IIoTset:** `python download_dataset.py` downloads it with `kagglehub` and places
  `DNN-EdgeIIoT-dataset.csv` in `data/raw/`. SHA-256 of the file used:
  `1d3ef6c7cc22784a528a117f1158f5cad750273679437e18a39c360ee6b79fd7`.
* **CICIoT2023:** downloaded automatically by `run_ciciot.py` via `kagglehub`
  (`dhoogla/ciciotdataset2023`, 46.8M rows) into the user's kagglehub cache, **outside the project
  folder** (and outside OneDrive). Only the sampled, processed splits are written to
  `data/processed_ciciot*/`.

---

## How to run

```bash
python audit_shortcuts.py --raw                                    # "0"/"0.0" artifact evidence
python run_clean_baselines.py --policy strict_no_mqtt --stage binary  # preprocess + shortcut scan + baselines
python run_tune_cnn_gru.py --trials 20 --prefix F                  # Optuna tuning + 3 seeds
python run_final_studies.py                                        # ablation, SHAP, fidelity, edge benchmark
python run_operating_points.py && python audit_ceiling.py          # trade-off and detection ceiling
python run_ciciot.py --policy strict                               # CICIoT2023 (data + training)
python make_figures.py                                             # results/figures/*.png
python watch_dashboard.py                                          # live progress (separate window)
```

* `run_clean_baselines.py --policy` takes `operational`, `strict` or `strict_no_mqtt`;
  `--study losses --seeds 42 7 2024` runs the multiclass loss comparison.
* Long runs: `python run_queue.py "<script> <args>" ...` runs commands one after another and keeps
  Windows awake for the whole sequence. Keep the lid open and the laptop plugged in (on battery the
  GPU is throttled).

### Safeguards built into the pipeline

* `data/processed/metadata.json` records target, feature policy, canonicalisation flag, feature
  list and a fingerprint. Training refuses data built for a different target, and SHAP refuses to
  explain a model whose fingerprint does not match `data/processed/` (an earlier SHAP plot had
  explained a model with the wrong inputs).
* Automatic shortcut scan (single-feature stumps, depth-3 tree) after every preprocessing run.
* 3 seeds for every comparison; decision thresholds chosen on validation only.

### Live demo (for the presentation)

```bash
python -m uvicorn dashboard.api:app --port 8000
```

Open http://127.0.0.1:8000/#predict-demo. The final CNN-GRU classifies a real held-out test packet
(never seen in training), shows the attack probability, whether it was right, the CPU latency, and
the top SHAP reasons for that packet. Buttons pick a random attack, normal, or any packet. The model
loads in the background on start-up (about 30 s); it is only served when `data/processed/` matches
its training data (otherwise the page says which preprocessing command to run).

### Monitoring

* `python watch_dashboard.py`: terminal dashboard reading `run_status.json`,
  `training_progress.json`, `results/experiment_registry.csv` and `pipeline.log`. It can be started
  or stopped at any time without affecting the run.
* Web dashboard (optional): `uvicorn dashboard.api:app --reload --port 8000` plus
  `python pipeline_monitor.py` in a second terminal, then open http://localhost:8000/pipeline.

---

## Limitations and next steps

Limitations: on per-packet features the CNN-GRU is not better than an MLP or XGBoost; about a third
of Edge-IIoTset attack packets are indistinguishable from normal ones; the GRU sees no real time
dimension; attack-type detection needs flow/timing data; single split per dataset, no cross-dataset
test yet.

| Gap | Plan |
|---|---|
| Missed attacks on packet data | Use flow-level data (CICIoT2023) as the main setting; choose thresholds by acceptable false-alarm rate |
| Hybrid gives no gain | Give the GRU real sequences (consecutive flow windows per device/connection) |
| Weak rare classes | Sqrt class weights (already +0.06), targeted oversampling of Web/BruteForce |
| Generalisation | Cross-dataset test (train on one dataset, test on the other with shared features) |
| Deployment | FP16 model on a Raspberry-Pi-class device (live dashboard demo is already working) |

Results for every experiment: `results/experiment_registry.csv`; per-experiment artifacts in
`results/experiments/<id>/`.
