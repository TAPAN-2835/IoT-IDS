# 🎓 Complete Project Mastery & Defense Study Guide
### *Explainable Hybrid Deep Learning-Based Intrusion Detection System for IoT Networks*

> **Quick Reference**: Keep this document open during your revision. It covers the final, honest version of the project: what went wrong first, how we fixed it, the real results, where the code lives, and how to defend it. Every number comes from [`docs/FINAL_RESULTS.md`](docs/FINAL_RESULTS.md), the source of truth. If an older document disagrees, the older document is wrong.

---

## 📑 Table of Contents
1. [Core Project Summary & Story](#1-core-project-summary--story)
2. [Code Walkthrough: What to Open & Show](#2-code-walkthrough-what-to-open--show)
3. [Deep Dive: What is `best_model.pt`?](#3-deep-dive-what-is-best_modelpt)
4. [IoT Real-World Deployment Architecture](#4-iot-real-world-deployment-architecture)
5. [Complete Guide to Parameters & Hyperparameters](#5-complete-guide-to-parameters--hyperparameters)
6. [Metric Cheat-Sheet: What Every Result Means](#6-metric-cheat-sheet-what-every-result-means)
7. [The 5-Minute Faculty Presentation Script](#7-the-5-minute-faculty-presentation-script)
8. [Tough Faculty Q&A Defense](#8-tough-faculty-qa-defense)

---

## 1. Core Project Summary & Story

**Goal:** an IDS for IoT that is **accurate** (catches attacks, few false alarms), **explainable** (shows why it alarmed) and **lightweight** (fits an IoT gateway). Model: hybrid **CNN-GRU** with **SHAP**. Main dataset: **Edge-IIoTset** (2,219,201 packets, Normal + 14 attack types). Second dataset: **CICIoT2023** (flow-level, 33 attacks in 7 categories + Benign).

### What we first saw, and why it was fake
All first models (Random Forest, CNN, GRU, CNN-GRU) scored **100%** on Normal vs Attack. We investigated instead of celebrating. The dataset writes an *empty* protocol field as `"0"` in the Normal captures and `"0.0"` in the attack captures; after one-hot encoding one column alone gives away the label.

| Empty `dns.qry.name.len` written as | Normal rows | Attack rows |
|---|---|---|
| `"0"` | 1,613,798 | 0 |
| `"0.0"` | 0 | 603,331 |

The same holds for `mqtt.topic`, `mqtt.protoname`, `mqtt.conack.flags`. Our earlier "fix" of removing MQTT did not help (the DNS column carries the same artifact), and our SHAP plots had hidden it by stripping the `_0` / `_0.0` suffix from feature names.

### Honest pipeline (final)
```
[ Raw Edge-IIoTset ]
          ⬇
[ Canonicalise "0" = "0.0" before encoding ]
          ⬇
[ Strict feature policy: drop 7 per-packet identifiers + all mqtt.* (51 encoded features) ]
          ⬇
[ Shortcut scans (1-feature stumps, depth-3 tree) + data fingerprint guards ]
          ⬇
[ GPU training, 3 seeds, threshold chosen on validation only ]
          ⬇
[ Tuned CNN-GRU: Macro-F1 0.858 | 79,169 params | 315 KB (FP16: 159 KB) ]
          ⬇
[ Detection-ceiling audit | SHAP + fidelity test | Edge benchmark ]
          ⬇
[ Same pipeline on CICIoT2023 (second shortcut found and removed) ]
```

### The 4 Pillars
1. **Honest accuracy**: Macro-F1 **0.858** (90.0% accuracy, 0.7% false alarms, 65.3% of attacks detected). A perfect "memorise every pattern" classifier detects only **67.5%**, so we are **at the ceiling** of packet-level features.
2. **Proven explainability**: SHAP top features are real TCP behaviour (`tcp.flags`, `tcp.connection.rst`, `tcp.flags.ack`, `tcp.len`, `tcp.connection.fin`). Retraining without the top-5 SHAP features: **0.858 → 0.605**; without 5 random features: unchanged.
3. **Edge feasibility**: 79,169 parameters, 315 KB; **FP16 159 KB** with no loss; ~6.5 ms per packet on 1 CPU thread.
4. **Second dataset**: CICIoT2023, after removing a window-size shortcut: binary Macro-F1 **0.890** (CNN-GRU) / **0.917** (XGBoost).

Figures (`results/figures/`): [fig1 fake vs honest](results/figures/fig1_fake_vs_honest.png) · [fig2 SHAP fidelity](results/figures/fig2_shap_fidelity.png) · [fig3 model comparison](results/figures/fig3_model_comparison.png) · [fig4 datasets](results/figures/fig4_datasets.png) · [fig5 detection ceiling](results/figures/fig5_detection_ceiling.png)

---

## 2. Code Walkthrough: What to Open & Show

Follow the story: evidence → fix → training → explanations → experiments → results.

| File | What It Contains | What to Say |
|---|---|---|
| [`audit_shortcuts.py`](audit_shortcuts.py) | `--raw`: counts how each class writes empty fields → `results/audit/empty_token_audit.csv`. Also stumps + depth-3 tree on processed data | *"This proves the 100% was fake: one column, "0" vs "0.0", separates Normal from Attack."* |
| [`src/preprocessing.py`](src/preprocessing.py) | `canonicalize_column()` merges `"0"`/`"0.0"`; `feature_fingerprint()` stamps the processed data; streaming, low-RAM (peak 6.1 → 3.4 GB) | *"We canonicalise, apply the strict policy and fingerprint the data so nothing trains or explains on the wrong inputs."* |
| [`src/training.py`](src/training.py) | Data on the GPU (≈8 s/epoch vs ≈31 s), Adam, early stopping, sqrt class weights option, `choose_threshold()` on validation | *"The test set is never used for any choice; every comparison has 3 seeds."* |
| [`src/models.py`](src/models.py) | `MLP`, `CNN1D`, `GRUBaseline`, proposed `CNN_GRU` | *"Conv learns local feature combinations; the GRU reads the feature map step by step."* |
| [`src/explainability.py`](src/explainability.py) | `shap.GradientExplainer`; `_check_data_matches_experiment()` refuses mismatched data | *"An earlier SHAP plot explained a model with the wrong inputs; this guard prevents that."* |
| [`run_clean_baselines.py`](run_clean_baselines.py) | XGBoost (GPU) + CNN-GRU on clean data per policy | *"The honest numbers after the fix."* |
| [`run_tune_cnn_gru.py`](run_tune_cnn_gru.py) | Optuna, 20 trials, then 3 seeds of the best setting | *"Validation 0.8596, test 0.858: tuning cannot beat the ceiling."* |
| [`run_final_studies.py`](run_final_studies.py) | Ablation, SHAP, SHAP fidelity, edge benchmark | *"Here we test our own claims."* |
| [`run_edge_benchmark.py`](run_edge_benchmark.py) | FP32 / FP16 / INT8 / XGBoost on 1 CPU thread → `results/edge_benchmark.csv` | *"A gateway has no GPU, so we measured on one CPU thread."* |
| [`audit_ceiling.py`](audit_ceiling.py) | Identical-pattern analysis → `results/audit/detection_ceiling.json` | *"11,122 distinct patterns in 1,553,440 training packets; ceiling 67.5%."* |
| [`run_ciciot.py`](run_ciciot.py) + [`src/ciciot.py`](src/ciciot.py) | CICIoT2023 pipeline (`--policy strict`): dedup before split, drop `source_file`, `packet_count`, `total_sum` | *"We found a second shortcut here and removed it."* |
| [`watch_dashboard.py`](watch_dashboard.py) | Live terminal dashboard (epoch, loss, early stopping, finished runs) | *"Long runs are monitored live without affecting them."* |
| [`make_figures.py`](make_figures.py) | Builds `results/figures/fig1..fig5` from result files | *"Every chart is generated from result files."* |

### Where the Actual Results Live
- **Master registry**: [`results/experiment_registry.csv`](results/experiment_registry.csv) (all runs, including the old leaky `E01`–`E08`; final Edge-IIoTset runs are `N01`/`N02`, `F_*`, `A_*`, `FID_*`; final CICIoT2023 runs are `CS01`–`CS05`).
- **Final model**: `results/experiments/F_strict_no_mqtt_binary_best_s42/best_model.pt` (+ `experiment_record.json`, confusion matrix, SHAP plots).
- **Audits**: `results/audit/empty_token_audit.csv`, `results/audit/detection_ceiling.json`.
- **Other**: `results/shap_fidelity.json`, `results/edge_benchmark.csv`, `results/operating_points.csv`, charts in `results/figures/`.

---

## 3. Deep Dive: What is `best_model.pt`?

### What is it?
- The PyTorch file holding the trained weights (`state_dict`). The training loop saves it every time **validation loss** improves, so it is the best epoch on validation.
- Final file: `results/experiments/F_strict_no_mqtt_binary_best_s42/best_model.pt`: tuned CNN-GRU, **79,169 parameters**, **315 KB** (FP16 copy **159 KB**). SHAP and the edge benchmark load this file.
- ⚠️ The old `E01`–`E08` checkpoints were trained on leaky data. Their "100%" belongs only to the "what we first saw" story.

### How it compares (Edge-IIoTset, final setting, 3 seeds)
| Model | Macro-F1 | Parameters | File size |
|---|---|---|---|
| MLP | 0.858 | 5,441 | 24 KB |
| 1D-CNN | 0.858 | 25,857 | 106 KB |
| GRU | 0.858 | 24,577 | 99 KB |
| **CNN-GRU (tuned, ours)** | **0.858** | **79,169** | **315 KB** (FP16 159 KB) |
| XGBoost | 0.868 | 226 trees | 492 KB |

Honest reading: on per-packet features **the hybrid gives no accuracy gain**; all networks hit the same ceiling.

---

## 4. IoT Real-World Deployment Architecture

Sensors are too weak to run the model; it is meant for the **IoT edge gateway** (Raspberry-Pi-class device). We benchmarked on a laptop CPU with 1 thread; a real Pi test is a next step.

```
[ Smart Sensor / Camera ]
         │ (traffic)
         ▼
┌──────────────────────────────────────────────────────────────┐
│                IoT Edge Gateway (Pi-class device)            │
│                                                              │
│  1. Packet capture                                           │
│  2. Feature extraction (same 51 encoded features, same       │
│     canonicalisation and strict policy as training)          │
│  3. Inference: FP16 CNN-GRU, 159 KB, ~6.5 ms/packet (1 CPU)   │
│                                                              │
│      ┌───────────────────────┴───────────────────────┐       │
│      ▼                                               ▼       │
│ [ score ≤ threshold: Normal ]      [ score > threshold: ALERT ]│
│                                     + SHAP explanation        │
└──────────────────────────────────────────────────────────────┘
```
The threshold is chosen on validation and can be moved to trade detections against false alarms.

| Version (1 CPU thread) | Macro-F1 | Size | Latency / packet | Throughput |
|---|---|---|---|---|
| CNN-GRU FP32 | 0.858 | 315 KB | 6.4 ms | 1,500 rows/s |
| **CNN-GRU FP16 weights** | **0.858** | **159 KB** | 6.5 ms | 1,660 rows/s |
| CNN-GRU INT8 (Linear only) | 0.858 | 304 KB | 8.1 ms | 1,620 rows/s |
| CNN-GRU INT8 (GRU + Linear) | 0.782 | 88 KB | 13.3 ms | 2,050 rows/s |
| XGBoost | 0.868 | 492 KB | 0.8 ms | 30,700 rows/s |

### What this means for IoT
- **No sensor overhead**: sensors compute nothing.
- **Small**: FP16 halves the size with no loss; INT8 on the GRU costs 7.6 points.
- **Honest trade-off**: XGBoost is faster on a CPU; the networks are smaller (MLP 24 KB).
- **Alerting, not auto-blocking**: with 65.3% of attacks detected on packet data, the system raises explainable alerts.

---

## 5. Complete Guide to Parameters & Hyperparameters

### A. Training Hyperparameters (final model, Optuna 20 trials)
- **Batch size `8192`**: large because the data is held on the GPU.
- **Learning rate `0.0030`** with the **Adam** optimizer.
- **Dropout `0.239`**: switches off about 24% of neurons during training.
- **Loss: cross-entropy** (binary). The loss study (multiclass, 3 seeds) found sqrt class weights 0.337 > focal 0.254 > cross-entropy 0.238.
- **Epochs `15`** with early stopping on validation loss; `best_model.pt` keeps the best epoch.
- **Decision threshold**: tuned on validation only.
- **Seeds**: 42 / 7 / 2024 → 0.8584 / 0.8583 / 0.8582.

### B. Architecture Parameters
- **Conv1D (64 filters, kernel 5)** → BatchNorm → ReLU → **MaxPool(2)** → Dropout.
- **GRU (hidden 128)**: reads the pooled feature map step by step.
- **Dense (32)** → 1 output (attack score).
- **Total learnable parameters: 79,169** (315 KB FP32, 159 KB FP16).

### C. Input Features (Edge-IIoTset final setting: `strict_no_mqtt`, 51 encoded features)
- **Start**: operational policy (already no IPs, ports or timestamps).
- **Removed**: 7 per-packet identifiers (`tcp.seq`, `tcp.ack`, `tcp.ack_raw`, `tcp.checksum`, `icmp.checksum`, `icmp.seq_le`, `udp.stream`) and all `mqtt.*` fields (every MQTT packet in the dataset is Normal).
- **Kept**: TCP (`tcp.flags`, `tcp.flags.ack`, `tcp.len`, `tcp.connection.syn/synack/fin/rst`), HTTP (method, version, referer, content length, response), DNS (query name/length/type, retransmissions), ARP, ICMP, Modbus length.
- **CICIoT2023**: flow-level features (e.g. `header_length`, `rate`, flag fractions, protocol indicators) with `source_file`, `packet_count`, `total_sum` removed.

---

## 6. Metric Cheat-Sheet: What Every Result Means

Edge-IIoTset, binary, tuned CNN-GRU (XGBoost in brackets):

| Metric | Value | What It Means | Why It Matters |
|---|---|---|---|
| **Accuracy** | 90.0% (90.7%) | Share of all packets classified correctly | Misleading on imbalanced data, so we lead with Macro-F1 |
| **Macro-F1** | **0.858** (0.868) | Per-class F1 averaged equally | Headline number; Attack counts as much as Normal |
| **Detection rate (recall)** | 65.3% (67.3%) | Of real attacks, how many we caught | Missed attacks |
| **False-alarm rate (FPR)** | 0.7% (0.6%) | Of normal packets, how many we flagged | Admin trust |
| **Precision** | High (few false alarms) | When it alarms, is it a real attack? | Alarm quality |
| **ROC-AUC** | 0.904 (0.917) | How well scores rank attacks above normal, over all thresholds | 1.0 perfect, 0.5 guessing |
| **Detection ceiling** | 67.5% at 0.5% false alarms | Best *any* model can do: 31.9% of attack packets look exactly like mostly-Normal patterns | Shows we are at the limit |
| **Operating point** | 73.3% detection at 6.6% false alarms | Lower threshold = more detections, more false alarms | Tunable to the site |
| **SHAP fidelity** | top-5 removed: 0.858 → 0.605; 5 random: 0.858 → 0.858 | Do SHAP's top features really matter? | Proves the explanation is faithful |
| **Latency / size** | 6.5 ms, 159 KB (FP16) | Time per packet on 1 CPU thread; size on disk | Edge feasibility |
| **Multiclass Macro-F1** | 0.337 ± 0.033 (0.429) | Attack *type*, 14 + Normal | Weak: single packets carry no rate |

**CICIoT2023 (flow-level)**

| Task | XGBoost | CNN-GRU |
|---|---|---|
| Benign vs Attack, Macro-F1 | 0.917 | 0.890 |
| Detection / false alarms | 93.4% / 8.4% | 90.4% / 9.6% |
| ROC-AUC | 0.979 | 0.966 |
| Detection at ≤ 1% false alarms | 86.6% | 82.3% |
| 8 categories, Macro-F1 | 0.752 | 0.675 (sqrt weights; CE 0.620) |

CNN-GRU, Edge-IIoTset → CICIoT2023: binary 0.86 → 0.89, categories 0.34 → 0.68. SHAP on CICIoT2023: `header_length`, `ack_count_frac`, `rate`, `https`, `arp`.

---

## 7. The 5-Minute Faculty Presentation Script

### Stage 1: Problem Statement (0:00 – 0:40)
> *"Good morning sir/ma'am. Our project is **Explainable Hybrid Deep Learning-Based Intrusion Detection for IoT Networks**. IoT devices are easy to attack and too weak to run heavy security software. We want an IDS that is **accurate**, **explainable** and **light enough for an IoT gateway**: a hybrid **CNN-GRU** with **SHAP**, tested on **Edge-IIoTset** and **CICIoT2023**."*

### Stage 2: The 100% Was Fake (0:40 – 1:40)
> *"Our first models all scored **100%**. Instead of celebrating, we investigated. The dataset writes an empty field as **"0" in normal traffic and "0.0" in attack traffic**, so one column gives away the label. We treat "0" and "0.0" as the same value, removed 7 per-packet identifiers and all MQTT fields, and added shortcut scans and data-fingerprint guards. Figure 1 shows the fake 100% next to the honest result."*

### Stage 3: Honest Results & the Ceiling (1:40 – 2:50)
> *"On clean data our tuned CNN-GRU reaches **Macro-F1 0.858**: 90% accuracy, **0.7% false alarms**, and **65.3% of attacks** detected, stable over 3 seeds; 20 Optuna trials do not raise it. We measured why: 1.55 million training packets have only 11,122 distinct patterns, and a third of attack packets look exactly like normal ones. Even a perfect memoriser detects only **67.5%**, so our model is **at the ceiling**."*

### Stage 4: Explainability That Is Proven (2:50 – 3:40)
> *"SHAP says the model relies on real TCP behaviour: **tcp.flags, resets, ACK flags, packet length**. We tested it: retraining without SHAP's top-5 features drops Macro-F1 from **0.858 to 0.605**; removing 5 random features changes nothing. The explanation is faithful."*

### Stage 5: Edge & Second Dataset (3:40 – 4:30)
> *"The final model has 79,169 parameters; in FP16 it is **159 KB** with no loss, about 6.5 ms per packet on one CPU thread. On **CICIoT2023**, which has rate and timing, we found and removed a second hidden shortcut; the CNN-GRU reaches **0.890** binary and **0.675** on 8 categories, up from 0.34 on packet data."*

### Stage 6: Honest Limits & Next Steps (4:30 – 5:00)
> *"On per-packet features the CNN-GRU is **not better than an MLP or XGBoost**, because the GRU sees no real time dimension. Next: real sequences of flow windows for the GRU, a cross-dataset test, and the FP16 model on a Raspberry-Pi-class device. Contribution: two dataset shortcuts found and removed, a measured ceiling, proven SHAP fidelity, and reproducible evaluation. Thank you."*

---

## 8. Tough Faculty Q&A Defense

### Q1: *"Your first models got 100%. Why is the final result lower?"*
> **Answer**: *"The 100% was fake: empty fields were "0" in normal captures and "0.0" in attack captures, so one column revealed the label. Every model, including Random Forest, exploited it. The honest result is **0.858 Macro-F1**."*

### Q2: *"Isn't 65% detection too low?"*
> **Answer**: *"It is low, and we measured why: 31.9% of attack packets are identical to mostly-normal patterns. The ceiling is **67.5%**; we reach 65.3% (XGBoost 67.3%). Lowering the threshold gives 73.3% at 6.6% false alarms, and flow-level data (CICIoT2023) gives 90.4% detection."*

### Q3: *"If an MLP gets the same 0.858, why the CNN-GRU?"*
> **Answer**: *"On per-packet features the hybrid gives **no accuracy gain**; all networks hit the ceiling. The GRU needs a real time dimension; our next step feeds it consecutive flow windows per device or connection."*

### Q4: *"XGBoost is better and faster. Why not just use it?"*
> **Answer**: *"We report it openly: 0.868 vs 0.858, 0.8 ms vs 6.5 ms. The networks are smaller (FP16 159 KB, MLP 24 KB vs 492 KB). If CPU speed on per-packet data were the only goal, XGBoost would be fair; we keep the CNN-GRU for the planned sequence version, where a recurrent model can use time."*

### Q5: *"How do you know SHAP is right?"*
> **Answer**: *"Removing SHAP's top-5 features drops Macro-F1 to **0.605**; removing 5 random features changes it by 0.0001. A guard also stops SHAP from running on data the model was not trained on."*

### Q6: *"You earlier said MQTT fields drive detection. What changed?"*
> **Answer**: *"That came from the leaky data: MQTT fields carried the "0"/"0.0" artifact, and every MQTT packet is Normal. With vs without MQTT now gives 0.8579 vs 0.8575, so the model does not use them. The real top features are TCP flags, resets, ACKs and length."*

### Q7: *"Why is attack-type detection weak? Did you try focal loss?"*
> **Answer**: *"Macro-F1 0.337 (CNN-GRU) and 0.429 (XGBoost): floods and scans are defined by rate, which one packet cannot show, and `udp.time_delta` is 0 for every attack row. Over 3 seeds, sqrt class weights (0.337) beat focal loss (0.254) and cross-entropy (0.238). On CICIoT2023 it rises to 0.675 / 0.752."*

### Q8: *"How can we trust there is no other leakage?"*
> **Answer**: *"Shortcut scans (one-feature stumps, depth-3 tree), validation-only thresholds, 3 seeds, and data fingerprints. Limitations: one random stratified split per dataset and no cross-dataset test yet."*

### Q9: *"Where does the model sit, and how does SHAP work?"*
> **Answer**: *"On the **IoT edge gateway**: 159 KB, ~6.5 ms per packet on one CPU thread, raising explainable alerts (real Pi test is next). SHAP uses Shapley values from game theory: each feature is a player, and SHAP measures how much it pushed this packet toward Attack or Normal."*

---
*Created for Minor Project Defense: Explainable Hybrid Deep Learning-Based Intrusion Detection System for IoT Networks. Source of truth: `docs/FINAL_RESULTS.md`.*
