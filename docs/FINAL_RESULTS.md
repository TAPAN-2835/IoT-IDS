# Final Results: Explainable Lightweight IDS for IoT

*Source of truth for the presentation (2026-10-03). Every number here comes from a committed
result file; the file is named next to each table. Older documents that disagree are superseded.*

---

## 1. Problem statement

IoT devices are easy to attack and too weak to run heavy security software. Goal: an intrusion
detection system (IDS) that is

1. **accurate**: detects attacks with few false alarms,
2. **explainable**: shows *why* it raised an alarm, and
3. **lightweight**: small and fast enough for an IoT gateway.

Proposed model: hybrid **CNN-GRU** (1D convolution then a GRU) with **SHAP** explanations.
Main dataset: **Edge-IIoTset** (2,219,201 packets, Normal + 14 attack types). Second dataset:
**CICIoT2023** (flow-level, 33 attacks in 7 categories + Benign).

---

## 2. What went wrong first: the 100% accuracy was fake

All first models (Random Forest, CNN, GRU, CNN-GRU) scored **100%** on Normal vs Attack. We
investigated instead of celebrating.

**Root cause:** the dataset writes an empty protocol field as `"0"` in the Normal captures and as
`"0.0"` in the attack captures. After one-hot encoding these become different columns, so one
column alone gives away the label.

| Empty `dns.qry.name.len` written as | Normal rows | Attack rows |
|---|---|---|
| `"0"` | 1,613,798 | 0 |
| `"0.0"` | 0 | 603,331 |

The same holds for `mqtt.topic`, `mqtt.protoname`, `mqtt.conack.flags`; the HTTP columns use
`"0"` only for web attacks (leaking the attack *type*). Evidence: `results/audit/empty_token_audit.csv`
(`python audit_shortcuts.py --raw`).

Removing MQTT features (the earlier "fix") did **not** help, because the DNS column carries the
same artifact. SHAP plots had hidden it by stripping the `_0` / `_0.0` suffix from feature names.

---

## 3. What we did, and why

| Step | What | Why |
|---|---|---|
| 1 | Canonicalised numeric-looking text (`"0"` = `"0.0"`) before encoding | Removes the formatting shortcut |
| 2 | Strict feature policy: removed 7 per-packet identifiers (`tcp.seq`, `tcp.ack`, `tcp.ack_raw`, `tcp.checksum`, `icmp.checksum`, `icmp.seq_le`, `udp.stream`) | They identify the capture file, not attack behaviour |
| 3 | Removed all `mqtt.*` fields as a check | Every MQTT packet in the dataset is Normal |
| 4 | Automatic shortcut scans (single-feature stumps, depth-3 tree) | Detects any remaining one-feature giveaway |
| 5 | Guards: processed-data fingerprint; training and SHAP refuse mismatched data | An earlier SHAP plot explained a model with the wrong inputs |
| 6 | GPU training with data held on the GPU; streaming, low-RAM preprocessing | Runs on the laptop (≈8 s/epoch vs ≈31 s; preprocessing peak 6.1 → 3.4 GB RAM) |
| 7 | 3 random seeds for every comparison; thresholds chosen on validation only | Results must be stable and the test set untouched |

---

## 4. Results on Edge-IIoTset (strict features, no MQTT: final setting)

**Binary (Normal vs Attack)**, test set, `results/experiment_registry.csv`:

| Model | Macro-F1 | Accuracy | False alarms | Attacks detected | ROC-AUC |
|---|---|---|---|---|---|
| Original pipeline (leaky) | 1.000 | 100% | 0% | 100% | - |
| **CNN-GRU (tuned, ours)** | **0.858** | **90.0%** | **0.7%** | **65.3%** | 0.904 |
| XGBoost (GPU) | 0.868 | 90.7% | 0.6% | 67.3% | 0.917 |

Figure: `results/figures/fig1_fake_vs_honest.png`

**Robustness checks**

* Without MQTT vs with MQTT: 0.8575 vs 0.8579, so the MQTT shortcut is not used.
* 3 seeds (tuned CNN-GRU): 0.8584 / 0.8583 / 0.8582.
* Optuna, 20 trials: best validation 0.8596; test 0.858, same as untuned. Tuning cannot help.

**Why the score stops at ~0.86: a measured ceiling** (`results/audit/detection_ceiling.json`,
`python audit_ceiling.py`)

* The 1,553,440 training packets contain only **11,122 distinct feature patterns**.
* **31.9% of attack packets** have exactly the same features as patterns that are mostly Normal.
* Even a perfect "memorise every pattern" classifier detects only **67.5%** of attacks (at 0.5%
  false alarms). Our models detect 65.3% (CNN-GRU) and 67.3% (XGBoost): **they are at the ceiling**.

Figure: `results/figures/fig5_detection_ceiling.png`

**Detection vs false-alarm trade-off** (`results/operating_points.csv`, thresholds chosen on
validation): on Edge-IIoTset, lowering the threshold raises detection to 73.3% at 6.6% false
alarms (CNN-GRU), because the remaining attacks are indistinguishable from Normal packets.

**Is the hybrid needed? Ablation**, same training settings, 3 seeds each:

| Model | Macro-F1 (mean) | Parameters | File size |
|---|---|---|---|
| MLP | 0.858 | 5,441 | 24 KB |
| 1D-CNN | 0.858 | 25,857 | 106 KB |
| GRU | 0.858 | 24,577 | 99 KB |
| CNN-GRU (tuned) | 0.858 | 79,169 | 315 KB |
| XGBoost | 0.868 | 226 trees | 492 KB |

Honest finding: on these per-packet features **the hybrid gives no accuracy gain**; all networks hit
the same ceiling. Figure: `results/figures/fig3_model_comparison.png`

**Explainability, and proof that it is faithful** (`results/shap_fidelity.json`)

* SHAP top features (final CNN-GRU): `tcp.flags`, `tcp.connection.rst`, `tcp.flags.ack`,
  `tcp.len`, `tcp.connection.fin`: real TCP behaviour.
* Retrain without the **top-5 SHAP features**: Macro-F1 **0.858 → 0.605** (−0.253).
* Retrain without **5 random features**: 0.858 → 0.858 (−0.0001).
* So SHAP identifies what the model really depends on. Figure: `results/figures/fig2_shap_fidelity.png`

**Edge deployment** (1 CPU thread, `results/edge_benchmark.csv`)

| Version | Macro-F1 | Size | Latency per packet | Throughput |
|---|---|---|---|---|
| CNN-GRU FP32 | 0.858 | 315 KB | 6.4 ms | 1,500 rows/s |
| **CNN-GRU FP16 weights** | **0.858** | **159 KB** | 6.5 ms | 1,660 rows/s |
| CNN-GRU INT8 (Linear only) | 0.858 | 304 KB | 8.1 ms | 1,620 rows/s |
| CNN-GRU INT8 (GRU + Linear) | 0.782 | 88 KB | 13.3 ms | 2,050 rows/s |
| XGBoost | 0.868 | 492 KB | 0.8 ms | 30,700 rows/s |

FP16 halves the size with no loss; INT8 on the GRU costs 7.6 points. XGBoost is faster on a CPU;
the networks are smaller (MLP 24 KB).

**Attack-type classification (14 classes + Normal)** on Edge-IIoTset strict features is weak:
XGBoost 0.429 Macro-F1, CNN-GRU 0.337 ± 0.033 (best loss). Loss study over 3 seeds: sqrt class
weights 0.337 > focal 0.254 > cross-entropy 0.238. Floods and scans (DDoS_UDP, Port_Scanning) are
defined by packet *rate*, which a single packet cannot show, and the dataset has no timing for
attack packets (`udp.time_delta` is 0 for every attack row).

---

## 5. Second dataset: CICIoT2023 (flow-level, with rate and timing)

Prepared with the same honest pipeline (`run_ciciot.py --policy strict`):

* Kaggle `dhoogla/ciciotdataset2023` (46.8M rows, kept outside OneDrive); capped sample of 1.41M
  rows (≤ 40k per attack label, 400k Benign), **87,417 duplicates removed** before splitting.
* `source_file` removed (it names the capture, i.e. the label).
* **Second hidden shortcut found and removed:** `packet_count` is the feature-extraction window
  size: always 10 for Benign/Recon and 100 for DDoS/DoS/Mirai; `total_sum = packet_count ×
  mean_packet_size`. Both dropped; flag counts turned into fractions of the window.
* Shortcut scan after the fix: best single feature 0.85 (binary), depth-3 tree 0.85; no giveaway.
* No attack row shares its exact features with a mostly-Benign pattern (ceiling not limiting).

| Task | Model | Macro-F1 | Notes |
|---|---|---|---|
| Benign vs Attack | XGBoost | **0.917** | detects 93.4% of attacks, 8.4% false alarms; ROC-AUC 0.979 |
| Benign vs Attack | CNN-GRU | **0.890** | detects 90.4%, 9.6% false alarms (validation-tuned threshold); ROC-AUC 0.966 |
| 8 categories | XGBoost | **0.752** | |
| 8 categories | CNN-GRU, sqrt weights | **0.675** | cross-entropy: 0.620 |

* Removing the window artifacts barely changed the scores (binary 0.9166 → 0.9165), so they hold.
* At ≤ 1% false alarms (threshold on validation): XGBoost detects 86.6%, CNN-GRU 82.3%.
* SHAP (CNN-GRU): `header_length`, `ack_count_frac`, **`rate`**, `https`, `arp`: timing/rate
  information now matters, as expected.
* Edge-IIoTset vs CICIoT2023 (CNN-GRU): binary 0.86 → 0.89, categories 0.34 → 0.68.
  Figure: `results/figures/fig4_datasets.png`

---

## 6. Contribution

1. **Found and removed hidden shortcuts in two public IoT datasets** (the `"0"`/`"0.0"` artifact in
   Edge-IIoTset; the window-size artifact in CICIoT2023) that make models look perfect.
2. **Measured the detection ceiling** of packet-level Edge-IIoTset and showed our model reaches it.
3. **Proved explanation fidelity** (SHAP removal test), not just SHAP plots.
4. **Honest, reproducible evaluation**: 3 seeds, fair ablation against simpler models, validation-
   only thresholds, tested edge deployment, GPU pipeline that runs on a laptop.

---

## 7. Limitations (state them before faculty do)

* On per-packet features, the CNN-GRU is not better than an MLP or XGBoost.
* About a third of Edge-IIoTset attack packets are indistinguishable from normal ones.
* The GRU sees no real time dimension (each row is processed alone).
* Attack-type detection needs flow/timing data; rare classes (Web, BruteForce) remain weak.
* Single train/test split per dataset (random, stratified); no cross-dataset test yet.

## 8. Next steps (to fully meet the problem statement)

| Gap | Plan |
|---|---|
| Missed attacks on packet data | Use flow-level data (CICIoT2023) as the main setting; choose thresholds by acceptable false-alarm rate |
| Hybrid gives no gain | Give the GRU real sequences: consecutive flow windows per device/connection, so it can learn how traffic changes over time |
| Weak rare classes | Sqrt class weights (already +0.06), targeted oversampling of Web/BruteForce |
| Generalisation | Cross-dataset test (train on one dataset, test on another with shared features) |
| Deployment | FP16 model on a Raspberry-Pi-class device; live demo via the dashboard predict API |

---

## 9. How to reproduce

```bash
python audit_shortcuts.py --raw                                   # artifact evidence
python run_clean_baselines.py --policy strict_no_mqtt --stage binary
python run_tune_cnn_gru.py --trials 20 --prefix F                  # tuning + 3 seeds
python run_final_studies.py                                        # ablation, SHAP, fidelity, edge
python run_operating_points.py && python audit_ceiling.py
python run_ciciot.py --policy strict                               # CICIoT2023
python make_figures.py                                             # results/figures/*.png
python watch_dashboard.py                                          # live progress (other window)
```
Long runs: `python run_queue.py "<script> <args>" ...` keeps Windows awake; keep the lid open.
