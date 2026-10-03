# IoT-IDS Project Update: What We Did, Why, Results, Next Steps
*(Presentation status update, October 2026. All numbers from [FINAL_RESULTS.md](FINAL_RESULTS.md).)*

## 1. Goal

An IoT intrusion detection system that is **accurate**, **explainable** (SHAP) and **lightweight**
enough for an IoT gateway. Proposed model: hybrid **CNN-GRU**. Datasets: **Edge-IIoTset** (2.2M
packets, Normal + 14 attacks) and **CICIoT2023** (flow-level, 33 attacks in 7 categories + Benign).

## 2. What we found: the first 100% accuracy was fake

* All first models scored 100% on Normal vs Attack. We investigated instead of celebrating.
* **Cause:** Edge-IIoTset writes an empty field as `"0"` in Normal captures and `"0.0"` in attack
  captures (e.g. `dns.qry.name.len`: 1,613,798 Normal rows use `"0"`, 603,331 attack rows use
  `"0.0"`, no overlap). After one-hot encoding, one column gives away the label.
* Our earlier "fix" (removing MQTT) did not help: the DNS column carries the same artifact, and the
  old SHAP plots hid it by stripping the `_0` / `_0.0` suffix.
* A second hidden shortcut in CICIoT2023: `packet_count` is the feature-extraction window size (10
  for Benign/Recon, 100 for DDoS/DoS/Mirai).

## 3. What we did, and why

| What | Why |
|---|---|
| Canonicalised `"0"` = `"0.0"` before encoding | Removes the formatting shortcut |
| Removed 7 per-packet identifiers (`tcp.seq`, `tcp.ack`, `tcp.ack_raw`, checksums, `icmp.seq_le`, `udp.stream`) | They identify the capture file, not attack behaviour |
| Removed all `mqtt.*` as a check | Every MQTT packet in the dataset is Normal |
| Automatic shortcut scans; data fingerprint checked by training and SHAP | Catch any remaining one-feature giveaway or mismatched inputs |
| 3 seeds per comparison; thresholds chosen on validation only | Stable results, untouched test set |
| GPU training, low-RAM streaming preprocessing | Runs on the laptop (≈8 s/epoch; preprocessing peak 6.1 → 3.4 GB) |

## 4. Results

**Edge-IIoTset, Normal vs Attack (strict features, no MQTT)**

| Model | Macro-F1 | Accuracy | False alarms | Attacks detected |
|---|---|---|---|---|
| Original pipeline (leaky) | 1.000 | 100% | 0% | 100% |
| **CNN-GRU (tuned, ours)** | **0.858** | **90.0%** | **0.7%** | **65.3%** |
| XGBoost | 0.868 | 90.7% | 0.6% | 67.3% |

* Stable: 3 seeds 0.8584 / 0.8583 / 0.8582; with vs without MQTT 0.8579 vs 0.8575; tuning (20
  Optuna trials) does not improve the test score.
* **Measured ceiling:** 31.9% of attack packets look exactly like mostly-Normal packets; even a
  perfect memoriser detects only 67.5%. Both models are at that ceiling.
* **Ablation:** MLP, 1D-CNN, GRU and CNN-GRU all reach 0.858. The hybrid gives no gain on per-packet
  features.
* **SHAP is faithful:** top features are `tcp.flags`, `tcp.connection.rst`, `tcp.flags.ack`,
  `tcp.len`, `tcp.connection.fin`. Removing the top-5 drops Macro-F1 0.858 → 0.605; removing 5
  random features changes nothing.
* **Edge:** FP16 CNN-GRU is 159 KB with no loss (0.858), 6.5 ms per packet on one CPU thread. INT8 on
  the GRU costs 7.6 points. XGBoost is faster on CPU (0.8 ms).
* **Attack type** (15 classes) is weak on packet data: XGBoost 0.429, CNN-GRU 0.337 (sqrt class
  weights beat focal 0.254 and cross-entropy 0.238 over 3 seeds).

**CICIoT2023 (flow-level, has rate and timing)**

| Task | XGBoost | CNN-GRU |
|---|---|---|
| Benign vs Attack (Macro-F1) | 0.917 | 0.890 |
| 8 categories (Macro-F1) | 0.752 | 0.675 (sqrt weights) |

CNN-GRU improves from Edge-IIoTset to CICIoT2023: binary 0.86 → 0.89, categories 0.34 → 0.68. SHAP
now ranks `rate` among the top features.

## 5. Contribution

1. Found and removed hidden shortcuts in two public IoT datasets.
2. Measured the detection ceiling of packet-level Edge-IIoTset; our model reaches it.
3. Proved SHAP fidelity with a removal test.
4. Honest, reproducible evaluation: 3 seeds, fair ablation, validation-only thresholds, tested edge
   deployment.

## 6. Next steps

| Gap | Plan |
|---|---|
| Missed attacks on packet data | Flow-level data (CICIoT2023) as the main setting; thresholds by acceptable false-alarm rate |
| Hybrid gives no gain | Real sequences for the GRU: consecutive flow windows per device/connection |
| Weak rare classes | Sqrt class weights, targeted oversampling of Web/BruteForce |
| Generalisation | Cross-dataset test |
| Deployment | FP16 model on a Raspberry-Pi-class device; live demo via the dashboard predict API |
