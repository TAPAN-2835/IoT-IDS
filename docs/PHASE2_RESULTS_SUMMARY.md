> **Correction (2026-10-02):** The focal-loss result compares 5-epoch runs and the E02 no_mqtt runs were 1-epoch smoke tests; see the corrected numbers. See [LEAKAGE_FIX_AND_CLEAN_BASELINES.md](LEAKAGE_FIX_AND_CLEAN_BASELINES.md).

# PHASE 2 RESULTS SUMMARY

## 1. Experiments Completed
- **E01 (Leakage Audit):** Analyzed a representative subset of the dataset and confirmed severe leakage due to protocol associations.
- **E05 (MQTT Ablation):** Trained Random Forest and CNN-GRU models without MQTT protocol features to measure dependency.
- **E03 (Class Imbalance):** Evaluated baseline Cross-Entropy, Class-Weighted, and Focal Loss to address minority-class performance in the multiclass target.
- **Experiment Framework:** Upgraded `train.py` and `training.py` to securely log detailed configurations (including feature lists and timestamps) without overwriting past results.

## 2. Actual Measured Results
- **Baseline (Phase 1):** Binary Accuracy 1.0000, Macro-F1 1.0000. Multiclass Macro-F1 ~0.64.
- **E05 (No MQTT) RF:** Accuracy 1.0000, Macro-F1 1.0000
- **E05 (No MQTT) CNN-GRU:** Accuracy 1.0000, Macro-F1 1.0000
- **E03 (Class-Weighted) CNN-GRU:** Accuracy 0.8411, Macro-F1 0.3299
- **E03 (Focal Loss) CNN-GRU:** Accuracy 0.9163, Macro-F1 0.4084

## 3. Important Discoveries
- **Dataset Shortcut / Leakage Confirmed:** E01 proved that the 100% binary accuracy is an artifact. In the sampled dataset, **100% of rows containing MQTT features are labeled as Normal (0)** and **100% of rows containing specific HTTP features are labeled as Attack (1)**. The models act as simple protocol signature matchers.
- **Genuine Sequence Modeling is Not Currently Feasible:** An audit of `frame.time` and `ip.src_host` (over 500k rows) reveals the dataset is heavily synthesized. There are essentially only 2 dominant source IPs (`192.168.0.101` and `192.168.0.128`). Because the dataset stitches together disjoint PCAPs for different attacks, chronological splits or device grouping would essentially just split different attack classes into completely disjoint sets. True sequential learning is not viable; the current `seq_len=1` GRU is an architectural quirk rather than sequence modeling.

## 4. Unresolved Issues
- While E03 attempts to resolve the multiclass imbalance, fundamentally treating Edge-IIoTset as a tabular task remains overly optimistic.
- The CNN-GRU model requires hyperparameter tuning to fully maximize the Focal Loss improvements on multiclass data.

## 5. Answers to Key Research Questions
- **Does the 100% binary result survive the ablations?** Yes. E05 showed that removing MQTT features did not drop the 100% accuracy. The Random Forest simply pivoted to using DNS and HTTP missing-value indicators (like `dns.qry.name.len_0.0`). The leakage is systemic across all protocols.
- **Are MQTT features responsible for the shortcut?** They are one part of the shortcut, but the dataset lacks standardized background traffic, creating multiple parallel shortcuts across different protocols.
- **Multiclass minority-class performance:** Standard cross-entropy completely fails on minority classes (Macro-F1 0.26). Using **Focal Loss** massively improves this to 0.408 while boosting overall accuracy to 91.6%.
- **Is genuine sequence modeling possible?** No. As audited, the lack of sufficient device IP diversity and the disjoint nature of the stitched PCAPs means grouping by device/time would introduce massive leakage. The GRU approach remains exploratory.

## 6. Recommended Next Experiment (Phase 3)
- Execute **E02 (Hyperparameter Tuning)** using Focal Loss to squeeze out maximum multiclass performance.
- Proceed with **E08 & E09 (Model Compression & Edge Benchmarking)** to establish the lightweight viability of the CNN-GRU architecture for IoT edge deployment.
