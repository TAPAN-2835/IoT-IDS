> **Correction (2026-10-02):** The shortcut is real but the mechanism is the "0" vs "0.0" spelling of empty fields across 13 string columns, not MQTT presence. See [LEAKAGE_FIX_AND_CLEAN_BASELINES.md](LEAKAGE_FIX_AND_CLEAN_BASELINES.md).

# E01 - LEAKAGE AND SHORTCUT AUDIT

## 1. Objective
Investigate the suspicious 100% accuracy and 1.000 F1 score in the baseline binary classification (`Attack_label`) models to determine whether they are learning genuine intrusion patterns or merely exploiting dataset shortcuts (leakage).

## 2. Methodology
A random sample of 500,000 records (stratified representation) was analyzed to identify:
- Associations between protocol fields and class labels.
- Duplicate and near-duplicate records.
- Extreme sparsity differences between normal and attack traffic.

## 3. Findings

### Label Distribution in Sample
- **Normal (0):** 364,557
- **Attack (1):** 135,443

### Protocol Correlation (The Shortcut)
- **MQTT:** 18,739 rows contained active `mqtt.topic` features. **100% of these rows were labeled Normal (0).** 
- **HTTP:** 7,298 rows contained active `http.request.method` features. **100% of these rows were labeled Attack (1).**
- **TCP:** `tcp.dstport` was active in ~442,000 rows, distributed across both Normal and Attack.

### Duplicate Analysis
- **Exact duplicate rows:** 183 (0.04%). Duplication is minimal and not a primary source of leakage.

### High Cardinality Signatures
- String features like `tcp.options` (59,741 unique values) and `tcp.payload` (63,271 unique values) act as unique identifiers. While these were mostly dropped by the dynamic cardinality filter (>100 unique values limit), any remaining string fragments perfectly map to specific packet captures.

## 4. Conclusion & Classification

**CLASSIFICATION: Confirmed Dataset Shortcut / Severe Leakage**

The Edge-IIoTset contains protocol-specific shortcuts. The models are not actually detecting attacks; they are learning that the presence of MQTT implies "Normal" and the presence of certain HTTP traffic implies "Attack". Because the protocol fields (e.g., `mqtt.topic_len`, `mqtt.conflags`) are zeroed out for non-MQTT traffic, a tree-based model or CNN can achieve perfect binary classification simply by checking if the MQTT or HTTP feature blocks are non-zero. 

This completely explains the 100% binary accuracy observed in Phase 1 baselines.

**Next Action:** Proceed with **E05 (MQTT Ablation)** to quantify the models' dependency on these protocol features by retraining them with the MQTT features entirely removed.
