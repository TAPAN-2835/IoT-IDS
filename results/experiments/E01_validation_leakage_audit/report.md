# E01 - Leakage and Shortcut Audit Report

## 1. Label Distribution

- **Total samples (subset):** 500000
- **Normal (0):** 364557
- **Attack (1):** 135443

## 2. Protocol & Feature Correlation with Label

### MQTT Traffic Analysis
Rows with active `mqtt.topic`: 18739
Distribution of `Attack_label` when `mqtt.topic` is active:
- Label 0: 18739

**WARNING:** MQTT traffic is perfectly correlated with a single class!

### HTTP Traffic Analysis
Rows with active `http.request.method`: 7298
Distribution of `Attack_label` when `http.request.method` is active:
- Label 1: 7298

### TCP Traffic Analysis
Rows with active `tcp.dstport`: 442186
Distribution of `Attack_label` when `tcp.dstport` is active:
- Label 0: 362003
- Label 1: 80183

## 3. Duplicate Records Analysis

- **Exact duplicate rows:** 183 (0.04%)

## 4. High Cardinality Categorical Features

Features with >100 unique values that might be acting as signatures:
- `frame.time`: 497104 unique values
- `ip.src_host`: 31003 unique values
- `ip.dst_host`: 11756 unique values
- `http.file_data`: 584 unique values
- `http.request.uri.query`: 1872 unique values
- `http.request.full_uri`: 4476 unique values
- `tcp.options`: 59741 unique values
- `tcp.payload`: 63271 unique values
- `tcp.srcport`: 40862 unique values
- `mqtt.msg`: 132 unique values

## 5. Missing Values (Sparsity) as a Leakage Source

Differences in missing value rates between Normal (0) and Attack (1) can be a trivial shortcut.

## 6. Conclusion and Classifications


Based on the analysis above:
- **Probable Dataset Shortcut:** If specific protocols (like MQTT or HTTP) are only present in one class, the model learns the protocol presence, not an intrusion pattern.
- **Confirmed Leakage:** The extremely high binary accuracy is almost certainly due to these protocol-specific sparsity patterns. A tree-based model or CNN can simply check if an MQTT feature is non-zero to predict the attack label perfectly.
