> **Correction (2026-10-02):** Result confirmed and explained: removing MQTT cannot help because dns.qry.name.len carries the same "0" vs "0.0" artefact. See [LEAKAGE_FIX_AND_CLEAN_BASELINES.md](LEAKAGE_FIX_AND_CLEAN_BASELINES.md).

# E05 - MQTT ABLATION

## 1. Objective
Determine the extent to which the 100% baseline accuracy relies on MQTT protocol features. The previous E01 audit identified a severe shortcut where MQTT presence perfectly predicts the 'Normal' class.

## 2. Methodology
- **Baseline:** Random Forest and CNN-GRU trained on the full `operational` feature set.
- **Ablation:** Retrain the same models using a restricted `no_mqtt` feature policy, dropping all 14 `mqtt.*` features.
- **Target:** Binary `Attack_label`.

## 3. Results

| Model | Features | Accuracy | Macro-F1 | Inference Latency |
| ----- | -------- | -------- | -------- | ----------------- |
| Random Forest | Full (Operational) | 1.0000 | 1.0000 | ~0.005 ms |
| Random Forest | No MQTT | 1.0000 | 1.0000 | ~0.005 ms |
| CNN-GRU | Full (Operational) | 1.0000 | 1.0000 | ~0.02 ms |
| CNN-GRU | No MQTT | 1.0000 | 1.0000 | ~0.02 ms |

## 4. Conclusion
**Finding:** The removal of MQTT features did **NOT** degrade the 100% binary classification accuracy for either the Random Forest or the CNN-GRU model. 
**Analysis:** The Random Forest simply pivoted to other protocol-specific missing value indicators. The top features became `dns.qry.name.len_0.0` (missing DNS queries) and `http.request.method_0.0` (missing HTTP requests). This confirms that the dataset leakage is systemic across multiple protocol layers, not just MQTT. The dataset lacks standardized background traffic, creating distinct protocol signatures for different classes.
