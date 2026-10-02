# Leakage Fix and Clean Baselines (C-series)

*2026-10-02. Supersedes the conclusions in E01_LEAKAGE_AUDIT, E03_CLASS_IMBALANCE, E05_MQTT_ABLATION
and PRESENTATION_UPDATE_SUMMARY where they disagree with this document.*

## 1. The actual cause of the 100% binary accuracy

The shortcut is not "MQTT presence". It is how the dataset's CSV export wrote **empty protocol
fields**: the Normal captures wrote them as `"0"`, the attack captures as `"0.0"` (reversed for the
HTTP fields). These columns are strings, so one-hot encoding turns the two spellings into two
different features, and each one alone identifies the class.

Full dataset (2,219,201 rows), from `python audit_shortcuts.py --raw` → `results/audit/empty_token_audit.csv`:

| Column | Normal: empty as `"0"` | Normal: empty as `"0.0"` | Attack: empty as `"0"` | Attack: empty as `"0.0"` |
|---|---|---|---|---|
| `dns.qry.name.len` | 1,613,798 | 0 | 0 | 603,331 |
| `mqtt.topic` | 1,532,627 | 0 | 0 | 603,558 |
| `mqtt.protoname` | 1,532,608 | 0 | 0 | 603,558 |
| `mqtt.conack.flags` | 1,532,586 | 0 | 0 | 603,558 |
| `http.request.method` | 0 | 1,615,643 | 222,839 | 348,632 |

* `dns.qry.name.len`, `mqtt.topic`, `mqtt.protoname`, `mqtt.conack.flags` (and `mqtt.msg`, dropped for
  cardinality) each separate Normal from Attack **perfectly on their own**.
* The HTTP columns use `"0"` only for the web attacks (DDoS_HTTP, SQL_injection, XSS, Password,
  Uploading, Vulnerability_scanner), so they also leak the **attack type** in the multiclass task.
* This explains why the E05 MQTT ablation changed nothing: `dns.qry.name.len` carries the same signal.
  The RF's top features after removing MQTT were exactly `dns.qry.name.len_0.0` and `dns.qry.name.len_0`.
* SHAP plots hid this because `_clean_name` stripped the `_0` / `_0.0` suffix, so both appeared as
  "mqtt.topic".

## 2. Fix

`src/config.py: CANONICALIZE_NUMERIC_TOKENS = True` (default). Before encoding, numeric-looking
strings in categorical columns are canonicalised (`"0"` and `"0.0"` → `"0.0"`, `"1461073"` and
`"1461073.0"` → `"1461073.0"`); real text values such as `MQTT` or `0x00000000` are untouched.
Setting it to `False` reproduces the old pipeline exactly. Result: 83 features instead of 90
(seven duplicate "empty" columns merged).

**Shortcut check after the fix** (`audit_shortcuts.py --processed`, binary target):

| Check | Before fix | After fix |
|---|---|---|
| Best single feature (decision stump, balanced accuracy) | 1.00 | **0.72** (`tcp.seq`) |
| Depth-3 decision tree on all features | 1.00 | **0.91** |

## 3. Clean binary results (`Attack_label`, operational policy, same 70/15/15 split, seed 42)

| ID | Model | Accuracy | Macro-F1 | FPR | FNR (missed attacks) | Train time |
|---|---|---|---|---|---|---|
| E01 / E05 (old) | RF / CNN-GRU | 1.0000 | 1.0000 | 0 | 0 | — |
| **C01_xgb_binary_clean** | XGBoost (GPU) | **0.9694** | **0.9599** | 0.0003 | 0.112 | ~20 s |
| **C02_cnn_gru_binary_clean** | CNN-GRU (GPU, 15 epochs) | 0.9147 | 0.8795 | 0.0007 | 0.312 | ~2 min |

* Binary detection is no longer trivial, and the gap between models is now visible: the tree
  baseline beats the proposed CNN-GRU by ~8 Macro-F1 points.
* Both models are very precise (almost no false alarms) but **miss attacks**: CNN-GRU misses 31%.
  That is the number to improve, and the one to discuss honestly.
* **SHAP (C02)** now ranks behavioural TCP features first: `tcp.flags`, `tcp.flags.ack`,
  `icmp.seq_le`, `tcp.connection.syn`, followed by `tcp.seq`, `tcp.ack_raw`, `tcp.checksum`.
  XGBoost's gain ranking agrees (`tcp.flags`, `tcp.connection.syn`, `tcp.connection.rst`, `tcp.ack`).

## 4. Clean multiclass results (`Attack_type`, 15 classes) — E03 redone at equal epochs

Shortcut check after the fix: best single feature 0.13 balanced accuracy, depth-3 tree 0.32.

| ID | Model / loss | Accuracy | Macro-F1 | Weighted-F1 | Model size |
|---|---|---|---|---|---|
| E02_random_forest_multiclass (old, leaky) | RandomForest | 0.9825 | 0.8813 | 0.9829 | 1,589 MB |
| E08_cnn_gru_multiclass (old, leaky, 15 ep) | CNN-GRU, CE | 0.9474 | 0.6445 | 0.9398 | 0.09 MB |
| **C03_xgb_multiclass_clean** | XGBoost (GPU) | **0.9633** | **0.8233** | 0.9603 | 8.2 MB |
| **C04_cnn_gru_mc_ce_clean** | CNN-GRU, cross-entropy | 0.8929 | 0.4225 | 0.8572 | 0.09 MB |
| **C05_cnn_gru_mc_weighted_clean** | CNN-GRU, class-weighted CE | 0.0977 | 0.0156 | 0.1373 | 0.09 MB |
| **C06_cnn_gru_mc_focal_clean** | CNN-GRU, focal (γ=2) | 0.8834 | 0.3622 | 0.8408 | 0.09 MB |

All CNN-GRU runs: 15 epochs, same split and seed. (Training times are not comparable: C05/C06 ran on
battery power with a throttled GPU.)

Per-class F1 (test set):

| Class | Support | XGBoost (C03) | CNN-GRU CE (C04) | CNN-GRU focal (C06) |
|---|---|---|---|---|
| Normal | 242,347 | 0.980 | 0.943 | 0.940 |
| DDoS_UDP / DDoS_ICMP | 18,235 / 17,466 | 1.000 / 0.998 | 0.996 / 0.996 | 0.989 / 0.997 |
| DDoS_TCP | 7,510 | 1.000 | 0.867 | 0.808 |
| Vulnerability_scanner | 7,517 | 0.967 | 0.794 | 0.763 |
| Backdoor | 3,729 | 0.949 | 0.461 | 0.482 |
| Port_Scanning | 3,384 | 0.866 | 0.595 | 0.000 |
| DDoS_HTTP | 7,486 | 0.734 | 0.357 | 0.135 |
| Password / SQL_injection / Uploading / XSS | 7.5k / 7.7k / 5.6k / 2.4k | 0.81 / 0.79 / 0.75 / 0.80 | 0.09 / 0.15 / 0.10 / 0.00 | 0.00 / 0.09 / 0.24 / 0.00 |
| Ransomware | 1,639 | 0.885 | 0.000 | 0.000 |
| MITM / Fingerprinting | 182 / 150 | 0.46 / 0.36 | 0.00 / 0.00 | 0.00 / 0.00 |

Findings:

* **The E03 conclusion reverses.** At equal epochs on clean data, focal loss is *worse* than plain
  cross-entropy (0.362 vs 0.423 Macro-F1). Single seed, so treat as "no evidence focal helps", not
  as proof it hurts.
* **Class-weighted CE collapsed**: the loss stayed at ≈ ln(15) and the model never learned. The raw
  inverse-frequency weights span roughly 0.09 (Normal) to 150 (Fingerprinting). Softened weights
  (square-root inverse frequency, or clipping) are needed before this can be compared.
* **The leaky E08 score (0.644) was inflated** by the HTTP `"0"`/`"0.0"` tokens, which marked the
  web-attack classes. With them removed, the CNN-GRU cannot tell web attacks apart (F1 ≤ 0.24 for
  Password, SQL injection, Uploading, XSS), while XGBoost still reaches 0.75–0.81 on them.
* **The proposed CNN-GRU is far behind a tree baseline on this tabular data** (0.42 vs 0.82
  Macro-F1). Its advantage is size (0.09 MB vs 8.2 MB). The honest framing is a size/accuracy
  trade-off, and the CNN-GRU needs tuning before it is competitive.

## 5. Corrections to earlier claims

| Earlier claim | Status |
|---|---|
| "MQTT presence = Normal is the shortcut; removing MQTT forces generalisation" | Wrong mechanism. The shortcut is the `"0"`/`"0.0"` spelling across 13 string columns; removing MQTT leaves it in place (E05 itself showed 100% after removal). |
| "Focal loss raises Macro-F1 from 0.263 to 0.408 and should be adopted" (E03) | Reversed: those runs used `EPOCHS = 5`. At 15 epochs on clean data, focal (0.362) is below plain cross-entropy (0.423); see section 4. |
| E02 `no_mqtt` baselines (Macro-F1 0.10 / 0.24 / 0.12) | Smoke-test runs (1 epoch, `"epochs": 1` in their records). Not usable as baselines. |
| SHAP for `E05_cnn_gru_no_mqtt` showing `mqtt.topic` on top | Invalid: a 62-feature model was run on 91-feature data. CNN-GRU weights do not depend on input width, so it loaded without error. Now prevented (see 6). |
| `results/experiments/E02_hyperparameter_tuning/best_params.json` | Stale: 4 parameters from an older script (current search space has 8), no trial history saved. Re-run the tuning on clean data. |
| "Immune to dataset shortcuts", "leakage-free" (presentation summary, mentor guide) | Should not be presented. Remaining suspects are listed in 7. |

## 6. Pipeline changes (what to know as a contributor)

* **`data/processed/metadata.json`** records the target, feature policy, canonicalisation flag,
  feature list and a fingerprint. Training refuses to run on data built for a different target,
  and every experiment record stores the fingerprint under `"preprocessing"`.
* **SHAP refuses mismatched data**: `run_shap_analysis` compares the experiment's fingerprint (or
  feature list) with `data/processed/` and stops instead of explaining the wrong inputs. Older
  E03/E04/E05 models predate feature tracking and are refused. The `models/cnn_gru_final.pt`
  fallback was removed (it is overwritten by every CNN-GRU run). SHAP labels keep the one-hot
  category (`mqtt.topic=0.0`).
* **GPU-first, laptop-friendly**: data is moved to the GPU once and batched there (about 8 s/epoch
  vs about 31 s before on the RTX 4060). The tree baseline is GPU XGBoost instead of the
  200-tree RandomForest, which used all CPU cores, several GB of RAM and a 1.6 GB model file.
  Preprocessing streams the CSV, never loads payload columns, writes float32, and runs in a child
  process so its RAM is returned to the OS. The runner lowers its own process priority and asks
  Windows not to sleep while it runs (the 3–5 minute idle-sleep timer froze one run overnight).
  Measured: preprocessing peak 6.05 GB → 3.42 GB (released on exit); training process peak
  2.54 GB; GPU peak 3.7 GB of 8 GB. Plug the laptop in for long runs: on battery the GPU is
  throttled (≈18 s/epoch instead of ≈8 s).
* `EPOCHS` is back to 15; `train_dl_model(..., epochs=N)` overrides it per run.
* New experiment IDs use the `C` prefix to avoid the E02/E03/E05 name collisions.
* Added `torch`, `optuna`, `xgboost`, `psutil` to `requirements.txt`.

### How to reproduce

```bash
python audit_shortcuts.py --raw                       # empty-token evidence table
python run_clean_baselines.py --stage binary          # preprocess + scan + C01, C02
python run_explainability.py                          # SHAP for C02 (run before the next stage)
python run_clean_baselines.py --stage multiclass      # preprocess + scan + C03-C06
python run_E02_hyperparameter_tuning.py               # Optuna on the clean multiclass data (GPU)
```

## 7. What is still open

1. **Capture-identifier features.** After the fix, the strongest single features are raw per-packet
   values: `tcp.seq`, `tcp.ack_raw`, `tcp.checksum`, `udp.stream` (a stream index). These identify
   a capture more than a behaviour. Next experiment: a stricter feature policy without them, and
   compare.
2. **Missed attacks.** Binary FNR is 11% (XGBoost) and 31% (CNN-GRU). In multiclass, the CNN-GRU
   fails on the web attacks, Ransomware, MITM and Fingerprinting (section 4). Consider a decision
   threshold tuned on validation for binary.
3. **Class-weighted loss** needs softened weights (sqrt inverse frequency or clipped) and a re-run.
4. **Multiple seeds** for every comparison before claiming one loss or model is better.
5. **Optuna tuning** of the CNN-GRU on the clean multiclass data (script updated for the GPU, not
   yet run; plug the laptop in, it runs about 20 trials).
6. **Compression and edge benchmark (E08/E09)** after the model is final.
7. **Docs and slides**: update the presentation summary, mentor guide and study guide numbers.
