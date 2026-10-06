# Task 1: Replicate the experiments and prove our progress

**Goal:** (A) re-check, from code and saved results, every claim we already make, and (B) run the new
experiments that test our work against the five reference papers. Every result must come out of a
script and land in a file, so the paper (Task 2) can use it without anyone typing a number.

Repository: https://github.com/TAPAN-2835/IoT-IDS · Read first: `docs/FINAL_RESULTS.md`,
`docs/LEAKAGE_FIX_AND_CLEAN_BASELINES.md`, and the friend's review `IoT_IDS_Deep_Dive_Review.docx`.

---

## 0. Ground rules (apply to every experiment)

| Rule | Detail |
|---|---|
| New code | One script per experiment in `experiments/` (e.g. `experiments/b1_ferrag_replication.py`). Reuse `src/` (preprocessing, training, explainability); do not change existing behaviour in `src/` without a unit test |
| Outputs | `results/paper/<name>.json` (numbers) and `results/paper/<name>.csv` (tables); figures in `paper/figures/` |
| Seeds | 42, 7 and 2024 for anything trained; report mean ± standard deviation |
| Test set | Used once, for the final number. Thresholds, tuning and feature choices use the **validation** split only |
| Data check | Every model must be evaluated on the processed data it was trained on (compare the `fingerprint` in `data/processed/metadata.json` with the experiment record) |
| Long runs | `python run_queue.py "<script> <args>" ...` keeps Windows awake; keep the laptop plugged in and the lid open; watch with `python watch_dashboard.py` |
| Honesty | Report the result whatever it is. If an experiment does not support a claim, the claim changes, not the experiment |

Final setting used throughout (unless an experiment says otherwise): Edge-IIoTset, binary
(`Attack_label`), feature policy `strict_no_mqtt` (51 encoded features), final model
`results/experiments/F_strict_no_mqtt_binary_best_s42` (tuned CNN-GRU), XGBoost baseline
`results/experiments/N01_xgb_binary_strict_nomqtt`. Recreate the processed data with
`python run_clean_baselines.py --policy strict_no_mqtt --stage binary` if it is missing.

---

## Part A: Prove our current claims

Write `experiments/verify_claims.py`. For each claim below it recomputes the value (from raw data or
saved results), compares it with the expected value, and writes PASS/FAIL to
`results/paper/claims_check.json` and a readable table to `results/paper/claims_check.md`.

| # | Claim | How to verify | Expected |
|---|---|---|---|
| A1 | Empty fields are written `"0"` in Normal and `"0.0"` in attack rows | Stream the raw CSV (`audit_shortcuts.py --raw`); check `dns.qry.name.len`, `mqtt.topic`, `mqtt.protoname`, `mqtt.conack.flags` | `dns.qry.name.len`: Normal "0" = 1,613,798, attack "0.0" = 603,331, zero overlap |
| A2 | No single feature gives the answer away after the fix | Read the shortcut scan for the `strict_no_mqtt` binary data (`results/audit/shortcut_scan_Attack_label_*`) | Best single-feature balanced accuracy 0.70 (`tcp.flags`); depth-3 tree 0.73 |
| A3 | Per-packet ID fields removed | Compare `results/audit/strict_no_mqtt_feature_policy.csv` with the operational policy | `tcp.seq`, `tcp.ack`, `tcp.ack_raw`, `tcp.checksum`, `icmp.checksum`, `icmp.seq_le`, `udp.stream` and all `mqtt.*` absent |
| A4 | Honest final scores | Re-evaluate the saved final CNN-GRU and XGBoost on the test split | CNN-GRU Macro-F1 0.858, accuracy 90.0%, false alarms 0.7%; XGBoost 0.868 |
| A5 | Stable over seeds | Read the three `F_strict_no_mqtt_binary_best_s*` records | 0.8584 / 0.8583 / 0.8582 |
| A6 | Detection ceiling | `python audit_ceiling.py` | 11,122 distinct patterns; 31.9% of attacks match mostly-Normal patterns; ceiling 67.5% detected |
| A7 | The invisible attacks are label noise from web/application attacks | Build `strict_no_mqtt` with target `Attack_type` into a scratch folder (do not overwrite `data/processed`); for each test attack row whose exact feature pattern is mostly Normal in training, count its attack type | ~94% from web/application attacks (SQL injection 83% of its rows, Password 82%, XSS 79%, Uploading 70%, DDoS_HTTP 68%); 0 from DDoS UDP/TCP/ICMP; most common hidden pattern = TCP packet with empty HTTP fields that also appears ~99k times in Normal test traffic |
| A8 | All networks tie (ablation) | Read `A_mlp_s*`, `A_cnn1d_s*`, `A_gru_s*`, `F_*_best_s*` | All 0.858 (mean of 3 seeds); MLP 24 KB |
| A9 | SHAP fidelity (retraining) | Read `results/shap_fidelity.json` | Top-5 SHAP removed: 0.858 → 0.605; 5 random removed: 0.858 |
| A10 | Edge benchmark | Re-run `python run_edge_benchmark.py --model F_strict_no_mqtt_binary_best_s42 --xgb N01_xgb_binary_strict_nomqtt` | FP32 315 KB ~6.4 ms; FP16 159 KB same F1; INT8 GRU+Linear F1 0.782; XGBoost ~0.8 ms |
| A11 | Loss study | Read `L_strict_<loss>_s*` | sqrt weights 0.337 ± 0.033 > focal 0.254 > cross-entropy 0.238 |
| A12 | CICIoT2023 | Read `CS01`–`CS05` records; check `pipeline.log` for the duplicate count | Binary XGBoost 0.917, CNN-GRU 0.890; 8 categories 0.752 / 0.675; 87,417 duplicates removed; `packet_count` = 10 for Benign/Recon, 100 for DDoS/DoS/Mirai |
| A13 | Unit tests pass | `python -m pytest tests` | All pass |

Tolerance: ±0.002 for scores, exact for counts. A FAIL is reported, investigated and explained, never hidden.

---

## Part B: New experiments against the reference papers

Each experiment lists **why** (which paper it tests), **how**, and **what to save**.

### B1. Replicate the Edge-IIoTset authors' pipeline (tests Ferrag et al. 2022)
- **Why:** the dataset paper reports 99.99% binary accuracy. We want to show the number depends on the `"0"`/`"0.0"` shortcut.
- **How:**
  1. Load the raw CSV. Drop their 15 columns: `frame.time, ip.src_host, ip.dst_host, arp.src.proto_ipv4, arp.dst.proto_ipv4, http.file_data, http.request.full_uri, icmp.transmit_timestamp, http.request.uri.query, tcp.options, tcp.payload, tcp.srcport, tcp.dstport, udp.port, mqtt.msg`.
  2. Remove duplicates and NaN/INF rows (their order: drop columns first, then duplicates).
  3. One-hot encode the remaining text columns with `pandas.get_dummies` **without** canonicalising; StandardScaler; random stratified 80/20 split (their ratio is not stated; record our choice).
  4. Train XGBoost (GPU) and a small DNN; binary and 15-class.
  5. Run the single-feature stump scan; record the top features by XGBoost gain.
  6. Repeat with only our fix (canonicalise `"0"`/`"0.0"`), then with our full strict policy.
- **Save:** `results/paper/ferrag_replication.json` with, per setting (`reported`, `replicated`, `canonicalised`, `strict`): accuracy, Macro-F1, best single feature and its balanced accuracy, top-10 features.
- **Expected outcome to test:** replicated ≈ 99.99%, a `*_0` / `*_0.0` column at the top, and a clear drop after the fix.

### B2. Zeroing vs retraining as a fidelity test (tests Udurume et al. 2026)
- **Why:** they test SHAP by setting top features to zero at prediction time, without retraining and without a random control. Zeroed inputs are out of distribution, so the drop may be exaggerated.
- **How:** on the final CNN-GRU, (a) set the top-5 SHAP features to 0 in the model's input space (= training mean after StandardScaler) and evaluate; (b) same with the training minimum (their MinMax equivalent); (c) same with 5 random features (20 draws). Compare with our retraining result (A9).
- **Save:** `results/paper/fidelity_zero_vs_retrain.json`: drops for zero-mean, zero-min, retrain; random-feature mean ± SD.

### B3. "All raw features" setup (tests Munilla & Khammas 2026)
- **Why:** they explain Edge-IIoT models trained on all 63 columns, and their top SHAP features include MQTT header data (our shortcut).
- **How:** train XGBoost on all features (except the two labels), text columns label-encoded, 80/20 split, no balancing. Compute exact TreeSHAP top-10. Repeat on our cleaned `strict_no_mqtt` data.
- **Save:** `results/paper/munilla_style.json`: Macro-F1 for both, top-10 features for both, Jaccard overlap of the two top-10 lists, and how many leaky-model top features are shortcut columns (empty-field indicators, MQTT, IDs, ports, IPs).

### B4. Shortcut audit of UNSW-NB15 (tests Sharma 2025, Wang 2025, Udurume 2026)
- **Why:** three papers use UNSW-NB15 and rank TTL / sequence fields (`sttl`, `ct_state_ttl`, `stcpb`) highly.
- **How:** download the official training and testing CSVs (kagglehub); code in `experiments/unsw/`. Stump scan per feature on the official split. XGBoost Macro-F1 with (1) all features, (2) without `sttl, dttl, ct_state_ttl`, (3) also without `stcpb, dtcpb`.
- **Save:** `results/paper/unsw_audit.json`: `best_stump_feature`, `best_stump_bal_acc`, `xgb_all_f1`, `xgb_no_ttl_f1`, `xgb_no_ttl_no_seq_f1`, `top_features`. Claim only what the numbers show.

### B5. Fidelity across explainers with many random controls (strengthens A9; no paper does this)
- **Why:** removing any 5 informative features will hurt; a fair test compares explainers.
- **How:** retrain the final CNN-GRU configuration without the top-5 features from (a) SHAP, (b) permutation importance (on validation), (c) XGBoost gain; plus 20 random 5-feature draws. Seed 42, 15 epochs, same settings as `run_final_studies.py`.
- **Save:** `results/paper/fidelity_methods.json` and a bar/strip chart in `paper/figures/fidelity_methods.png`.

### B6. SHAP stability (matches Udurume, Munilla)
- **How:** SHAP top-10 on the three seed models, and on 5 bootstrap resamples of the explained test packets (seed 42 model). Jaccard similarity of the top-10 lists (pairwise mean ± SD). Optionally compare the CNN-GRU SHAP ranking with exact TreeSHAP on XGBoost (Spearman rank correlation).
- **Save:** `results/paper/shap_stability.json`.

### B7. LIME vs SHAP (matches Wang, Sharma, Munilla)
- **How:** explain 200 test packets (100 attack, 100 normal) with LIME (`lime` package) and SHAP; top-5 overlap per packet, global top-10 agreement, time per explanation; run the B5-style retraining fidelity for LIME's global top-5.
- **Save:** `results/paper/lime_vs_shap.json`.

### B8. Per-class results as mean ± SD over seeds (matches Sharma's reporting)
- **How:** per-class precision/recall/F1 for the 3 seeds of the attack-type models (Edge-IIoTset loss study `L_strict_sqrt_weighted_s*`; CICIoT2023 category models; train the missing seeds if needed).
- **Save:** `results/paper/perclass_mean_sd.csv`.

### B9. Event-level detection (no paper reports it)
- **Why:** a flood has thousands of packets; catching 65% of packets can still catch every attack within milliseconds.
- **How:** first check whether the raw CSV keeps capture order (are rows of one attack contiguous, is `frame.time` increasing?). If yes, define events (contiguous runs of one attack type) and report: share of events detected (at least k flagged packets), time or packets until first detection, and false alerts per hour after windowed aggregation. If order is not preserved, document that and skip.
- **Save:** `results/paper/event_level.json`.

### B10. Accuracy vs speed vs size (honest edge comparison)
- **How:** same data and single CPU thread: logistic regression, depth-limited decision tree, LightGBM, XGBoost, MLP, CNN-GRU FP32/FP16 (optionally ONNX Runtime). Report Macro-F1, latency per packet (median and p99), throughput, file size.
- **Save:** `results/paper/pareto.csv` and `paper/figures/pareto.png`.

### B11. Adversarial robustness (matches Munilla, Sharma)
- **How:** FGSM (and optionally PGD) on CNN-GRU and MLP in the standardised feature space, epsilon 0 to 0.5; Macro-F1 vs epsilon. State clearly that perturbing one-hot or flag features is not physically realistic.
- **Save:** `results/paper/adversarial.json` and `paper/figures/adversarial.png`.

---

## Deliverables checklist

- [ ] `experiments/verify_claims.py` runs end to end; `results/paper/claims_check.md` shows all PASS (or explained FAILs)
- [ ] B1–B11 scripts in `experiments/`, each runnable alone
- [ ] All result files in `results/paper/` with the keys listed above
- [ ] New figures in `paper/figures/`
- [ ] `docs/FINAL_RESULTS.md` updated with every new number (and nothing that the results do not support)
- [ ] Unit tests still pass (`python -m pytest tests`)

