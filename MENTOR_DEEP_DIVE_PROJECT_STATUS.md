# Explainable Hybrid Deep Learning-Based Intrusion Detection System for IoT Networks
## Master Faculty Mentor Preparation & Project Technical Deep-Dive

*Updated October 2026. Every number in this document comes from [docs/FINAL_RESULTS.md](docs/FINAL_RESULTS.md)
(the source of truth) or [docs/LEAKAGE_FIX_AND_CLEAN_BASELINES.md](docs/LEAKAGE_FIX_AND_CLEAN_BASELINES.md).
Older numbers (100% accuracy, 0.09 MB, 1.59 GB Random Forest, multiclass 0.644) appear only as
"what we first saw and why it was fake".*

---

## PART 1 — PROJECT STATUS AT A GLANCE

**Project Title:** Explainable Hybrid Deep Learning-Based Intrusion Detection System for IoT Networks
**Project Objective:** To design, implement, and honestly evaluate an Intrusion Detection System (IDS) for resource-constrained IoT environments that is (1) accurate, (2) explainable, and (3) lightweight.
**Current Research Problem:** IoT environments are heavily targeted by cyberattacks, but traditional IT security solutions are too resource-heavy. Deep learning models are often "black boxes", and public IoT datasets contain hidden shortcuts (artifacts that give away the label) that make models look perfect in the lab.
**Why this problem matters in IoT:** IoT devices control physical systems (e.g., industrial sensors). A missed attack (False Negative) can cause physical damage, while a false alarm (False Positive) can disrupt critical services. Transparency (explainability) is required for operators to trust the IDS.

### Status Summary
- **🟢 What has been completed:**
  - Edge-IIoTset pipeline; discovery and removal of the `"0"`/`"0.0"` empty-field shortcut; strict feature policy (7 per-packet identifiers removed); no-MQTT check.
  - Clean baselines (GPU XGBoost) and the CNN-GRU; Optuna tuning; 3 seeds for every comparison.
  - Ablation against MLP, 1D-CNN and GRU; SHAP with a fidelity (removal) test.
  - Measured detection ceiling; detection vs false-alarm operating points.
  - Edge (CPU) benchmark: FP32 / FP16 / INT8 vs XGBoost.
  - Second dataset (CICIoT2023, flow-level) with the same honest pipeline, where a second hidden shortcut was found and removed.
- **🟢 What is currently working:** End-to-end pipeline from raw CSV/Parquet to trained models, SHAP, figures and a live terminal dashboard (`watch_dashboard.py`). Safeguards: data fingerprint check, automatic shortcut scans.
- **🟡 What is honest but weak:** Attack-type classification on packet-level Edge-IIoTset (Macro-F1 0.34 CNN-GRU, 0.43 XGBoost). The hybrid gives no accuracy gain over simpler networks on per-packet features.
- **🔵 What is next:** Real sequences for the GRU (consecutive flow windows), cross-dataset test, rare-class improvements, deployment on a Raspberry-Pi-class device.

### Current Overall Pipeline

```text
       IoT Network Traffic (Raw CSV / Parquet)
                  ↓
       Raw shortcut audit ("0" vs "0.0" empty fields) 🟢 IMPLEMENTED
                  ↓
       Leakage policy (IPs, ports, timestamps removed) 🟢 IMPLEMENTED
                  ↓
       Strict policy (7 per-packet identifiers removed) + no-MQTT check 🟢 IMPLEMENTED
                  ↓
       Canonicalise numeric text ("0" = "0.0"), streaming low-RAM preprocessing 🟢 IMPLEMENTED
                  ↓
       Train / Validation / Test Split (70/15/15, stratified, seed 42) 🟢 IMPLEMENTED
                  ↓
       Processed-data shortcut scan (stumps, depth-3 tree) + fingerprint 🟢 IMPLEMENTED
                  ↓
       Baseline (GPU XGBoost) + ablation (MLP, 1D-CNN, GRU) 🟢 IMPLEMENTED
                  ↓
       Proposed CNN-GRU (Optuna-tuned, 3 seeds) 🟢 IMPLEMENTED
                  ↓
       Thresholds chosen on validation only; operating points 🟢 IMPLEMENTED
                  ↓
       SHAP (GradientExplainer) + fidelity test 🟢 IMPLEMENTED
                  ↓
       Detection ceiling audit 🟢 IMPLEMENTED
                  ↓
       Edge benchmark (FP32 / FP16 / INT8, 1 CPU thread) 🟢 IMPLEMENTED
                  ↓
       Second dataset: CICIoT2023 (flow-level) 🟢 IMPLEMENTED
                  ↓
       Dashboards (terminal; FastAPI web) 🟢 IMPLEMENTED   Live predict + SHAP demo 🟢 IMPLEMENTED
```

---

## PART 2 — 1-MINUTE EXPLANATION

### How to Explain Our Project in 60 Seconds
"Our project builds an Intrusion Detection System for IoT networks. IoT devices have limited memory and power, so the IDS must be small, and operators need to know *why* it raised an alarm, so it must be explainable. We built a hybrid CNN-GRU model with SHAP explanations.
Our first models scored 100%. Instead of celebrating, we investigated and found the score was fake: the Edge-IIoTset dataset writes an empty field as `"0"` in normal traffic and `"0.0"` in attack traffic, so one column gives away the answer. We fixed that, removed per-packet identifiers, and added automatic shortcut checks.
The honest result is 0.858 Macro-F1, 90% accuracy, with only 0.7% false alarms. We measured that this is the ceiling of the dataset: about a third of attack packets are identical to normal ones. We proved our SHAP explanations are faithful, showed the model shrinks to 159 KB with no loss, and confirmed the pipeline on a second dataset, CICIoT2023, where it reaches 0.89 and where we found and removed a second hidden shortcut."

---

## PART 3 — 5-MINUTE FACULTY EXPLANATION

### How to Explain the Complete Project in 5 Minutes

1. **Problem:** IoT networks are vulnerable and cannot run standard enterprise security tools.
2. **Motivation:** We need an IDS that is accurate, explainable, and lightweight.
3. **Existing approaches:** Many papers use Machine Learning (Random Forest, XGBoost) or Deep Learning (CNNs, LSTMs/GRUs), often reporting 99-100% on public IoT datasets.
4. **Research gaps:** Near-perfect scores are rarely checked for dataset shortcuts; explanations are shown as plots but rarely tested for fidelity; model size and CPU latency are often ignored.
5. **Datasets:** **Edge-IIoTset** (2,219,201 packets, Normal + 14 attacks) and **CICIoT2023** (flow-level, 33 attacks in 7 categories + Benign).
6. **What went wrong first:** All first models scored 100%. Root cause: the `"0"`/`"0.0"` spelling of empty fields, which separates Normal from Attack perfectly by itself.
7. **What we did:** Canonicalised the spelling, removed 7 per-packet identifiers, removed MQTT as a check, added automatic shortcut scans and a data fingerprint check, used 3 seeds and validation-only thresholds.
8. **Why CNN:** To extract local combinations of features from the input vector.
9. **Why GRU:** GRUs are lighter than LSTMs and are designed for sequences. *(Honest note: on per-packet data there is no real time dimension; see PART 20.)*
10. **Why CNN + GRU:** To test whether combining the two helps. **Finding:** on per-packet features it does not; MLP, 1D-CNN, GRU and CNN-GRU all reach 0.858.
11. **Why SHAP/XAI:** To show which features drive each alarm. Top features are now real TCP behaviour (`tcp.flags`, `tcp.connection.rst`, ...), and a removal test proves SHAP is faithful.
12. **Evaluation strategy:** Macro-F1 (main), accuracy, false-alarm rate, detection rate, ROC-AUC, model size and CPU latency. 3 seeds; thresholds chosen on validation only.
13. **Contribution:** Found and removed hidden shortcuts in two public IoT datasets; measured the detection ceiling and showed our model reaches it; proved explanation fidelity; honest, reproducible evaluation with a tested edge deployment.
14. **Future work:** Real sequences for the GRU (consecutive flow windows), cross-dataset test, rare-class improvements, Raspberry-Pi-class deployment.

---

## PART 4 — PROJECT PROBLEM STATEMENT

**What is an IoT intrusion?** Unauthorized access or malicious activity targeting IoT devices or the network connecting them (e.g., a Mirai botnet taking over a smart camera).
**What is an IDS?** An Intrusion Detection System monitors network traffic for suspicious activity and issues alerts.
**Why is IoT different from enterprise networks?** IoT networks have heterogeneous devices, operate on low power/memory, use specific protocols (like MQTT, CoAP), and have highly dynamic but predictable traffic patterns.
**What attacks are relevant?** DDoS, DoS, Botnets (Mirai), Information gathering (Scanning), injection/web attacks, and protocol-specific attacks.
**Why are traditional signature-based approaches insufficient?** They only detect *known* attacks. They fail against zero-day (new) attacks and require constant, heavy signature updates.
**Why use anomaly/ML/DL-based detection?** ML/DL can learn the difference between normal and malicious behaviour and generalise beyond exact signatures.
**Why does explainability matter?** A black-box model might flag traffic for arbitrary reasons, including dataset artifacts. Human analysts need to see the logic (e.g., "flagged because of the TCP flags and connection resets") to verify and act on the alert. In our project, an explanation tool that trimmed feature names actually hid the fake 100% (PART 34), so explanations must be exact and tested.
**Why does computational efficiency matter?** IoT edge gateways (like a Raspberry Pi) have limited RAM and CPU. Our final CNN-GRU is 159 KB (FP16) and takes about 6.5 ms per packet on one CPU thread.

---

## PART 5 — IOT FUNDAMENTALS

**IoT Architecture**
- **Perception/Device Layer:** The physical sensors and actuators collecting data (e.g., temperature sensors, cameras).
- **Network Layer:** The communication medium (Wi-Fi, Zigbee, 5G) and protocols (MQTT, TCP/IP) moving data.
- **Edge/Fog Layer:** Local gateways (e.g., routers) that aggregate data and perform initial processing close to the devices. *This is where our IDS is meant to live.*
- **Cloud/Service Layer:** Centralized servers for heavy data analytics and storage.
- **Application Layer:** The user-facing software (e.g., a smart home app).

**IoT Characteristics Affecting IDS Design**
- **Heterogeneity & Device Diversity:** An IDS must understand traffic from many different device types.
- **Resource Constraints:** IDS models must be lightweight (low RAM/CPU).
- **Scalability & Dynamic Traffic:** Traffic volume can spike; the IDS must infer rapidly.
- **Distributed Deployment:** Centralized cloud IDS has too much latency; IDS must be deployable at the edge.

---

## PART 6 — INTRUSION DETECTION FUNDAMENTALS

- **IDS (Intrusion Detection System):** General term for monitoring systems.
- **NIDS (Network IDS):** Monitors traffic across the entire network (analyzing packets/flows). *Our project is a NIDS.*
- **HIDS (Host IDS):** Installed on a specific device to monitor its internal OS/file system.
- **Signature-based IDS:** Matches traffic against a database of known attack fingerprints. (High accuracy for known, zero for unknown).
- **Anomaly-based IDS:** Learns normal behavior and flags deviations. (Detects unknowns, but higher false positives).
- **Hybrid IDS:** Combines Signature and Anomaly based approaches.
- **Edge IDS:** Deployed on local network gateways (e.g., IoT routers) for fast, low-latency detection without sending all data to the cloud.

**Crucial Distinction:**
- **Hybrid IDS:** Combining two detection *philosophies* (Signature + Anomaly).
- **Hybrid Deep Learning Architecture:** Combining two *neural network types* (CNN + GRU) into one model. *Our project focuses on this.*

**Packet-level vs flow-level:** Edge-IIoTset rows are single packets; CICIoT2023 rows summarise a window of packets (flow features, including rate and timing). This difference turns out to be central to our results (PART 42).

---

## PART 7 — ATTACKS

*Relevant to our datasets:*
- **DDoS/DoS (Distributed Denial of Service):** Overwhelming a target with traffic to make it unavailable. *Observable features: high packet rates, specific TCP flags (SYN).* Defined by *rate*, which a single packet cannot show; flow-level data with timing handles this much better.
- **Scanning/Reconnaissance:** Attackers probing the network to find open ports and vulnerabilities. *Observable features: connection attempts to many ports rapidly* (again a rate/behaviour-over-time property).
- **Injection / Web Attacks (XSS, SQLi, uploading, password attacks):** Malicious payloads sent to web interfaces. Hard to detect without the payload, which is dropped (high cardinality). These remain weak classes in our multiclass results.
- **Ransomware / Malware / Backdoor:** Malicious software encrypting data or taking control. *Observable features: unusual communication with external Command & Control (C2) servers.*
- **MITM / Fingerprinting:** Very rare in Edge-IIoTset (hundreds of test rows), hence weak per-class scores.

---

## PART 8 — DATASET DEEP DIVE

**🟢 MAIN DATASET: Edge-IIoTset**
- **Source:** Kaggle (`DNN-EdgeIIoT-dataset.csv`)
- **Hash:** `1d3ef6c7cc22784a528a117f1158f5cad750273679437e18a39c360ee6b79fd7`
- **Total Records:** 2,219,201 packets
- **Total Features:** 63 raw columns
- **Classes:** Normal + 14 attack types (15 classes for `Attack_type`; `Attack_label` binary)
- **Format:** Tabular, one row per packet.
- **Strengths:** Modern IoT/IIoT testbed, realistic protocols (MQTT, Modbus), diverse attacks.
- **Limitations found by us:**
  - Normal and attack traffic were captured separately, and the CSV export writes empty fields differently (`"0"` vs `"0.0"`): a label shortcut (PART 11).
  - Every MQTT packet is Normal (composition shortcut).
  - No timing for attack packets (`udp.time_delta` is 0 for every attack row).
  - The 1,553,440 training packets contain only 11,122 distinct feature patterns; 31.9% of attack packets look exactly like mostly-Normal patterns (detection ceiling, PART 42).

**🟢 SECOND DATASET: CICIoT2023**
- **Source:** Kaggle `dhoogla/ciciotdataset2023` (46.8M rows), downloaded via `kagglehub` into the user cache, outside the project/OneDrive.
- **Content:** Flow-level features (with rate and timing), 33 attacks in 7 categories + Benign.
- **Our sample:** 1.41M rows (≤ 40k per attack label, 400k Benign), 87,417 duplicates removed **before** splitting.
- **Shortcuts found and removed:** `source_file` (names the capture, i.e. the label) and the window-size artifact (`packet_count` is always 10 for Benign/Recon and 100 for DDoS/DoS/Mirai; `total_sum = packet_count × mean_packet_size`).

---

## PART 9 — DATASET COMPARISON

| Dataset | IoT-specific | Scale | Attack Coverage | Strength | Limitation | Why we use / don't use |
|---|---|---|---|---|---|---|
| NSL-KDD | No | Small | Old/Outdated | Well-studied | Too old (1999), not IoT | **Don't use:** Not relevant for modern IoT. |
| Bot-IoT | Yes | Large | Botnet focused | Good IoT traffic | Limited attack diversity | **Don't use:** Too narrow in scope. |
| **Edge-IIoTset** | **Yes** | **2.2M packets** | **Normal + 14 attacks** | **Modern protocols** | **Packet-level, no timing for attacks; `"0"`/`"0.0"` artifact** | **USE (main):** Modern IIoT testbed; our shortcut audit is a contribution. |
| **CICIoT2023** | **Yes** | **46.8M flows** | **33 attacks, 7 categories** | **Flow-level with rate/timing** | **Window-size artifact; duplicates** | **USE (second):** Tests whether timing information helps and whether our pipeline generalises. |

---

## PART 10 — ACTUAL DATASET SCHEMA

**🟢 IMPLEMENTED & VERIFIED (Edge-IIoTset)**
- **Total Raw Features:** 63
- **Target Fields:** `Attack_label` (binary: 0/1) and `Attack_type` (multiclass: 15 classes).
- **Feature policies:** `operational` (identifiers/timestamps removed; 83 encoded features after canonicalisation), `strict` (also 7 per-packet identifiers removed), `strict_no_mqtt` (also all `mqtt.*` removed; 51 encoded features, the final setting).

**Sample Feature Schema Table:**

| Feature | Type | Meaning | Used in final model? | Reason |
|---|---|---|---|---|
| `ip.src_host` / `ip.dst_host` | String | Source/Dest IP | 🔴 NO | Identifier |
| `frame.time` | String | Packet Timestamp | 🔴 NO | Leakage (capture time) |
| `tcp.payload` | String | Raw packet data | 🔴 NO | High cardinality (>100 unique) |
| `tcp.seq`, `tcp.ack`, `tcp.ack_raw` | Float | Sequence/ack numbers | 🔴 NO | Position inside one capture (strict policy) |
| `tcp.checksum`, `icmp.checksum` | Float | Per-packet checksum | 🔴 NO | Effectively a random identifier (strict policy) |
| `icmp.seq_le`, `udp.stream` | Float | Sequence counter / stream index | 🔴 NO | Assigned per capture (strict policy) |
| `mqtt.topic`, `mqtt.conack.flags`, ... | String | MQTT fields | 🔴 NO | Every MQTT packet is Normal; removed as a check (score unchanged) |
| `dns.qry.name.len` | String | DNS query name length | 🟢 YES, canonicalised | Was the main `"0"`/`"0.0"` shortcut before canonicalisation |
| `tcp.flags`, `tcp.connection.rst`, `tcp.len` | Float | TCP behaviour | 🟢 YES | Behavioural; top SHAP features |

*Note: Features with >100 unique categorical values (like `tcp.options`, `http.request.full_uri`) are dropped to prevent memory explosion during One-Hot Encoding.*

---

## PART 11 — LEAKAGE

**What is data leakage?** When a model learns to predict the target using information that will not be available in the real world, or using dataset-specific artifacts rather than generalized patterns.
**Why is leakage dangerous?** It produces falsely high accuracy in the lab, but the model fails in the real world.

**🟢 Step 1: Identifier/timestamp removal (operational policy).**
- **Removed:** `frame.time`, `ip.src_host`, `ip.dst_host`, `arp.dst.proto_ipv4`, `arp.src.proto_ipv4`, `icmp.transmit_timestamp`, `tcp.dstport`, `tcp.srcport`, `udp.port`, `udp.time_delta`, `mbtcp.trans_id`, `mbtcp.unit_id`.
- **Why:** IPs, ports and timestamps identify *who/when*, not *how* the network behaves.
- **This was NOT enough.** After this step all models still scored 100%.

**🟢 Step 2: The real cause, the `"0"` vs `"0.0"` artifact (found by us).**
The dataset writes an empty protocol field as `"0"` in the Normal captures and as `"0.0"` in the attack captures. These columns are strings, so one-hot encoding turns the two spellings into two different columns, and one column alone gives away the label.

| Empty `dns.qry.name.len` written as | Normal rows | Attack rows |
|---|---|---|
| `"0"` | 1,613,798 | 0 |
| `"0.0"` | 0 | 603,331 |

- The same holds for `mqtt.topic`, `mqtt.protoname`, `mqtt.conack.flags`; the HTTP columns use `"0"` only for web attacks (leaking the attack *type*). Evidence: `results/audit/empty_token_audit.csv` (`python audit_shortcuts.py --raw`).
- **Fix:** canonicalise numeric-looking text (`"0"` = `"0.0"`) before encoding. Real text values (e.g. `MQTT`) are untouched.
- **Shortcut scan after the fix:** best single feature 1.00 → 0.72, depth-3 tree 1.00 → 0.91 (balanced accuracy, binary).

**🟢 Step 3: Per-packet identifiers (strict policy).** After the fix, the strongest single features were `tcp.seq`, checksums and counters, which locate a packet inside one capture. Removed: `tcp.seq`, `tcp.ack`, `tcp.ack_raw`, `tcp.checksum`, `icmp.checksum`, `icmp.seq_le`, `udp.stream`. Binary shortcut scan under strict: best single feature 0.70, depth-3 tree 0.73.

**🟢 Step 4: MQTT check (strict_no_mqtt).** Every MQTT packet is Normal, so "is this MQTT" partly answers "is this benign". We removed all `mqtt.*` fields: Macro-F1 0.8579 (with) vs 0.8575 (without), so the model does not use that shortcut.

**🟢 Step 5: Guards so it cannot happen silently again.** `data/processed/metadata.json` stores target, policy, canonicalisation flag, feature list and a fingerprint. Training refuses data built for another target; SHAP refuses to explain a model whose fingerprint does not match (an earlier SHAP plot had explained a model with the wrong inputs).

**What we claim:** we found and removed the shortcuts our audits can detect, and we test for single-feature giveaways automatically. **What we do not claim:** "leakage-free" or "immune to shortcuts". A random split of packets from separately captured files can still hide subtler artifacts.

---

## PART 12 — WHY OUR FEATURE POLICY MATTERS

**Shortcut Features:** Features that allow the model to cheat. If all attacks in the dataset happened on a Tuesday, "Day=Tuesday" becomes a shortcut feature. The model looks artificially accurate but learns nothing about network security.
**Our policy, in three layers:** (1) remove identifiers and timestamps, (2) canonicalise the empty-field spelling and remove per-packet identifiers, (3) remove MQTT as a check.
**What we first believed (and why it was wrong):** We used to say the 100% binary accuracy came from real protocol behaviour, "specifically MQTT, as revealed by SHAP". That was wrong twice: the 100% came from the `"0"`/`"0.0"` spelling, and the old SHAP plot hid it because it stripped the `_0` / `_0.0` suffix from feature names, so both spellings appeared as `mqtt.topic`. Removing MQTT did not change the 100% at the time, because `dns.qry.name.len` carried the same artifact.
**What the policy gives us now:** an honest 0.858 Macro-F1 that does not depend on MQTT and is explained by TCP behaviour.

---

## PART 13 — PREPROCESSING DEEP DIVE

**🟢 ACTUALLY IMPLEMENTED (in `src/preprocessing.py`)**
1. **Data Loading:** Streams the CSV and never loads payload columns; writes float32. Runs in a child process so its RAM is returned to the OS (peak 6.1 → 3.4 GB).
2. **Cleaning:** Drops columns according to the selected feature policy (`operational`, `strict`, `strict_no_mqtt`). Drops high-cardinality string columns (>100 unique values). Drops rows with missing target labels.
3. **Canonicalisation:** Numeric-looking strings in categorical columns are canonicalised (`"0"` and `"0.0"` → `"0.0"`) before encoding (`CANONICALIZE_NUMERIC_TOKENS = True` in `src/config.py`).
4. **Target Selection:** Excludes `Attack_type` when training `Attack_label`, and vice versa.
5. **Splitting:** Train/Val/Test (70/15/15), stratified, **BEFORE** any scaling or encoding.
6. **Imputation & Scaling:** Missing numeric values filled with the median; `StandardScaler` fitted on train only.
7. **Categorical Encoding:** `OneHotEncoder` fitted on train only.
8. **Metadata & fingerprint:** `data/processed/metadata.json` records target, policy, canonicalisation flag, features and a fingerprint; every experiment record stores it.
9. **Shortcut scan:** single-feature decision stumps and a depth-3 tree on the processed data.
10. **Serialization:** Processed splits saved as `.parquet`.

---

## PART 14 — MOST IMPORTANT: WHY SPLIT BEFORE PREPROCESSING?

**🟢 IMPLEMENTED & VERIFIED**
Our pipeline explicitly splits the data **before** fitting the `StandardScaler` or `OneHotEncoder`.
- **Correct Way (Implemented):** `Raw Data` → `Split (Train/Val/Test)` → `Fit Scaler on TRAIN ONLY` → `Transform Train, Val, Test`.
- **Wrong Way:** `Raw Data` → `Fit Scaler on ALL Data` → `Split`.

**Why the wrong way causes leakage:** If you scale using the mean/variance of the *entire* dataset, information about the Test set's distribution "leaks" into the Training process. We actively prevent this.
**The same rule for thresholds and tuning:** decision thresholds and Optuna tuning use the validation split only; the test set is touched once, for the final number.

---

## PART 15 — TRAIN / VALIDATION / TEST SPLITTING

**🟢 ACTUALLY IMPLEMENTED:**
- **Random Stratified Split:** `train_test_split` with `stratify=y`, 70% / 15% / 15%, split seed 42.
- **Multiple seeds:** every model comparison is repeated with 3 training seeds (42, 7, 2024).
- **CICIoT2023:** duplicates removed before splitting, so copies of one flow window cannot land in both train and test.

**🔵 NOT IMPLEMENTED (acknowledged limitation):**
- Chronological or group split (by device/capture). One random split per dataset; no cross-dataset test yet.

---

## PART 16 — MACHINE LEARNING BASICS

**Why baseline models?** We need to know if Deep Learning is actually necessary. If a simple model works just as well, DL is a waste of IoT resources.
- **XGBoost on the GPU (🟢 current baseline):** gradient-boosted trees. Very strong on tabular data, trains in seconds on the GPU.
- **Random Forest (historical):** our first baseline. It used all CPU cores, several GB of RAM and produced a 1.6 GB multiclass model file, so we replaced it with GPU XGBoost.
- **Honest result:** XGBoost is slightly better than our CNN-GRU on both datasets (Edge-IIoTset 0.868 vs 0.858; CICIoT2023 binary 0.917 vs 0.890) and faster on a CPU. We report this openly.

---

## PART 17 — DEEP LEARNING BASICS

**Deep Learning / Neural Networks:** Composed of layers of artificial neurons.
- **Weights/Biases:** Parameters the network learns.
- **Forward propagation:** Data moving input-to-output.
- **Loss:** Calculating how wrong the prediction was.
- **Backpropagation / Gradient Descent:** Updating weights to minimize the loss.

---

## PART 18 — CNN DEEP DIVE

**🟢 IMPLEMENTED (1D-CNN, also part of the ablation)**
- **What is a CNN?** Convolutional Neural Network. Uses filters (kernels) to scan across data and extract local patterns.
- **Why 1D CNN?** We apply a 1D convolution across the tabular feature vector.
- **Limitation Acknowledged:** Tabular feature order is arbitrary (unlike pixels in an image). In our ablation the 1D-CNN reaches the same Macro-F1 (0.858) as an MLP, so the convolution gives no measurable advantage on these features.

---

## PART 19 — GRU DEEP DIVE

**🟢 IMPLEMENTED (GRU, also part of the ablation)**
- **What is an RNN/GRU?** Recurrent Neural Networks process sequential data. GRU (Gated Recurrent Unit) is a lighter version of LSTM that addresses the "vanishing gradient" problem using Update and Reset gates.
- **Why GRU?** It has fewer parameters than LSTM, making it lighter for Edge IoT.
- **⚠️ CRITICAL CONTEXT:** See Part 20.

---

## PART 20 — CRITICAL QUESTION: DO WE REALLY HAVE TEMPORAL SEQUENCES?

**🟢 ACTUAL STATUS: NO REAL TIME DIMENSION**
**This is very important for the viva.**
- Each row is processed on its own. On Edge-IIoTset a row is one packet; timestamps were removed as leakage, and attack rows have no timing at all (`udp.time_delta` is 0 for every attack row).
- **Standalone GRU baseline:** treats the feature vector as a sequence of length 1.
- **Inside the CNN-GRU:** the GRU steps over the positions of the CNN's feature map, i.e. over positions in the feature vector, **not over time**.
- **Honest claim:** We do not artificially reshape independent rows into fake time sequences.
- **Consequence (measured):** the hybrid gives no accuracy gain over an MLP (PART 42). The plan is to give the GRU real sequences: consecutive flow windows per device/connection (PART 52).

---

## PART 21 — CNN + GRU

**🟢 IMPLEMENTED (proposed model, tuned)**
**How the hybrid works:**
```text
Input (Tabular Vector, 51 features in the final setting)
        ↓
Conv1D (Extracts local feature combinations)
        ↓
BatchNorm + ReLU + MaxPool (Reduces dimensionality) + Dropout
        ↓
Transposed to sequence format (feature-map positions as steps)
        ↓
GRU (Processes the feature maps; last hidden state)
        ↓
Dense Classifier (Final Prediction)
```
- **CNN Contribution:** Local pattern extraction and dimensionality reduction.
- **GRU Contribution:** Processing the extracted feature maps.
- **Why combine?** Hypothesis: extracting features first (CNN) makes the recurrent processing (GRU) more effective. **Tested result:** on per-packet features the combination matches but does not beat simpler networks.

---

## PART 22 — OUR ACTUAL CNN-GRU ARCHITECTURE

**🟢 IMPLEMENTED (`src/models.py`, final tuned settings from Optuna, 20 trials)**

| Layer | Configuration | Purpose |
|---|---|---|
| Input | `(Batch, 1, 51)` | 1D tabular input (strict, no-MQTT features) |
| Conv1D | `filters=64, kernel=5, same padding` | Local feature combination |
| BatchNorm1D | `features=64` | Training stability |
| ReLU + MaxPool1D | `kernel=2` | Activation & dimension reduction |
| Dropout | `p≈0.24` | Regularization |
| Transpose | `(Batch, 25, 64)` | Prepare for GRU |
| GRU | `input_size=64, hidden=128` | Processes feature-map positions |
| Dense | `Linear(128 -> 32) + ReLU + Dropout` | Classification head |
| Output | `Linear(32 -> 1)` | Final logit |

- **Optimizer:** Adam, LR ≈ 0.003; **Loss:** binary cross-entropy (`BCEWithLogitsLoss`)
- **Batch Size:** 8192
- **Epochs:** 15 (with early stopping); decision threshold chosen on validation
- **Model Size:** 79,169 parameters, **315 KB** (FP32), **159 KB** with FP16 weights
- **Tuning:** best validation Macro-F1 0.8596; test 0.858, the same as the untuned model.

*Historical note: the original (untuned) CNN-GRU had 21,121 parameters and was reported as "0.09 MB with 100% accuracy". The size was real; the accuracy was the `"0"`/`"0.0"` shortcut.*

---

## PART 23 — WHY NOT TRANSFORMER / GNN?

- **Transformers:** Excellent at long-range dependencies, but computationally heavy and data-hungry. Our results show the limit on Edge-IIoTset is the information in a single packet (the measured ceiling), not model capacity, so a bigger model would not help there.
- **GNNs (Graph Neural Networks):** Great for modeling network topologies (device-to-device relationships). However, constructing graphs in real time adds preprocessing overhead, unsuitable for high-speed edge inference.

---

## PART 24 — WHY NOT ONLY CNN? & PART 25 — WHY NOT ONLY GRU?

- **CNN alone:** Good at local patterns, but lacks the internal memory mechanisms of recurrent models.
- **GRU alone:** Good at states, but lacks the feature reduction of a CNN.
- **Hypothesis:** The Hybrid (CNN-GRU) provides complementary benefits.
- **What we measured (ablation, same training settings, 3 seeds each):** MLP 0.858, 1D-CNN 0.858, GRU 0.858, CNN-GRU 0.858. On per-packet features **all networks hit the same ceiling**; the hybrid is the largest network (315 KB vs 24 KB for the MLP). The hybrid's value can only show with real sequences, which is our next step.
- *Historical note: we earlier argued the CNN-GRU was "optimal for the edge" because all models hit 1.0 F1 and it was smallest. Both halves no longer hold: the 1.0 was fake, and the tuned CNN-GRU is not the smallest.*

---

## PART 26 — CLASSIFICATION

**🟢 IMPLEMENTED**
- **Binary Classification (main result):** Normal (0) vs Attack (1). Edge-IIoTset: CNN-GRU 0.858, XGBoost 0.868. CICIoT2023: CNN-GRU 0.890, XGBoost 0.917.
- **Multiclass, Edge-IIoTset:** Normal + 14 attack types. Weak on packet data: XGBoost 0.429, CNN-GRU 0.337 ± 0.033.
- **Multiclass, CICIoT2023:** 8 categories (7 attack categories + Benign). XGBoost 0.752, CNN-GRU 0.675 (sqrt weights).

---

## PART 27 — CLASS IMBALANCE

**🟢 IMPLEMENTED: Macro-F1 + a loss study over 3 seeds**
- **The Problem:** If most traffic is Normal, a model that guesses "Normal" every time gets high accuracy but fails to detect attacks.
- **Evaluation:** **Macro-F1**, which averages the F1 score across all classes equally.
- **Loss study (Edge-IIoTset strict, multiclass CNN-GRU, 15 epochs, seeds 42 / 7 / 2024):**

| Loss | Macro-F1 mean ± sd |
|---|---|
| Cross-entropy | 0.238 ± 0.020 |
| Focal (γ=2) | 0.254 ± 0.008 |
| **Sqrt class weights** `sqrt(N / (K * count))` | **0.337 ± 0.033** |

- Sqrt weights beat both alternatives on every seed. Raw inverse-frequency weights (span ~1600:1) made training collapse; the square root (~40:1) is stable.
- On CICIoT2023 categories, sqrt weights again help: 0.675 vs 0.620 (cross-entropy).
- *Historical note: an earlier document claimed focal loss raised Macro-F1 from 0.263 to 0.408. Those runs used only 5 epochs, before the shortcut fix; at equal epochs on clean data the claim did not hold.*

---

## PART 28 — EVALUATION METRICS

- **True Positive (TP):** Attack correctly flagged as Attack.
- **True Negative (TN):** Normal correctly flagged as Normal.
- **False Positive (FP):** Normal falsely flagged as Attack (False Alarm).
- **False Negative (FN):** Attack falsely flagged as Normal (Missed Intrusion - Very dangerous).
- **Accuracy:** Overall correctness.
- **F1-Score:** Harmonic mean of Precision and Recall.
- **Macro-F1:** The unweighted average of F1 scores for all classes. *Our main metric.*
- **Detection rate (recall on attacks) and false-alarm rate:** what an operator actually experiences.
- **ROC-AUC:** threshold-independent ranking quality (CNN-GRU 0.904, XGBoost 0.917 on Edge-IIoTset).
- **Efficiency:** model size, latency per packet and throughput on one CPU thread.

---

## PART 29 — ERROR TYPES

- **False Positive (FP) Operational Cost:** Causes alert fatigue. If the IDS cries wolf too often, security teams ignore it.
- **False Negative (FN) Operational Cost:** The attacker breaches the system. Can lead to physical damage in IIoT.
- **Our trade-off (`results/operating_points.csv`, thresholds chosen on validation):** on Edge-IIoTset the CNN-GRU runs at 0.7% false alarms and detects 65.3% of attacks; lowering the threshold raises detection to 73.3% at 6.6% false alarms. On CICIoT2023 at ≤ 1% false alarms, XGBoost detects 86.6% and the CNN-GRU 82.3%.

---

## PART 30 — XAI (EXPLAINABLE AI)

- **Deep learning is a "black box":** Humans cannot read weights and biases.
- **Why Cybersecurity needs it:** An analyst receiving an alert needs to know *why* to investigate efficiently and trust the system.
- **Interpretability vs Explainability:** Interpretability means the model is inherently simple (like a small Decision Tree). Explainability means using a secondary tool (like SHAP) to explain a complex black-box model.
- **Lesson from our project:** explanations are only useful if the feature names are exact and the explained data matches the model. Both went wrong in our first SHAP run (PART 34).

---

## PART 31 & 32 — SHAP vs LIME

- **SHAP (SHapley Additive exPlanations):** Based on game theory. It attributes a prediction to each feature, compared to a baseline (e.g., `tcp.flags` pushed this prediction towards "attack").
- **LIME:** Builds a local, simple approximation model around a single prediction.
- **Our Choice (🟢 Implemented):** SHAP `GradientExplainer`, which works directly with neural network gradients and gives both global (ranking) and local (single-prediction waterfall) explanations.

---

## PART 33 — WHY SHAP PLOT ALONE IS NOT ENOUGH

A SHAP plot explains *what the model learned*, but it does **not** prove that the explanation is faithful or that the features are legitimate. If a model learned a shortcut, SHAP will happily highlight it, and a labelling bug can hide it. *Generating* an explanation is different from *evaluating* its fidelity (does removing the feature actually drop performance?). We now do both (PART 35).

---

## PART 34 — OUR CURRENT SHAP IMPLEMENTATION

**🟢 ACTUALLY IMPLEMENTED (`src/explainability.py`)**
- **Model explained:** final CNN-GRU (strict, no MQTT, seed 42): `results/experiments/F_strict_no_mqtt_binary_best_s42/`. CICIoT2023 CNN-GRU: `results/experiments/CS02_cnn_gru_binary_strict/`.
- **Method:** `shap.GradientExplainer`.
- **Outputs generated:** Beeswarm plot, mean |SHAP| bar chart, single-sample waterfall plot, CSV ranking.
- **Safeguards:** SHAP compares the experiment's data fingerprint with `data/processed/` and refuses mismatched data; labels keep the one-hot category (e.g. `mqtt.topic=0.0`), so a `"0"`/`"0.0"` split can no longer be hidden.
- **Results:**
  - Edge-IIoTset top features: `tcp.flags`, `tcp.connection.rst`, `tcp.flags.ack`, `tcp.len`, `tcp.connection.fin` (real TCP behaviour).
  - CICIoT2023 top features: `header_length`, `ack_count_frac`, `rate`, `https`, `arp` (rate/timing information now matters, as expected).
- **What we first saw and why it was wrong:** The first SHAP run ranked MQTT fields on top (`mqtt.topic`, `mqtt.protoname`, `mqtt.conack.flags`). That plot stripped the `_0` / `_0.0` suffix, so it showed the formatting shortcut under an innocent-looking name. A later "no-MQTT" SHAP plot was invalid: a 62-feature model had been run on 91-feature data without an error. Both problems are now prevented automatically.

---

## PART 35 & 36 — EXPLANATION FIDELITY & STABILITY

**🟢 FIDELITY: IMPLEMENTED (`results/shap_fidelity.json`)**
- Retrain the final CNN-GRU **without its top-5 SHAP features**: Macro-F1 **0.858 → 0.605** (−0.253).
- Control: retrain **without 5 random features**: 0.858 → 0.858 (−0.0001).
- Conclusion: SHAP identifies what the model really depends on. Figure: `results/figures/fig2_shap_fidelity.png`.

**🟡 STABILITY:**
- Model scores are stable over 3 seeds (0.8584 / 0.8583 / 0.8582).
- Comparing SHAP rankings across seeds has not been reported yet (future work).

---

## PART 37 — EFFICIENCY

**🟢 ACTUALLY MEASURED (1 CPU thread, `results/edge_benchmark.csv`)**

| Version | Macro-F1 | Size | Latency per packet | Throughput |
|---|---|---|---|---|
| CNN-GRU FP32 | 0.858 | 315 KB | 6.4 ms | 1,500 rows/s |
| **CNN-GRU FP16 weights** | **0.858** | **159 KB** | 6.5 ms | 1,660 rows/s |
| CNN-GRU INT8 (Linear only) | 0.858 | 304 KB | 8.1 ms | 1,620 rows/s |
| CNN-GRU INT8 (GRU + Linear) | 0.782 | 88 KB | 13.3 ms | 2,050 rows/s |
| XGBoost | 0.868 | 492 KB | 0.8 ms | 30,700 rows/s |

- FP16 halves the size with no loss; INT8 on the GRU costs 7.6 points.
- XGBoost is faster on a CPU; the networks are smaller (MLP 24 KB).
- **Training efficiency:** data held on the GPU, ≈8 s/epoch vs ≈31 s before (RTX 4060 Laptop GPU); preprocessing peak RAM 6.1 → 3.4 GB.
- *Historical note: we used to argue "Random Forest 1.59 GB vs CNN-GRU 0.09 MB, a 17,000x reduction". That compared a bloated 200-tree Random Forest (on leaky multiclass data) against a network, and is not a fair argument. A properly configured tree model (XGBoost, 492 KB) is small too. Our honest efficiency claim is the measured table above.*

---

## PART 38 — EDGE DEPLOYMENT

**🟡 CURRENT STATUS:** CPU benchmark done; physical device pending.
- Benchmarked on one CPU thread of the laptop (PART 37): FP16 CNN-GRU at 159 KB, 6.5 ms per packet, no accuracy loss.
- **🟢 Implemented:** live demo in the FastAPI dashboard (`uvicorn dashboard.api:app`, open `http://127.0.0.1:8000/#predict-demo`): the final CNN-GRU classifies held-out test packets and shows the top SHAP reasons.
- **🔵 Planned:** run the FP16 model on a Raspberry-Pi-class device.

---

## PART 39 — RESEARCH GAPS

| Research Gap | Why Important | Our Response / Current Status |
|---|---|---|
| Hidden dataset shortcuts | Cause fake 99-100% accuracy | 🟢 Found and removed the `"0"`/`"0.0"` artifact (Edge-IIoTset) and window-size artifact (CICIoT2023); automatic shortcut scans. |
| Unknown limits of the data | A "low" score may be the data, not the model | 🟢 Measured the detection ceiling (67.5%); our model reaches it. |
| Black-box detection, untested explanations | Security teams lack trust; plots can mislead | 🟢 SHAP plus a fidelity (removal) test. |
| Model Size / Edge Suitability | Giant models can't run on IoT | 🟢 Measured CPU latency and size; FP16 CNN-GRU 159 KB, no loss. |
| Fair comparison | Hybrid models are often not compared to simple ones | 🟢 Ablation vs MLP / 1D-CNN / GRU / XGBoost, 3 seeds. |

---

## PART 40 — NOVELTY

**What is NOT novel:** CNN-GRU architectures exist. SHAP exists. Quantisation exists.
**Our defensible contribution:**
1. **Found and removed hidden shortcuts in two public IoT datasets** (the `"0"`/`"0.0"` artifact in Edge-IIoTset; the window-size artifact in CICIoT2023) that make models look perfect.
2. **Measured the detection ceiling** of packet-level Edge-IIoTset and showed our model reaches it.
3. **Proved explanation fidelity** (SHAP removal test), not just SHAP plots.
4. **Honest, reproducible evaluation:** 3 seeds, fair ablation against simpler models, validation-only thresholds, tested edge deployment, GPU pipeline that runs on a laptop.

*We do not claim "leakage-free" or that the hybrid beats simpler models on packet data.*

---

## PART 41 — CURRENT PHASE STATUS

| Phase | Description | Status | Evidence |
|---|---|---|---|
| Phase 1 | Dataset + audit + preprocessing | 🟢 DONE | `src/preprocessing.py`, `results/audit/empty_token_audit.csv`, `data/processed/metadata.json` |
| Phase 2 | Shortcut fix + strict / no-MQTT policies | 🟢 DONE | `audit_shortcuts.py`, `results/audit/strict_feature_policy.csv` |
| Phase 3 | Baselines (XGBoost) + CNN-GRU, tuning, 3 seeds | 🟢 DONE | `results/experiment_registry.csv` (N01, F_* runs), `run_tune_cnn_gru.py` |
| Phase 4 | Ablation, SHAP + fidelity, edge benchmark | 🟢 DONE | `run_final_studies.py`, `results/shap_fidelity.json`, `results/edge_benchmark.csv` |
| Phase 5 | Detection ceiling + operating points | 🟢 DONE | `results/audit/detection_ceiling.json`, `results/operating_points.csv` |
| Phase 6 | Second dataset (CICIoT2023) | 🟢 DONE | `run_ciciot.py --policy strict`, CS01-CS05 experiments |
| Phase 7 | Dashboards | 🟢 DONE | `watch_dashboard.py`, FastAPI in `dashboard/` |
| Phase 8 | Figures + write-up | 🟢 Figures done, 🟡 report in progress | `results/figures/`, `docs/FINAL_RESULTS.md` |
| Phase 9 | Real sequences, cross-dataset, device deployment | 🔵 PLANNED | PART 52 |

---

## PART 42 — ACTUAL CURRENT RESULTS

**🟢 VERIFIED (from `results/experiment_registry.csv` and the files named below)**

**What we first saw, and why it was fake:** all first models scored 100% (Macro-F1 1.000) on binary, and the old multiclass CNN-GRU scored 0.644 Macro-F1 at 94.7% accuracy. The binary 100% came from the `"0"`/`"0.0"` artifact; the multiclass 0.644 was inflated by the HTTP `"0"`/`"0.0"` tokens, which marked the web-attack classes.

**Edge-IIoTset, binary (strict features, no MQTT: final setting), test set:**

| Model | Macro-F1 | Accuracy | False alarms | Attacks detected | ROC-AUC |
|---|---|---|---|---|---|
| Original pipeline (leaky) | 1.000 | 100% | 0% | 100% | - |
| **CNN-GRU (tuned, ours)** | **0.858** | **90.0%** | **0.7%** | **65.3%** | 0.904 |
| XGBoost (GPU) | 0.868 | 90.7% | 0.6% | 67.3% | 0.917 |

**Robustness checks:**
- Without MQTT vs with MQTT: 0.8575 vs 0.8579, so the MQTT shortcut is not used.
- 3 seeds (tuned CNN-GRU): 0.8584 / 0.8583 / 0.8582.
- Optuna, 20 trials: best validation 0.8596; test 0.858, same as untuned. Tuning cannot help.

**Why the score stops at ~0.86, a measured ceiling (`results/audit/detection_ceiling.json`):**
- The 1,553,440 training packets contain only **11,122 distinct feature patterns**.
- **31.9% of attack packets** have exactly the same features as patterns that are mostly Normal.
- Even a perfect "memorise every pattern" classifier detects only **67.5%** of attacks (at 0.5% false alarms). Our models detect 65.3% (CNN-GRU) and 67.3% (XGBoost): **they are at the ceiling**.

**Ablation (same training settings, 3 seeds each):**

| Model | Macro-F1 (mean) | Parameters | File size |
|---|---|---|---|
| MLP | 0.858 | 5,441 | 24 KB |
| 1D-CNN | 0.858 | 25,857 | 106 KB |
| GRU | 0.858 | 24,577 | 99 KB |
| CNN-GRU (tuned) | 0.858 | 79,169 | 315 KB |
| XGBoost | 0.868 | 226 trees | 492 KB |

**Attack type (14 classes + Normal), Edge-IIoTset strict:** XGBoost 0.429, CNN-GRU 0.337 ± 0.033 (sqrt weights). Floods and scans (DDoS_UDP, Port_Scanning) are defined by packet rate, which a single packet cannot show.

**CICIoT2023 (`run_ciciot.py --policy strict`):**

| Task | Model | Macro-F1 | Notes |
|---|---|---|---|
| Benign vs Attack | XGBoost | **0.917** | detects 93.4% of attacks, 8.4% false alarms; ROC-AUC 0.979 |
| Benign vs Attack | CNN-GRU | **0.890** | detects 90.4%, 9.6% false alarms (validation-tuned threshold); ROC-AUC 0.966 |
| 8 categories | XGBoost | **0.752** | |
| 8 categories | CNN-GRU, sqrt weights | **0.675** | cross-entropy: 0.620 |

- Shortcut scan after the fix: best single feature 0.85 (binary), depth-3 tree 0.85; no giveaway.
- Removing the window artifacts barely changed the scores (binary 0.9166 → 0.9165), so they hold.
- Edge-IIoTset vs CICIoT2023 (CNN-GRU): binary 0.86 → 0.89, categories 0.34 → 0.68.

**Figures (`results/figures/`):** `fig1_fake_vs_honest.png`, `fig2_shap_fidelity.png`, `fig3_model_comparison.png`, `fig4_datasets.png`, `fig5_detection_ceiling.png`.

---

## PART 43 — WHAT WE CAN DEMONSTRATE TODAY

**Demo Checklist:**
- 🟢 **STATIC:** `python audit_shortcuts.py --raw` output / `results/audit/empty_token_audit.csv`: the `"0"`/`"0.0"` evidence table.
- 🟢 **STATIC:** Figures in `results/figures/` (fake vs honest, SHAP fidelity, model comparison, datasets, detection ceiling).
- 🟢 **STATIC:** SHAP plots for the final CNN-GRU showing TCP behaviour (`results/experiments/F_strict_no_mqtt_binary_best_s42/`).
- 🟢 **LIVE:** `python watch_dashboard.py` terminal dashboard during a run; FastAPI web dashboard.
- 🟢 **LIVE:** `python audit_ceiling.py` (detection ceiling) and `python run_operating_points.py`.
- 🔵 **PLANNED:** Live POST-request inference against the deployed model.

---

## PART 44 — END-TO-END DATA FLOW

```text
CSV (Edge-IIoTset) / Parquet (CICIoT2023, kagglehub cache)
  ↓ (streaming loader, payload columns never loaded)
Feature policy (identifiers, timestamps, per-packet IDs, optional MQTT removed)
  ↓ (high-cardinality drop, "0" = "0.0" canonicalisation)
Split (Train/Val/Test 70/15/15, stratified)
  ↓ (Fit Scaler/Encoder on TRAIN ONLY)
Encoder/Scaler + metadata.json fingerprint
  ↓
Parquet Files + automatic shortcut scan
  ↓ (data held on the GPU)
XGBoost / MLP / 1D-CNN / GRU / CNN-GRU (3 seeds; threshold chosen on validation)
  ↓
Metrics Tracker (results/experiment_registry.csv)
  ↓ (GradientExplainer, fingerprint check)
SHAP + fidelity test
  ↓
Edge benchmark, detection ceiling, operating points, figures
```

---

## PART 45 — FILE-BY-FILE CODEBASE EXPLANATION

| File | Purpose | Important Functions |
|---|---|---|
| `src/config.py` | Paths, hyperparameters, seeds, `CANONICALIZE_NUMERIC_TOKENS`. | `set_seeds()` |
| `src/preprocessing.py` | Feature policy, canonicalisation, split, scaling, metadata/fingerprint. | `run_preprocessing_pipeline()` |
| `src/ciciot.py` | CICIoT2023 sampling, de-duplication, artifact removal. | `build_ciciot_datasets()` |
| `src/models.py` | MLP, 1D-CNN, GRU, CNN-GRU. | `MLP`, `CNN1D`, `GRUBaseline`, `CNN_GRU` |
| `src/training.py` | GPU training loop, early stopping, losses (CE, focal, sqrt weights). | `train_dl_model()` |
| `src/train.py` | GPU XGBoost baseline. | `train_xgb_gpu()` |
| `src/explainability.py` | SHAP with fingerprint check. | `run_shap_analysis()` |
| `audit_shortcuts.py` | Raw `"0"`/`"0.0"` audit and processed shortcut scan. | `--raw`, `--processed` |
| `run_clean_baselines.py` | Preprocess + scan + baselines per policy; loss study. | `--policy`, `--study losses` |
| `run_tune_cnn_gru.py` | Optuna tuning + 3 seeds. | `--trials`, `--prefix` |
| `run_final_studies.py` | Ablation, SHAP, fidelity, edge benchmark. | `main()` |
| `run_operating_points.py` / `audit_ceiling.py` | Trade-off and detection ceiling. | `main()` |
| `run_ciciot.py` | CICIoT2023 pipeline. | `--policy strict` |
| `make_figures.py` | Presentation figures from result files. | `main()` |
| `watch_dashboard.py` / `run_queue.py` | Live terminal dashboard; run queue that keeps Windows awake. | `main()` |

---

## PART 46 — TECHNOLOGY STACK

- **Python:** Primary language.
- **pandas / NumPy / PyArrow:** Data manipulation, Parquet I/O.
- **scikit-learn:** Splitting, scaling, one-hot encoding, shortcut-scan trees.
- **XGBoost (GPU):** Tree baseline.
- **PyTorch (CUDA):** MLP, CNN, GRU, CNN-GRU; FP16/INT8 benchmark.
- **Optuna:** Hyperparameter tuning.
- **SHAP:** Explainable AI.
- **kagglehub:** Dataset download (CICIoT2023 kept in the user cache).
- **rich / psutil:** Terminal dashboard and memory measurement.
- **FastAPI / Uvicorn:** Web dashboard backend.
- **Matplotlib:** Plots and figures.

---

## PART 47 — FACULTY QUESTIONS AND ANSWERS

**Basic**
- *What is IoT?* Internet of Things, a network of physical devices with sensors connected to the internet.
- *What is IDS?* Intrusion Detection System, software that monitors a network for malicious activity.
- *What is an anomaly?* A deviation from the established "normal" baseline behavior of the network.

**Dataset**
- *Why Edge-IIoTset?* Modern IoT/IIoT testbed with diverse attacks; widely used, which makes our shortcut finding relevant to other work.
- *Why a second dataset?* Edge-IIoTset is packet-level with no timing for attacks. CICIoT2023 is flow-level with rate and timing, and checks that our pipeline generalises.
- *What are the limitations?* Class imbalance; the `"0"`/`"0.0"` artifact; about a third of attack packets are identical to normal ones.

**ML & Deep Learning**
- *Why XGBoost as baseline?* It is the strongest standard model for tabular data; if DL cannot match it, we must say so.
- *Why CNN?* To extract local combinations from the feature vector.
- *What is a GRU?* Gated Recurrent Unit, a lighter alternative to LSTM for sequence processing.
- *What does hybrid mean?* The CNN's output feature maps are fed into the GRU within one network.

**XAI & Methodology**
- *Why explainability?* Security analysts need to know *why* an alarm fired, and it helps catch shortcuts.
- *Is SHAP causal?* Not for the real world. It explains what the *model* relied on. Our removal test shows those features really matter to the model (0.858 → 0.605).
- *Why split before scaling?* To prevent test statistics leaking into training.
- *Why macro-F1?* Accuracy is misleading when most data is normal; Macro-F1 weighs every class equally.
- *Did you tune on the test set?* No. Tuning and thresholds use validation only; test is used once.

**Critical**
- *How do you know your model learned attack behaviour rather than dataset artifacts?* We found the artifact that caused the fake 100% and removed it, removed per-packet identifiers, ran automatic single-feature and depth-3-tree shortcut scans (no giveaway left), checked the score is the same without MQTT, and SHAP (with a fidelity test) points to TCP behaviour. We do not claim this rules out every possible artifact.
- *Can this run on an IoT device?* The FP16 CNN-GRU is 159 KB and takes about 6.5 ms per packet on one laptop CPU thread. A test on a Raspberry-Pi-class device is planned.

---

## PART 48 — DIFFICULT FACULTY QUESTIONS (Weakness Exposure)

- **Q: Your CNN-GRU is already published. What is new?**
  *A:* The architecture is not new. Our contribution is the honest evaluation: we found and removed hidden shortcuts in two public datasets, measured the detection ceiling, proved SHAP fidelity, and compared fairly against simpler models over 3 seeds.
- **Q: Your first results said 100%. Why should we trust 0.858?**
  *A:* Because we found exactly why 100% happened (the `"0"`/`"0.0"` spelling, table with 1,613,798 vs 603,331 rows), removed it, and added automatic checks. 0.858 is stable over 3 seeds, independent of MQTT, and matches a measured data ceiling.
- **Q: You miss a third of attacks. Isn't that a bad IDS?**
  *A:* On this packet-level data, 31.9% of attack packets are identical to mostly-Normal packets, so no model can separate them; a perfect memoriser detects 67.5%, and we detect 65.3% (XGBoost 67.3%). Lowering the threshold detects 73.3% at 6.6% false alarms. With flow-level data (CICIoT2023) the CNN-GRU detects 90.4%.
- **Q: The MLP scores the same as your hybrid. Why propose a CNN-GRU?**
  *A:* Honestly, on per-packet features the hybrid gives no gain; all networks hit the same ceiling. The GRU needs a real time dimension to add value, so our next step is consecutive flow windows per device/connection. We report the ablation instead of hiding it.
- **Q: XGBoost beats you. Why deep learning at all?**
  *A:* XGBoost is slightly better (0.868 vs 0.858; 0.917 vs 0.890 on CICIoT2023) and faster on CPU (0.8 ms vs 6.5 ms). The networks are smaller (MLP 24 KB, FP16 CNN-GRU 159 KB vs 492 KB) and are the model family that can use real sequences. We present it as a trade-off, not a win.
- **Q: Where does your temporal information come from for the GRU?**
  *A:* It doesn't. Each row is processed alone; in the CNN-GRU the GRU steps over feature-map positions, not time. Attack rows in Edge-IIoTset have no timing at all. This is a stated limitation.
- **Q: Earlier you said focal loss was the solution to class imbalance. Was that wrong?**
  *A:* Yes. Those runs used only 5 epochs, before the shortcut fix. Over 3 seeds on clean data, sqrt class weights (0.337) beat focal loss (0.254) and cross-entropy (0.238).
- **Q: Earlier you said SHAP showed MQTT drives detection.**
  *A:* That SHAP plot stripped the `_0` / `_0.0` suffix and so showed the formatting shortcut under the name `mqtt.topic`. With exact labels and a fingerprint check, the final model's top features are TCP behaviour, and removing MQTT does not change the score.
- **Q: Is your dataset split realistic?**
  *A:* It is one random stratified split per dataset. A cross-dataset test (train on one, test on the other) is planned and listed as a limitation.

---

## PART 49 — KEY TERMS CHEAT SHEET

- **Data leakage / shortcut:** The model uses information that would not exist in the real world (an IP, a capture-specific spelling) instead of attack behaviour.
- **`"0"`/`"0.0"` artifact:** Edge-IIoTset writes empty fields as `"0"` in Normal and `"0.0"` in attack captures; after one-hot encoding one column gives away the label.
- **Canonicalisation:** Treating `"0"` and `"0.0"` as the same value before encoding.
- **Strict policy:** Removes 7 per-packet identifiers (sequence numbers, checksums, stream index).
- **Detection ceiling:** The best detection any model can reach when some attack and normal inputs are identical (67.5% on Edge-IIoTset).
- **Fidelity test:** Retrain without the top SHAP features and check the score drops (0.858 → 0.605) compared with random features (no drop).
- **Concept drift:** When network behavior changes over time, causing old models to fail.
- **Class imbalance / sqrt class weights:** Rare classes get larger loss weights, `sqrt(N / (K * count))`.
- **CNN (Conv1D):** Extracts local features using scanning filters.
- **GRU:** Processes data utilizing update/reset memory gates.
- **SHAP:** Attributes each prediction to feature contributions.
- **Macro-F1:** The average F1 score across all classes, preventing majority classes from skewing the metric.
- **FP16 / INT8:** Storing weights in 16-bit floats / 8-bit integers to shrink the model.

---

## PART 50 — “WHY DID WE CHOOSE THIS?”

| Decision | Our Choice | Why | Alternative | Why Not |
|---|---|---|---|---|
| Target | Binary as main result, multiclass reported | Binary survives the strict policy; multiclass needs timing | Multiclass only | Packet data cannot show rate |
| Features | Strict, no MQTT, canonicalised | Removes all shortcuts we found; score unchanged without MQTT | All features | Fake 100% accuracy |
| Baseline | GPU XGBoost | Strongest tabular model, fast, small | Random Forest | 1.6 GB file, all CPU cores, several GB RAM |
| DL Seq | GRU | Lighter parameter count | LSTM | Heavier, slower |
| Imbalance | Sqrt class weights | Best over 3 seeds (0.337) | Focal / raw weights | Focal 0.254; raw weights collapsed |
| XAI | SHAP GradientExplainer + fidelity test | Works on network gradients; fidelity proves it | LIME | Local surrogate only |
| Validation | Random Stratified, 3 seeds, validation-only thresholds | No usable timestamps | Chronological | Timestamps dropped as leakage |
| Compression | FP16 weights | Halves size, no loss | INT8 on GRU | Costs 7.6 points |

---

## PART 51 — LIMITATIONS (Honest Disclosure)

- **Hybrid gives no gain on per-packet features:** the CNN-GRU is not better than an MLP or XGBoost.
- **Detection ceiling:** about a third of Edge-IIoTset attack packets are indistinguishable from normal ones.
- **No real time dimension:** each row is processed alone; the GRU sees no time sequence.
- **Attack-type detection** needs flow/timing data; rare classes (Web, BruteForce in CICIoT2023) remain weak.
- **Evaluation design:** one random stratified split per dataset; no cross-dataset test yet.
- **Shortcuts:** we removed the ones our audits found; we do not claim the data is free of every artifact.
- **Deployment:** CPU benchmark on a laptop; not yet measured on physical IoT hardware.

---

## PART 52 — FUTURE WORK

| Gap | Plan |
|---|---|
| Missed attacks on packet data | Use flow-level data (CICIoT2023) as the main setting; choose thresholds by acceptable false-alarm rate |
| Hybrid gives no gain | Give the GRU real sequences: consecutive flow windows per device/connection, so it can learn how traffic changes over time |
| Weak rare classes | Sqrt class weights (already +0.06), targeted oversampling of Web/BruteForce |
| Generalisation | Cross-dataset test (train on one dataset, test on another with shared features) |
| Deployment | FP16 model on a Raspberry-Pi-class device; live demo via the dashboard predict API |
| Explanation stability | Compare SHAP rankings across the 3 seeds |

---

## PART 53 — FINAL FACULTY TALKING SCRIPT

**"What We Should Tell the Faculty Today"**

"Good morning. Our project is an intrusion detection system for IoT that should be accurate, explainable, and light enough for an IoT gateway. Our proposed model is a hybrid CNN-GRU with SHAP explanations, evaluated on Edge-IIoTset and CICIoT2023.

Our first models scored 100%. Instead of celebrating, we investigated. We found that Edge-IIoTset writes an empty field as `"0"` in normal traffic and `"0.0"` in attack traffic: 1.6 million normal rows use one spelling, 600 thousand attack rows the other, with no overlap. One column gave away the answer. Our earlier fix, removing MQTT, did not help, because the DNS column had the same artifact, and our old SHAP plots had hidden it by trimming feature names.

We fixed the spelling, removed seven per-packet identifiers, removed MQTT as a check, and added automatic shortcut scans and a data fingerprint check so this cannot happen silently again.

The honest result: our CNN-GRU reaches 0.858 Macro-F1, 90% accuracy, with 0.7% false alarms, stable over three seeds. XGBoost reaches 0.868. We then asked why it stops there and measured it: 31.9% of attack packets are identical to normal packets, so even a perfect memoriser detects only 67.5%. Our model detects 65.3%; it is at the ceiling.

We were also honest about the hybrid: an MLP, a CNN and a GRU all reach the same 0.858. We proved our SHAP explanations are faithful: removing the top five SHAP features, which are TCP flags and connection resets, drops the score to 0.605, while removing five random features changes nothing. For the edge, the FP16 model is 159 KB with no loss and takes about 6.5 ms per packet on one CPU thread.

Finally, on CICIoT2023, which has rate and timing, we found and removed a second hidden shortcut, the window size. There the CNN-GRU reaches 0.890 and category detection doubles from 0.34 to 0.68.

Next, we will give the GRU real sequences of flow windows, test across datasets, improve rare classes, and deploy the FP16 model on a Raspberry-Pi-class device."

---

## PART 54 — FACULTY MEETING QUICK REVISION

**FACULTY MEETING CHEAT SHEET (1-Page)**
- **Project:** Explainable Hybrid DL IDS for IoT Networks (accurate, explainable, lightweight).
- **Datasets:** Edge-IIoTset (2,219,201 packets, Normal + 14 attacks); CICIoT2023 (flow-level, 33 attacks / 7 categories + Benign).
- **First result (fake):** 100% on binary, caused by `"0"` (Normal) vs `"0.0"` (Attack) empty fields.
- **Fixes:** canonicalise `"0"` = `"0.0"`; remove 7 per-packet identifiers; remove MQTT as a check; shortcut scans; fingerprint guard; 3 seeds; validation-only thresholds.
- **Split:** 70/15/15 stratified. *Scaled AFTER splitting.*
- **Edge-IIoTset binary:** CNN-GRU 0.858 (90.0% acc, 0.7% false alarms, 65.3% detected); XGBoost 0.868.
- **Ceiling:** 31.9% of attacks identical to Normal; perfect memoriser 67.5%; we are at the ceiling.
- **Ablation:** MLP = 1D-CNN = GRU = CNN-GRU = 0.858. Hybrid gives no gain on packet data.
- **XAI:** SHAP top features `tcp.flags`, `tcp.connection.rst`, `tcp.flags.ack`, `tcp.len`, `tcp.connection.fin`. Fidelity: top-5 removed 0.858 → 0.605; random-5 removed → 0.858.
- **Edge:** FP16 CNN-GRU 159 KB, 6.5 ms/packet, no loss; INT8 on GRU −7.6 points; XGBoost 0.8 ms.
- **Multiclass:** Edge-IIoTset weak (XGBoost 0.429, CNN-GRU 0.337, sqrt weights best). CICIoT2023 categories XGBoost 0.752, CNN-GRU 0.675.
- **CICIoT2023 binary:** XGBoost 0.917, CNN-GRU 0.890; second shortcut (window size) found and removed.
- **Contribution:** shortcuts found in two datasets; measured ceiling; proven SHAP fidelity; honest, reproducible evaluation.
- **Limitations:** hybrid no gain on packets; no real time dimension; single split; no device test yet.
- **Next Milestone:** real flow-window sequences for the GRU, cross-dataset test, Raspberry-Pi deployment.
