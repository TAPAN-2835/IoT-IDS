# The Complete Guide to Our IoT-IDS Research Project
*A plain-language explanation of what we built, what we discovered, and what comes next.
All numbers come from [FINAL_RESULTS.md](FINAL_RESULTS.md).*

---

## 1. The Big Picture: What Are We Building?

IoT devices (smart sensors, cameras, industrial controllers) are easy to attack and too weak to run
heavy security software. We are building an **Intrusion Detection System (IDS)**: a program that
watches network traffic and raises an alarm when it sees an attack. We want it to be:

1. **Accurate**: catch attacks without crying wolf.
2. **Explainable**: say *why* it raised the alarm.
3. **Lightweight**: small and fast enough to run on an IoT gateway.

Our model is a **CNN-GRU**:
- **CNN (Convolutional Neural Network):** finds patterns among the features of a record.
- **GRU (Gated Recurrent Unit):** a lightweight network designed for sequences.

To explain decisions we use **SHAP**, a tool that tells us how much each feature pushed the model
towards "attack" or "normal".

We use two public datasets: **Edge-IIoTset** (2.2 million network packets, normal traffic plus 14
attack types) and **CICIoT2023** (flow-level summaries of traffic, 33 attacks grouped into 7
categories plus benign traffic).

---

## 2. The Plot Twist: The 100% Accuracy Was Fake

Our first models (Random Forest, CNN, GRU, CNN-GRU) all scored **100%** on "Normal vs Attack". In
machine learning, a perfect score is almost always a warning sign: the model has probably found a
shortcut. So we investigated instead of celebrating.

### What the shortcut was

When a packet has no DNS or MQTT information, the dataset fills the empty field with a zero. But it
wrote that zero **differently** depending on where the traffic came from:

| How the empty `dns.qry.name.len` field was written | Normal packets | Attack packets |
|---|---|---|
| `"0"` | 1,613,798 | 0 |
| `"0.0"` | 0 | 603,331 |

To a human, `"0"` and `"0.0"` are the same number. To our preprocessing they were two different
words, so they became two different columns. The model simply learned "if it says `0.0`, it's an
attack". That is not intrusion detection; it is reading the answer off the formatting.

### Why our first fix did not work

Every MQTT packet in the dataset is Normal, and the old SHAP plots pointed at MQTT fields, so we
first removed all MQTT features. The score stayed at 100%, because the DNS column carried exactly
the same `"0"`/`"0.0"` trick. The old SHAP plots had hidden this by cutting the `_0` / `_0.0` ending
off the feature names, so both spellings looked like one feature.

### The real fix

1. **Treat `"0"` and `"0.0"` as the same value** before encoding.
2. **Remove 7 per-packet identifiers** (sequence numbers, checksums, stream index). They tell you
   *which capture file* a packet came from, not how it behaves.
3. **Remove all MQTT fields as a check.** The score is the same with or without them (0.8579 vs
   0.8575), so the model no longer leans on "MQTT means normal".
4. **Add automatic alarms:** a scan that checks whether any single feature still gives away the
   answer, and a "fingerprint" check so training and SHAP refuse to run on the wrong data.

---

## 3. The Honest Results

### Normal vs Attack on Edge-IIoTset

| Model | Macro-F1 | Accuracy | False alarms | Attacks caught |
|---|---|---|---|---|
| Before the fix (fake) | 1.000 | 100% | 0% | 100% |
| **Our CNN-GRU** | **0.858** | **90.0%** | **0.7%** | **65.3%** |
| XGBoost (tree model) | 0.868 | 90.7% | 0.6% | 67.3% |

The model rarely raises a false alarm, but it misses about a third of attacks. Is that a weak
model? We checked.

### Why it stops at about two-thirds: the ceiling

Many packets in this dataset look *exactly* the same. We found that **31.9% of attack packets have
exactly the same features as packets that are mostly Normal**. No model, however clever, can tell
identical inputs apart. Even a "perfect memoriser" would catch only **67.5%** of attacks. Our models
catch 65.3% and 67.3%: they are already at the ceiling. Tuning did not help (20 automatic tuning
trials gave the same test score), and three different random seeds give 0.8584, 0.8583 and 0.8582.

### Do we need the hybrid?

We compared it fairly against simpler networks. A tiny MLP (24 KB), a CNN, a GRU and our CNN-GRU
**all reach 0.858**. On single-packet data, the hybrid gives no extra accuracy. We say this openly.

### Can we trust the explanations?

SHAP now points at real TCP behaviour: `tcp.flags`, `tcp.connection.rst`, `tcp.flags.ack`,
`tcp.len`, `tcp.connection.fin`. To check SHAP is telling the truth, we retrained the model
**without its top-5 SHAP features**: Macro-F1 fell from 0.858 to **0.605**. Removing 5 *random*
features changed nothing (0.858). So SHAP really identifies what the model depends on.

### Is it small and fast?

On one CPU thread the CNN-GRU takes about 6.5 ms per packet. Storing its weights at half precision
(FP16) shrinks it from 315 KB to **159 KB** with no loss in accuracy. Squeezing the GRU to 8-bit
integers makes it 88 KB but costs 7.6 points of Macro-F1. XGBoost is faster on a CPU (0.8 ms).

### Naming the attack type

Telling *which* of the 14 attacks is happening is hard on single packets (XGBoost 0.429, CNN-GRU
0.337). Floods and scans are defined by how *fast* packets arrive, and one packet cannot show speed;
the dataset has no timing for attack packets. For the imbalance between common and rare attacks,
**square-root class weights** worked best over 3 seeds (0.337), ahead of focal loss (0.254) and plain
cross-entropy (0.238).

---

## 4. The Second Dataset: CICIoT2023

CICIoT2023 describes short windows of traffic (flows) and includes **rate and timing**, exactly what
Edge-IIoTset lacks. We ran the same honest pipeline and found **another hidden shortcut**: the
`packet_count` column is just the size of the measurement window (10 for Benign/Recon, 100 for
DDoS/DoS/Mirai). We removed it, along with the `source_file` column (which names the capture, i.e.
the answer), and removed 87,417 duplicate rows before splitting.

| Task | XGBoost | CNN-GRU |
|---|---|---|
| Benign vs Attack | 0.917 | 0.890 |
| 8 categories | 0.752 | 0.675 |

With timing information the CNN-GRU does much better than on single packets (binary 0.86 → 0.89,
categories 0.34 → 0.68), and SHAP now lists `rate` among its top features, as you would expect.

---

## 5. What Our Contribution Is

1. We found and removed hidden shortcuts in **two** public IoT datasets that make models look
   perfect.
2. We **measured the ceiling** of the packet-level dataset and showed our model reaches it.
3. We **proved** the explanations are faithful, instead of only showing SHAP plots.
4. Everything is honest and reproducible: 3 seeds, fair comparison with simpler models, thresholds
   chosen without looking at the test set, and a tested edge deployment.

---

## 6. Limitations (we say them first)

- On single-packet features, the CNN-GRU is not better than an MLP or XGBoost.
- About a third of Edge-IIoTset attack packets are indistinguishable from normal ones.
- The GRU sees no real time dimension: each record is processed on its own.
- Naming the attack type needs flow/timing data; rare classes (Web, BruteForce) remain weak.
- One train/test split per dataset; no cross-dataset test yet.

---

## 7. What Comes Next

1. **Use flow data as the main setting** (CICIoT2023) and pick the alarm threshold by how many false
   alarms are acceptable.
2. **Give the GRU real sequences**: consecutive flow windows per device or connection, so it can
   learn how traffic changes over time. This is where a hybrid should finally help.
3. **Help rare attacks**: keep square-root class weights, add targeted oversampling of Web and
   BruteForce attacks.
4. **Cross-dataset test**: train on one dataset, test on the other.
5. **Deploy** the FP16 model on a Raspberry-Pi-class device and demo it live through the dashboard.

To reproduce everything, see section 9 of [FINAL_RESULTS.md](FINAL_RESULTS.md) or the "How to run"
section of the [README](../README.md).
