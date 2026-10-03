# Viva Q&A: the hardest questions, with honest answers

*Prepared for the presentation. All numbers are from `docs/FINAL_RESULTS.md`. Rule for answering:
short, honest, then point to the evidence. Never claim more than the evidence shows.*

---

### 1. "Your first results were 100%. Why should we trust anything now?"
Because we treated the 100% as a red flag and found its cause: empty fields are written `"0"` in
Normal traffic and `"0.0"` in attack traffic (1,613,798 vs 603,331 rows, zero overlap). After
removing that and other shortcuts, an automatic scan shows no single feature can predict the label
(best single feature 0.72 → after strict features 0.70). Our honest score is 0.858.
**Evidence:** `results/audit/empty_token_audit.csv`, `results/figures/fig1_fake_vs_honest.png`.

### 2. "How do you know the model isn't still cheating somewhere?"
Four checks: (1) shortcut scans after every preprocessing run; (2) removing MQTT changes nothing
(0.8575 vs 0.8579); (3) results are identical across 3 seeds; (4) the SHAP fidelity test shows the
model depends on real TCP behaviour (`tcp.flags`, RST/ACK/FIN, `tcp.len`). We do not claim
"leakage-free"; we claim every shortcut we could detect is removed and tested.

### 3. "35% of attacks are missed. Isn't that bad for an IDS?"
Yes, and we measured why. The 1.55M training packets contain only 11,122 distinct feature patterns,
and 31.9% of attack packets look *exactly* like mostly-normal packets. Even a perfect lookup
classifier detects only 67.5%; ours detects 65.3%. The limit is the data, not the model. On
CICIoT2023 (flow data) the miss rate falls to about 7–10%.
**Evidence:** `results/figures/fig5_detection_ceiling.png`, `results/audit/detection_ceiling.json`.

### 4. "Can't you just lower the threshold to catch more attacks?"
We tested it, choosing thresholds on validation data only. On Edge-IIoTset, detection rises to
73.3% but false alarms jump to 6.6%, because the remaining attacks are indistinguishable from normal
packets. On CICIoT2023 the trade-off is smooth: at ≤ 1% false alarms the CNN-GRU still detects 82.3%.
**Evidence:** `results/operating_points.csv`.

### 5. "Your CNN-GRU is no better than a simple MLP. Why use it?"
Honest answer: on per-packet features it isn't better. MLP, 1D-CNN, GRU and CNN-GRU all reach
0.858 (3 seeds), because all hit the same data ceiling. The value of the hybrid is for *sequences*:
a GRU is designed to learn how traffic changes over time, and per-packet rows give it no time
dimension. That is exactly our next step. We report the ablation openly instead of hiding it.
**Evidence:** `results/figures/fig3_model_comparison.png`.

### 6. "XGBoost is slightly better (0.868) and faster. Why not just use XGBoost?"
For a pure accuracy/speed contest on a CPU, XGBoost wins and we say so. The neural model is smaller
(159 KB in FP16 vs 492 KB; an MLP is 24 KB), works with gradient-based SHAP, and is the basis for
sequence modelling. We include XGBoost as a strong baseline precisely so the comparison is fair.

### 7. "Is SHAP trustworthy? Plots can look convincing and still be wrong."
That is why we tested it. Retraining without the top-5 SHAP features drops Macro-F1 from 0.858 to
0.605; removing 5 random features changes nothing (0.858). So SHAP identifies what the model
really depends on. We also found and fixed an earlier SHAP bug where a model was explained with the
wrong input data; the pipeline now refuses to do that.
**Evidence:** `results/figures/fig2_shap_fidelity.png`, `results/shap_fidelity.json`.

### 8. "Earlier you said MQTT features were the key. What changed?"
That earlier SHAP view was misleading: names were shortened so `mqtt.topic_0` and `mqtt.topic_0.0`
looked like one feature, hiding that the real signal was the `"0"`/`"0.0"` spelling. After the fix,
removing MQTT entirely makes no difference.

### 9. "Why is attack-type classification so weak on Edge-IIoTset?"
Floods and scans are defined by *rate* (packets per second). A single packet cannot show rate, and
the dataset records no timing for attack packets (`udp.time_delta` is 0 for every attack row).
Best result: XGBoost 0.43, CNN-GRU 0.34 Macro-F1. With flow data (CICIoT2023) the CNN-GRU reaches
0.68 on 8 categories, and SHAP ranks `rate` among the top features.

### 10. "Why did you switch datasets? Isn't that cherry-picking?"
We did not switch; we added a second dataset to test whether the limitation is the model or the
data. Edge-IIoTset remains the main dataset. CICIoT2023 went through the same honest pipeline, and
we found and removed a second shortcut there too (`packet_count` is the window size: 10 for
Benign/Recon, 100 for floods). Results held after removing it (0.9166 → 0.9165).

### 11. "Did focal loss help with class imbalance? Your earlier slides said so."
No. The earlier comparison used undertrained models (5 epochs). Re-done at 15 epochs over 3 seeds:
square-root class weights 0.337 > focal 0.254 > cross-entropy 0.238. Raw inverse-frequency weights
made training collapse; the square root fixes that.

### 12. "Can this actually run on an IoT device?"
Measured on one CPU thread: the CNN-GRU takes 6.4 ms per packet; in FP16 it is 159 KB with no
accuracy loss. INT8 on the GRU loses 7.6 points, so we don't use it. We have not yet run it on
real gateway hardware; a Raspberry-Pi-class test is in our next steps. The live demo shows real-time
classification with explanations.

### 13. "What exactly is novel in your project?"
Not the CNN-GRU itself. Our contributions: (1) finding and removing hidden shortcuts in two public
IoT datasets that make models look perfect; (2) measuring the detection ceiling of packet-level
Edge-IIoTset and showing our model reaches it; (3) proving SHAP fidelity with a removal test; (4) an
honest, reproducible evaluation (3 seeds, fair ablation, validation-only thresholds, edge benchmark).

### 14. "How did you split the data? Could the test set leak into training?"
Stratified random 70/15/15 with seed 42. The scaler and encoder are fitted on the training split
only. Thresholds and tuning use validation only; the test split is used once for reporting. On
CICIoT2023 we removed 87,417 exact duplicates *before* splitting so copies cannot land in both.
Limitation: we have not yet done a cross-dataset or time-based split.

### 15. "What would you do with three more months?"
(1) Build real time sequences for the GRU from consecutive flow windows, so the hybrid can use time;
(2) cross-dataset testing (train on one dataset, test on another); (3) deploy the FP16 model on a
Raspberry Pi and measure live throughput; (4) improve rare attack classes (Web, BruteForce) with
targeted oversampling.

---

## Phrases to avoid

* "Our model is 100% accurate" / "leakage-free" / "real-time on IoT devices" (not tested on hardware)
* "The hybrid outperforms other models" (it ties them)
* "SHAP shows MQTT drives detection" (that was the artifact)

## One-sentence summary

"We found that the 100% accuracy commonly reported on this dataset comes from data shortcuts. We
removed them, proved our model's explanations are faithful, measured the ceiling of what packet data
allows, and showed that flow-level data is needed to identify attack types reliably."
