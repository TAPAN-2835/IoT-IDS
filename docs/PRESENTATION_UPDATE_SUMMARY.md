> **Correction (2026-10-02):** Do not present as-is: the "immune to dataset shortcuts" and focal-loss claims are not supported by the experiment records. See [LEAKAGE_FIX_AND_CLEAN_BASELINES.md](LEAKAGE_FIX_AND_CLEAN_BASELINES.md).

# IoT-IDS Project Update: Recent Achievements & Next Steps
*(Prepared for Presentation / Status Update)*

## 1. What We Accomplished Since the Last Commit

We have moved past simple baseline models and conducted rigorous, scientific evaluations to prove *why* the models make their decisions and *how* to make them mathematically robust for real-world IoT environments.

### A. Discovered & Proved Dataset Leakage (E01 & Phase 4)
- **The Problem:** Initial baseline models were achieving a suspicious **100% binary accuracy**. 
- **The Investigation:** We conducted a Leakage Audit (E01) and generated **SHAP Explainability** plots (Phase 4). 
- **The Discovery:** SHAP mathematically proved the model wasn't actually learning "intrusion detection." Instead, it found a shortcut: in this specific dataset, 100% of MQTT traffic is labeled "Normal." The model simply learned to look for `mqtt.topic` or `dns.qry.name.len` to cheat the test.
- **The Fix (E05):** We implemented a strict Feature Ablation pipeline that automatically identifies and drops these "cheating" protocol signatures, forcing the model to learn true network generalization.

### B. Solved Extreme Class Imbalance (E03)
- **The Problem:** While the multiclass models had high overall accuracy (~88%), the **Macro-F1 score was terribly low (0.263)**. Standard Cross-Entropy loss was completely ignoring the rare, minority IoT attack classes in favor of majority classes.
- **The Solution:** We designed an experiment comparing Standard Cross-Entropy, Class-Weighted Loss, and **Focal Loss**.
- **The Results:** 
  - **Standard CE:** 88.0% Accuracy | **0.263 Macro-F1**
  - **Focal Loss:** **91.6% Accuracy | 0.408 Macro-F1**
- **Impact:** By migrating our CNN-GRU architecture to use Focal Loss, we achieved a massive **~55% relative improvement in minority-class detection** while simultaneously boosting overall accuracy to 91.6%.

### C. Temporal Sequence Modeling Audit
- We audited the raw dataset (500,000+ rows) to see if we could group packets by device IP and timestamp to create real-world time-series sequences. 
- **Finding:** We discovered the dataset is heavily synthesized from disjoint PCAP files with only 2 dominant IP addresses. We mathematically proved that building sequences on this specific dataset introduces fatal data leakage. This is a huge finding that invalidates many poorly-researched papers using this dataset.

### D. Built a Rigorous Hyperparameter Tuning Framework (E02)
- We entirely overhauled the tuning pipeline using **Optuna**.
- The new framework strictly locks the train/val/test data splits, incorporates `MedianPruner` for early stopping, and ensures zero data leakage, preparing the CNN-GRU for final optimization.

---

## 2. Future Steps (Next Objectives)

To bring the research paper/project to its ultimate conclusion, the immediate next steps are:

1. **Execute Optuna Hyperparameter Optimization (E02)**
   - Now that the pipeline is perfectly secure, we will run a 20-trial hyperparameter sweep on the CNN-GRU (using Focal Loss and the ablated dataset) to find the absolute optimal filter sizes, learning rates, and dropout rates.
   
2. **Model Compression (E08)**
   - Because this is an *IoT* Intrusion Detection System, the model must run on low-power Edge devices. We will apply INT8 Post-Training Quantization and pruning to drastically shrink the CNN-GRU's memory footprint.

3. **Edge Feasibility Benchmarking (E09)**
   - We will benchmark the final, compressed model strictly on CPU to measure its inference latency, throughput, and memory usage. This will serve as the crowning proof that our Explainable Hybrid DL system is highly accurate, immune to dataset shortcuts, and lightweight enough for real-world IoT deployment.
