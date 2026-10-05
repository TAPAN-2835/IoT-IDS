# Task 2: Make the project research-paper ready

**Goal:** turn the project into a paper that a reviewer would take seriously: a clear contribution,
honest claims, every number traceable to a result file, verified references, and a repository that
reproduces every table with one command.

Repository: https://github.com/TAPAN-2835/IoT-IDS · Read first: `docs/FINAL_RESULTS.md`,
`docs/VIVA_QA.md`, `docs/TASK1_REPLICATE_AND_PROVE.md` (the experiments that feed the paper) and
the friend's review `IoT_IDS_Deep_Dive_Review.docx` (sections 0 and 9 especially).

---

## 1. What the paper is about (the contribution)

The paper is **not** "a new CNN-GRU IDS". The CNN-GRU ties with an MLP and loses to XGBoost; claiming
otherwise would be rejected. The contribution is the **audit**:

1. A leakage audit protocol for IoT IDS datasets (representation scan, removal of per-packet
   identifiers, single-feature probes, data fingerprints) that found a `"0"` vs `"0.0"` encoding
   shortcut in Edge-IIoTset and a window-size shortcut (`packet_count`) in CICIoT2023.
2. A measured detection ceiling for Edge-IIoTset (31.9% of attack packets indistinguishable from
   normal ones), explained as label noise from file-level labelling of web-attack captures.
3. A retraining-based test of explanation fidelity with random controls, compared across explainers
   and against the zero-at-inference test used in prior work.
4. An honest benchmark: multiple seeds, validation-only thresholds, ablation against simpler models,
   CPU latency/size trade-offs, two datasets.

**Working title:** *Auditing Shortcut Learning in IoT Intrusion Detection: Leakage, Detection Ceilings
and Faithful Explanations*

**Novelty statement (conservative, from the review):** "Reported near-perfect accuracy on IoT/IIoT
intrusion benchmarks is rarely accompanied by an audit of non-behavioural shortcuts. We present a
leakage audit protocol and apply it to Edge-IIoTset and CICIoT2023, identifying representation and
capture-level artifacts, quantifying the irreducible-error ceiling under a cleaned feature policy,
and showing how reported accuracy changes once shortcuts are removed. Within this controlled setting
we compare per-packet models against strong tabular baselines, evaluate SHAP fidelity via
retraining-based removal, and report edge deployment trade-offs."

---

## 2. Claims: allowed, softened, forbidden

| Do not write | Write instead | Why |
|---|---|---|
| "mathematically proved" | "empirically demonstrated" | It is an experiment, not a proof |
| "the 100% commonly reported on this dataset comes from shortcuts" | "a standard pipeline on this dataset reaches near-perfect accuracy through shortcuts" (and cite the Ferrag replication, Task 1 B1, once it exists) | We can only speak about pipelines we re-ran |
| "leakage-free" | "with every shortcut our audit detected removed" | Unknown shortcuts may remain |
| "our hybrid outperforms" | "all networks reach the same ceiling; XGBoost is slightly better" | Our own ablation |
| "theoretical ceiling" | "empirical ceiling under the cleaned feature set and this split" | It depends on features and duplicates |
| "runs on IoT devices" | "159 KB, 6.5 ms per packet on one CPU thread; device test pending" | No hardware test yet |
| "paper X is wrong" | "paper X does not report a check for Y" | Professional and defensible |

Compare scores only on the same dataset and task. Of the five reference papers, only Ferrag et al.
and Munilla & Khammas use Edge-IIoTset.

---

## 3. Paper structure (IEEE conference format, LaTeX, in `paper/`)

| Section | Content | Inputs |
|---|---|---|
| Abstract (≤ 200 words) | Problem, audit, 3 key numbers, takeaway | `paper/numbers.tex` |
| 1. Introduction | Why near-perfect IDS scores are suspicious; research questions; 4 contribution bullets (section 1) | Review §9 |
| 2. Related work | (a) IoT IDS datasets; (b) deep learning for IoT IDS; (c) XAI for IDS (Munilla, Sharma, Wang, Udurume); (d) ML pitfalls in security and shortcut learning. End with a comparison table (section 6) | Section 5 references |
| 3. Datasets | Edge-IIoTset (testbed, 10 single-sensor Normal captures, one file per attack, file-level labels); CICIoT2023 (sample, de-duplication, window features); UNSW-NB15 if Task 1 B4 is included | Ferrag paper; `src/ciciot.py` |
| 4. Audit protocol | Representation scan, identifier removal, MQTT check, single-feature and depth-3 probes, detection ceiling, fingerprints | `audit_shortcuts.py`, `audit_ceiling.py`, `src/preprocessing.py` |
| 5. Models and training | CNN-GRU (figure), MLP, 1D-CNN, GRU, XGBoost; Optuna; seeds; validation-only thresholds; hardware | `src/models.py`, `src/training.py`, `run_tune_cnn_gru.py` |
| 6. Explanation and fidelity | SHAP GradientExplainer (background 200, 1,000 explained packets); retraining fidelity; comparisons from Task 1 B2, B5, B6, B7 | `src/explainability.py`, `run_final_studies.py` |
| 7. Results | 7.1 Shortcuts and the replication (B1); 7.2 Honest performance; 7.3 Ceiling and label noise; 7.4 Ablation; 7.5 Fidelity; 7.6 CICIoT2023; 7.7 UNSW-NB15 (B4); 7.8 Edge trade-offs (B10); 7.9 Robustness (B11) | `results/paper/*`, `paper/figures/*` |
| 8. Discussion | What near-perfect scores mean; implications for dataset builders (label background traffic, publish capture IDs) | |
| 9. Threats to validity | Random split (99.5% of test rows share a pattern with training); duplicates; approximate SHAP; single dataset version; no device test | Review §§2, 6, 7 |
| 10. Conclusion and future work | Sequences for the GRU with a fair windowed baseline; relabelling; leave-one-attack-out; Raspberry Pi | |

Length target: 8 pages + references (conference); an extended version can go to a journal later.

---

## 4. Numbers, tables and figures

- **No number is typed by hand.** Write `paper/build_numbers.py`: it reads `results/` and writes
  `paper/numbers.tex` with one macro per number (for example `\CnnGruFone`, `\XgbFone`,
  `\CeilingDetect`, `\HiddenPct`, `\FidTopFive`, `\FerragRepAcc`). A missing result prints a red
  `[TBD]` so the paper always compiles. The paper starts with `\input{numbers}`.
- **Tables:** (1) dataset summary; (2) shortcuts found and fixes; (3) Ferrag reported / replicated /
  cleaned (Task 1 B1); (4) main results with mean ± SD; (5) ablation with size and latency;
  (6) fidelity comparison; (7) CICIoT2023 per category; (8) related-work comparison (section 6).
- **Figures:** reuse `results/figures/fig1`–`fig5` (fake vs honest, SHAP fidelity, model comparison,
  datasets, ceiling); add the CNN-GRU architecture diagram, the label-noise breakdown by attack type,
  the fidelity-methods chart (B5) and the Pareto chart (B10). Vector PDF or 300 dpi PNG; readable in
  greyscale; captions state dataset, split and metric.

---

## 5. References

Cite only papers someone on the team has opened; check every DOI. Starting list:

| Reference | Used for |
|---|---|
| Ferrag et al., "Edge-IIoTset…", IEEE Access, 2022 | Dataset; replication |
| Neto et al., "CICIoT2023…", Sensors, 2023 | Second dataset |
| Moustafa & Slay, "UNSW-NB15…", MilCIS, 2015 | If B4 is included |
| Munilla & Khammas, Sensors, 2026, doi 10.3390/s26102924 | XAI on Edge-IIoT |
| Sharma et al., Scientific Reports, 2025, doi 10.1038/s41598-025-23750-0 | Self-attention XAI IDS |
| Wang et al., Internet of Things 33 (2025) 101714 | SHAP+LIME interpretable models |
| Udurume, Shakhov & Koo, Scientific Reports, 2026, doi 10.1038/s41598-025-34334-3 | SHAP 1D-CNN, zero-masking fidelity |
| Lundberg & Lee, "A Unified Approach to Interpreting Model Predictions", NeurIPS 2017 | SHAP |
| Ribeiro, Singh & Guestrin, "Why Should I Trust You?", KDD 2016 | LIME |
| Chen & Guestrin, "XGBoost", KDD 2016 | Baseline |
| Cho et al., 2014 (GRU) | Model |
| Akiba et al., "Optuna", KDD 2019 | Tuning |
| Arp et al., "Dos and Don'ts of Machine Learning in Computer Security", USENIX Security 2022 | ML pitfalls in security |
| Engelen, Rimmer & Joosen, CICIDS2017 case study, IEEE SPW 2021 | Dataset artifacts |
| Kapoor & Narayanan, "Leakage and the reproducibility crisis…", Patterns 2023 | Leakage taxonomy |
| Geirhos et al., "Shortcut learning in deep neural networks", Nature Machine Intelligence 2020 | Shortcut learning |

Keep `paper/references.bib` complete (authors, title, venue, year, DOI). Add more only after reading them.

---

## 6. Related-work comparison table (verified facts from the five papers)

| Paper | Dataset(s) | Reported | Shortcut / leakage check | Explanation evaluation | Notes found when reading |
|---|---|---|---|---|---|
| Ferrag 2022 | Edge-IIoTset | 99.99% binary, 94.67% 15-class (DNN) | No | n/a | File-level labels, random split, keeps `tcp.seq`/`tcp.ack`, encoding described three ways |
| Munilla 2026 | Edge-IIoT, BoT-IoT, N-BaIoT | No accuracy reported | No (all 63 features) | Sparsity, completeness, robustness under DeepFool | Top Edge-IIoT features include MQTT header data |
| Sharma 2025 | BoT-IoT, N-BaIoT, UNSW-NB15 | 97.9–99.6% | No | ~85% SHAP–LIME alignment | Resampling apparently before split; module named five ways; caption mismatch |
| Wang 2025 | UNSW-NB15, CICIDS2017 | up to 99.8% | No | None (8 single-sample LIME plots) | Tables 4 and 5 identical; destination port used |
| Udurume 2026 | UNSW-NB15, WUSTL-IIoT | 97.5% F1 | No | Zeroing at inference, Jaccard 0.84–0.89, top-k pruning | `stcpb` (sequence number) in top 10 |
| **Ours** | Edge-IIoTset, CICIoT2023 (+UNSW-NB15) | 0.858 (honest) | **Yes, two shortcuts found** | **Retraining with random controls, across explainers** | Ceiling and label noise measured |

Present this as "what each paper checks", not as a ranking of accuracy.

---

## 7. Reproducibility package (repository)

- [ ] `reproduce_paper.py`: one command that runs Task 1 (A and B) in order through `run_queue.py`, then `make_figures.py` and `paper/build_numbers.py`
- [ ] Pinned environment: `requirements-lock.txt` with exact versions (Python 3.13, torch 2.11 CUDA 12.8, xgboost 3.0.5, shap 0.52, scikit-learn 1.7.1, lightgbm 4.6, lime) and hardware used (RTX 4060 Laptop GPU)
- [ ] Data availability: datasets are public and downloaded by `download_dataset.py` (Edge-IIoTset) and `run_ciciot.py` (CICIoT2023, CC BY-NC-SA 4.0 mirror); no data in the repository
- [ ] Code availability statement and a tagged release (for example `v1.0-paper`) once results are final
- [ ] `CITATION.cff` and a `LICENSE` file (the team must choose the licence)
- [ ] Legacy scripts that produced the leaky E01–E08 results moved to `legacy/` or clearly marked
- [ ] README section "Reproducing the paper"
- [ ] Unit tests for the audit functions (canonicalisation, fingerprint guard, ceiling computation)

---

## 8. Before submission: review checklist

- [ ] Every number in the PDF comes from `numbers.tex`; `[TBD]` appears nowhere
- [ ] Every claim in section 2's table is phrased the allowed way
- [ ] Every reference opened and its DOI checked
- [ ] Every figure readable in greyscale and at column width
- [ ] Threats-to-validity section names the random split, duplicates and SHAP approximation
- [ ] A team member who did not write a section re-reads it against the code
- [ ] Mentor review, then choose the venue together (IEEE conference first, journal extension later)
- [ ] Author names, affiliations and acknowledgements filled in
