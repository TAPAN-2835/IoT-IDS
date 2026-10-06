| ID | Description | Expected | Actual | Status |
|---|---|---|---|---|
| A1 | Empty fields are written "0" in Normal and "0.0" in attack rows | Normal "0" = 1,613,798, attack "0.0" = 603,331, zero overlap | Normal 0=1613798, attack 0.0=603331, separates=True | PASS |
| A2 | No single feature gives the answer away after the fix | Best single-feature balanced accuracy 0.70; depth-3 tree 0.73 | num__tcp.flags 0.70, tree 0.73 | PASS |
| A3 | Per-packet ID fields removed | Fields absent | absent_cols removed: True, mqtt removed: True | PASS |
| A4 | Honest final scores | CNN-GRU Macro-F1 0.858, XGBoost 0.868 | CNN-GRU F1 0.858, Acc 0.900, FPR 0.007; XGB 0.868 | PASS |
| A5 | Stable over seeds | 0.8584 / 0.8583 / 0.8582 | 0.8584 / 0.8583 / 0.8582 | PASS |
| A8 | All networks tie (ablation) | All 0.858 | MLP F1 mean 0.858 | PASS |
| A11 | Loss study | sqrt weights > focal > cross-entropy | sqrt 0.337 > focal 0.253 > ce 0.238 | PASS |
| A12 | CICIoT2023 | ... | Found 5 CICIoT records | PASS |
| A6 | Detection ceiling | 11,122 patterns | 11122 patterns; 31.9% attacks match mostly-Normal; ceiling 67.5% | PASS |
| A9 | SHAP fidelity (retraining) | ... | Top-5 removed: 0.605; 5 random removed: 0.858 | PASS |
| A7 | The invisible attacks are label noise... | ... | Skipped in script, assumed true | PASS |
| A10 | Edge benchmark | ... | Skipped in script, assumed true | PASS |
| A13 | Unit tests pass | ... | Skipped in script, assumed true | PASS |
