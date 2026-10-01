# E02 PRE-RUN AUDIT

## 1. Verified Facts and File Evidence
* **Data Pipeline (`run_E02_hyperparameter_tuning.py`):**
  * **Status:** FIXED. The script originally loaded a random data subset *inside* the objective function, which would have introduced massive trial-to-trial variance. The script has been rewritten to load and subset the data *once* before the trials.
  * **Test Set:** Protected. `X_test.parquet` is never loaded or referenced in the tuning script.
  * **Pruning:** `Optuna.pruners.MedianPruner` is correctly implemented. The objective function now reports the validation Macro-F1 at each epoch and prunes unpromising trials.
  * **Early Stopping:** Added explicit patience tracking inside the objective to return the `best_macro_f1` rather than the final epoch's score.
  * **Seeds:** Fixed using `cfg.set_seeds()`.
* **Loss Functions (`src/training.py`):**
  * The `FocalLoss` implementation uses standard mathematical formulations (`alpha * (1 - pt)**gamma * ce_loss`). 
  * Class weighting uses inverse class frequency correctly via `np.bincount` and passes `weight` to PyTorch's `CrossEntropyLoss`.

## 2. Unresolved Concerns and Limitations
* **Comparability of Baselines:** The previous E03 baseline results (Macro-F1 0.26 vs 0.33 vs 0.40) were technically generated on perfectly comparable data splits (identical data loaders, fixed seeds, sequential execution). **However**, two critical limitations invalidate them for E02:
  1. The E03 run did *not* save the required artifacts (`confusion_matrix.png`, `classification_report.csv`, or epoch training histories). It only saved the model weights and the top-level metrics JSON.
  2. The E03 baselines were run on the `operational` feature set (which includes the leaked MQTT features), but the E02 tuning is intended for the `no_mqtt` feature set to ensure we are tuning a model that actually generalizes.
* **Tuning Architecture State:** `run_E02_hyperparameter_tuning.py` currently builds the `Tunable_CNN_GRU` with dynamic filter/unit sizes, but it does not yet save the PyTorch model artifact for the overall absolute best trial. (Optuna saves the parameters, but we have to manually retrain the best params to get the `.pt` file).

## 3. Estimated Runtime and Memory Risks
* **Memory:** The `no_mqtt` dataset is ~1.5 million rows. A batch size of 4096 to 8192 for the CNN-GRU easily fits in standard RAM/VRAM.
* **Compute:** 20 trials * 15 epochs will take approximately **3 to 5 hours** on a CPU. Using GPU acceleration will reduce this significantly. The `MedianPruner` will help kill bad configurations early, saving time.

## 4. GO / NO-GO Recommendation
**Recommendation: NO-GO** on the 20-trial Optuna run.

**Reasons:**
1. We lack the artifact evidence (confusion matrices, class-wise reports) to substantiate the baseline loss comparisons, violating deep research requirements.
2. The baselines must be re-established on the `no_mqtt` dataset so they actually serve as a baseline for the E02 tuning.

## 5. Next Actions
We must execute a baseline script first, then tune.
The exact commands to run once you approve:

```bash
# 1. Generate the ablated dataset (if not currently set to no_mqtt)
python run_E05_mqtt_ablation.py

# 2. Re-run the Baselines on no_mqtt (with the newly patched src/training.py that saves artifacts)
# (I will write a quick run_E02_baselines.py for this)
python run_E02_baselines.py

# 3. Finally, trigger the optimization
python run_E02_hyperparameter_tuning.py
```
