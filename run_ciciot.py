"""
run_ciciot.py

Part 2 of the project: the same honest pipeline on CICIoT2023, a flow-level IoT
dataset that has the rate/timing features Edge-IIoTset lacks.

  --stage data   build the sampled, de-duplicated datasets (child process, CPU)
                 and run the shortcut scan on both targets
  --stage train  GPU experiments:
                   CI01_xgb_binary            XGBoost, Benign vs Attack
                   CI02_cnn_gru_binary        CNN-GRU, sqrt class weight, validation-tuned threshold
                   CI03_xgb_category          XGBoost, 8 categories
                   CI04_cnn_gru_category_ce   CNN-GRU, cross-entropy
                   CI05_cnn_gru_category_sqrt CNN-GRU, sqrt class weights
  --stage all    both

Progress appears in watch_dashboard.py.
"""
import os

os.environ.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")

import argparse
import multiprocessing as mp

import torch

import src.config as cfg
from run_clean_baselines import be_nice, shortcut_summary, log_peak_memory
from src.ciciot import CICIOT_DIR, CICIOT_STRICT_DIR
from src.run_status import RunStatus
from src.train import train_xgb_gpu
from src.training import train_dl_model
from src.utils import setup_logger

logger = setup_logger("run_ciciot")

EXPERIMENTS = [
    ("CI01_xgb_binary", "binary", "xgb", None),
    ("CI02_cnn_gru_binary", "binary", "CNN-GRU", "sqrt_weighted"),
    ("CI03_xgb_category", "category", "xgb", None),
    ("CI04_cnn_gru_category_ce", "category", "CNN-GRU", "ce"),
    ("CI05_cnn_gru_category_sqrt", "category", "CNN-GRU", "sqrt_weighted"),
]
TARGET_COL = {"binary": "is_attack", "category": "category"}
STRICT = False  # set from --policy


def _data_job(strict):
    be_nice()
    from src.ciciot import build_ciciot_datasets
    from audit_shortcuts import processed_shortcut_scan
    for name, out in build_ciciot_datasets(strict=strict).items():
        cfg.PROCESSED_DATA_DIR = out
        processed_shortcut_scan()


def use_dataset(name):
    cfg.PROCESSED_DATA_DIR = (CICIOT_STRICT_DIR if STRICT else CICIOT_DIR) / name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["data", "train", "all"], default="all")
    parser.add_argument("--policy", choices=["flow", "strict"], default="flow",
                        help="strict: drop window-size artefacts (packet_count, total_sum), counts -> fractions")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    be_nice()
    global STRICT
    STRICT = args.policy == "strict"
    experiments = [(e.replace("CI", "CS", 1) + "_strict" if STRICT else e, d, m, l) for e, d, m, l in EXPERIMENTS]

    steps = []
    if args.stage in ("data", "all"):
        steps.append("Build CICIoT2023 sample + shortcut scans")
    if args.stage in ("train", "all"):
        steps += [e[0] for e in experiments]
    status = RunStatus(f"CICIoT2023 | policy={args.policy} | stage={args.stage}", steps)

    if args.stage in ("data", "all"):
        step = steps[0]
        status.start(step, "stream 46.8M-row Parquet, cap per attack label, de-duplicate, split, scale")
        proc = mp.get_context("spawn").Process(target=_data_job, args=(STRICT,))
        proc.start()
        proc.join()
        if proc.exitcode != 0:
            status.fail(step, f"exit code {proc.exitcode}; see pipeline.log")
            raise SystemExit(1)
        summaries = []
        for name in ("binary", "category"):
            use_dataset(name)
            summaries.append(f"{name}: {shortcut_summary(TARGET_COL[name])}")
        status.done(step, " || ".join(summaries))

    if args.stage in ("train", "all"):
        for exp_id, dataset, model, loss in experiments:
            if args.skip_existing and (cfg.EXPERIMENTS_DIR / exp_id / "experiment_record.json").exists():
                status.skip(exp_id)
                continue
            use_dataset(dataset)
            task = "binary" if dataset == "binary" else "multiclass"
            status.start(exp_id, "XGBoost on GPU" if model == "xgb" else f"{model}, loss={loss}, {cfg.EPOCHS} epochs")
            try:
                if model == "xgb":
                    record = train_xgb_gpu(exp_id, TARGET_COL[dataset], task)
                else:
                    record = train_dl_model(exp_id, model, TARGET_COL[dataset], task, loss_type=loss,
                                            epochs=cfg.EPOCHS, tune_threshold=task == "binary")
            except Exception as e:
                status.fail(exp_id, e)
                raise
            status.done(exp_id, f"acc {record['accuracy']:.4f} | macro-F1 {record['macro_f1']:.4f}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        log_peak_memory("ciciot")
    status.finish()


if __name__ == "__main__":
    main()
