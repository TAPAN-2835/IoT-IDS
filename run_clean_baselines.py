"""
run_clean_baselines.py

Re-runs the core baselines after fixing the "0" vs "0.0" empty-field artefact
(cfg.CANONICALIZE_NUMERIC_TOKENS = True). Experiment IDs never collide with the
earlier E01-E08 / Phase-2 runs.

--policy operational  (C-series)
  binary stage      C01_xgb_binary_clean          XGBoost (GPU)
                    C02_cnn_gru_binary_clean      CNN-GRU (GPU), 15 epochs
  multiclass stage  C03_xgb_multiclass_clean      XGBoost (GPU)
                    C04_cnn_gru_mc_ce_clean       CNN-GRU, cross-entropy
                    C05_cnn_gru_mc_weighted_clean CNN-GRU, class-weighted CE
                    C06_cnn_gru_mc_focal_clean    CNN-GRU, focal loss

--policy strict  (S-series): operational minus per-packet identifier fields
(sequence/ack numbers, checksums, stream index); see STRICT_DROP below.
  binary stage      S01_xgb_binary_strict, S02_cnn_gru_binary_strict
  multiclass stage  S03_xgb_multiclass_strict, S04_cnn_gru_mc_ce_strict

Laptop-friendly: heavy work runs on the GPU, the process lowers its own
priority, CPU threads are capped, preprocessing streams the CSV in a child
process, and Windows is kept awake only while the run lasts.
Progress is written to run_status.json; watch it with `python watch_dashboard.py`.

--study losses: multiclass CNN-GRU with cross-entropy, sqrt-softened class
weights and focal loss, each repeated over --seeds (IDs L_<policy>_<loss>_s<seed>).

Usage:
    python run_clean_baselines.py --policy strict
    python run_clean_baselines.py --policy strict --study losses --seeds 42 7 2024
    python run_clean_baselines.py --stage binary --skip-existing
"""
import os

# Use the system allocator for Arrow (must be set before pyarrow is imported):
# its default mimalloc pool keeps freed parquet/CSV buffers instead of returning
# them to the OS, which left GBs of RAM held during GPU training.
os.environ.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")

import argparse
import multiprocessing as mp

import pandas as pd
import psutil
import torch

import src.config as cfg
from src.run_status import RunStatus
from src.train import train_xgb_gpu
from src.training import train_dl_model
from src.utils import setup_logger

logger = setup_logger("run_clean_baselines")

PLANS = {
    "operational": {
        "binary": ("Attack_label", "binary", [
            ("C01_xgb_binary_clean", "xgb", None),
            ("C02_cnn_gru_binary_clean", "CNN-GRU", "ce"),
        ]),
        "multiclass": ("Attack_type", "multiclass", [
            ("C03_xgb_multiclass_clean", "xgb", None),
            ("C04_cnn_gru_mc_ce_clean", "CNN-GRU", "ce"),
            ("C05_cnn_gru_mc_weighted_clean", "CNN-GRU", "class_weighted"),
            ("C06_cnn_gru_mc_focal_clean", "CNN-GRU", "focal"),
        ]),
    },
    "strict": {
        "binary": ("Attack_label", "binary", [
            ("S01_xgb_binary_strict", "xgb", None),
            ("S02_cnn_gru_binary_strict", "CNN-GRU", "ce"),
        ]),
        "multiclass": ("Attack_type", "multiclass", [
            ("S03_xgb_multiclass_strict", "xgb", None),
            ("S04_cnn_gru_mc_ce_strict", "CNN-GRU", "ce"),
        ]),
    },
    # strict minus every mqtt.* field: every MQTT packet in Edge-IIoTset is Normal,
    # so this measures how much of the strict binary score is "is it MQTT".
    "strict_no_mqtt": {
        "binary": ("Attack_label", "binary", [
            ("N01_xgb_binary_strict_nomqtt", "xgb", None),
            ("N02_cnn_gru_binary_strict_nomqtt", "CNN-GRU", "ce"),
        ]),
    },
}

# Fields that identify a packet or capture rather than describe behaviour.
# After the empty-token fix these became the strongest single features.
STRICT_DROP = {
    "tcp.seq": "Relative TCP sequence number: position inside one captured connection",
    "tcp.ack": "Relative TCP acknowledgement number: position inside one captured connection",
    "tcp.ack_raw": "Raw TCP acknowledgement number: random per connection",
    "tcp.checksum": "Per-packet checksum: effectively a random identifier",
    "icmp.checksum": "Per-packet checksum: effectively a random identifier",
    "icmp.seq_le": "ICMP sequence counter: position inside one capture",
    "udp.stream": "Wireshark stream index: assigned per capture file",
}


def loss_study_plan(policy, seeds):
    """Multiclass loss comparison repeated over several seeds (same split, same data)."""
    runs = [(f"L_{policy}_{loss}_s{seed}", "CNN-GRU", loss, seed)
            for seed in seeds for loss in ("ce", "sqrt_weighted", "focal")]
    return {"multiclass": ("Attack_type", "multiclass", runs)}


def ensure_strict_policy(no_mqtt=False):
    """Write results/audit/strict[_no_mqtt]_feature_policy.csv (operational minus STRICT_DROP)."""
    base = pd.read_csv(cfg.AUDIT_DIR / "operational_feature_policy.csv")
    strict = base[~base["feature"].isin(STRICT_DROP)]
    name = "strict"
    if no_mqtt:
        strict = strict[~strict["feature"].str.startswith("mqtt.")]
        name = "strict_no_mqtt"
    strict.to_csv(cfg.AUDIT_DIR / f"{name}_feature_policy.csv", index=False)
    logger.info(f"{name} policy: {len(strict)} features ({len(base) - len(strict)} removed from operational)")


def keep_awake():
    """Stop Windows from sleeping mid-run (the screen may still turn off).

    Applies only while this process runs; normal sleep settings resume when it exits.
    Without it, the laptop's idle-sleep timer froze a training run halfway through.
    """
    if os.name != "nt":
        return
    import ctypes
    ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)


def be_nice():
    """Lower this process's priority and CPU thread use so the laptop stays usable."""
    try:
        proc = psutil.Process()
        proc.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if hasattr(psutil, "BELOW_NORMAL_PRIORITY_CLASS") else 10)
    except Exception as e:
        logger.warning(f"Could not lower process priority: {e}")
    torch.set_num_threads(2)
    keep_awake()
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.warning("CUDA not available: deep-learning runs will fall back to the CPU (slow).")


def _summary(record):
    return f"acc {record['accuracy']:.4f} | macro-F1 {record['macro_f1']:.4f}"


def run_stage(plan, name, skip_existing, status):
    target_col, task_type, experiments = plan[name]
    prep_step = f"Preprocess {target_col} ({cfg.FEATURE_POLICY})"
    logger.info(f"=== Stage: {name} ({target_col}) | policy={cfg.FEATURE_POLICY} | "
                f"canonicalize={cfg.CANONICALIZE_NUMERIC_TOKENS} ===")
    todo = []
    for exp_id, model, loss, *rest in experiments:
        if skip_existing and (cfg.EXPERIMENTS_DIR / exp_id / "experiment_record.json").exists():
            status.skip(exp_id)
        else:
            todo.append((exp_id, model, loss, rest[0] if rest else None))
    if not todo:
        status.skip(prep_step, "not needed")
        logger.info("All experiments in this stage already exist; skipping.")
        return

    if processed_data_matches(target_col):
        status.skip(prep_step, "reused existing data/processed/")
        logger.info("data/processed/ already holds this target/policy; reusing it.")
    else:
        status.start(prep_step, "streaming CSV -> encode -> shortcut scan (child process)")
        preprocess_in_child(target_col)
        status.done(prep_step, shortcut_summary(target_col))

    for exp_id, model, loss, seed in todo:
        seed_note = f", seed {seed}" if seed is not None else ""
        status.start(exp_id, "XGBoost on GPU" if model == "xgb" else f"{model}, loss={loss}, {cfg.EPOCHS} epochs{seed_note}")
        try:
            if model == "xgb":
                record = train_xgb_gpu(exp_id, target_col, task_type)
            else:
                record = train_dl_model(exp_id, model, target_col, task_type, loss_type=loss,
                                        epochs=cfg.EPOCHS, seed=seed)
        except Exception as e:
            status.fail(exp_id, e)
            raise
        status.done(exp_id, _summary(record))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    log_peak_memory(name)


def shortcut_summary(target_col):
    """One-line result of the shortcut scan the child process just wrote."""
    from src.preprocessing import load_processed_metadata
    meta = load_processed_metadata()
    tag = f"{target_col}_{meta['fingerprint']}"
    try:
        scan = pd.read_csv(cfg.AUDIT_DIR / f"shortcut_scan_{tag}.csv")
        with open(cfg.AUDIT_DIR / f"shortcut_scan_{tag}_tree.txt", encoding="utf-8") as f:
            tree_line = f.readlines()[1].strip()
        top = scan.iloc[0]
        return (f"{meta['n_features']} features | best single feature {top['feature']} "
                f"{top['stump_balanced_accuracy']:.2f} | {tree_line.split(':')[-1].strip()} depth-3 tree")
    except Exception:
        return f"{meta['n_features']} features"


def _preprocess_job(target_col, feature_policy, canonicalize):
    """Runs in a child process: all of its RAM is returned to the OS when it exits."""
    be_nice()
    cfg.FEATURE_POLICY = feature_policy
    cfg.CANONICALIZE_NUMERIC_TOKENS = canonicalize
    from src.preprocessing import run_preprocessing_pipeline
    from audit_shortcuts import processed_shortcut_scan
    if not run_preprocessing_pipeline(target_col=target_col):
        raise SystemExit(1)
    processed_shortcut_scan()


def processed_data_matches(target_col):
    """True when data/processed/ was already built with the settings this stage needs."""
    from src.preprocessing import load_processed_metadata
    try:
        meta = load_processed_metadata()
    except FileNotFoundError:
        return False
    return (meta["target_col"] == target_col and meta["feature_policy"] == cfg.FEATURE_POLICY
            and meta["canonicalize_numeric_tokens"] == cfg.CANONICALIZE_NUMERIC_TOKENS)


def preprocess_in_child(target_col):
    proc = mp.get_context("spawn").Process(
        target=_preprocess_job, args=(target_col, cfg.FEATURE_POLICY, cfg.CANONICALIZE_NUMERIC_TOKENS))
    proc.start()
    proc.join()
    if proc.exitcode != 0:
        raise RuntimeError(f"Preprocessing for {target_col} failed (exit code {proc.exitcode}); see pipeline.log")


def log_peak_memory(label):
    mem = psutil.Process().memory_info()
    peak = getattr(mem, "peak_wset", None)  # Windows only
    msg = f"[{label}] RAM now {mem.rss / 1e9:.2f} GB"
    if peak:
        msg += f", peak {peak / 1e9:.2f} GB"
    if torch.cuda.is_available():
        msg += f" | GPU peak {torch.cuda.max_memory_allocated() / 1e9:.2f} GB"
    logger.info(msg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["binary", "multiclass", "all"], default="all")
    parser.add_argument("--policy", choices=list(PLANS), default="operational")
    parser.add_argument("--study", choices=["baselines", "losses"], default="baselines")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 2024])
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    be_nice()
    cfg.FEATURE_POLICY = args.policy
    cfg.CANONICALIZE_NUMERIC_TOKENS = True
    if args.policy in ("strict", "strict_no_mqtt"):
        ensure_strict_policy(no_mqtt=args.policy == "strict_no_mqtt")

    if args.study == "losses":
        plan = loss_study_plan(args.policy, args.seeds)
        stages = ["multiclass"]
    else:
        plan = PLANS[args.policy]
        stages = ["binary", "multiclass"] if args.stage == "all" else [args.stage]
    steps = []
    for stage in stages:
        steps.append(f"Preprocess {plan[stage][0]} ({args.policy})")
        steps.extend(e[0] for e in plan[stage][2])
    title = (f"Loss study | policy={args.policy} | seeds={args.seeds}" if args.study == "losses"
             else f"Baselines | policy={args.policy} | stages={', '.join(stages)}")
    status = RunStatus(title, steps)

    for stage in stages:
        run_stage(plan, stage, args.skip_existing, status)
    status.finish()
    logger.info(f"Run complete ({args.policy}). See results/experiment_registry.csv.")


if __name__ == "__main__":
    main()
