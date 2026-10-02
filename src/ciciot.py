"""
src/ciciot.py

CICIoT2023 (flow-level IoT dataset, 46.8M rows, 33 attacks + benign) prepared
for the same training pipeline as Edge-IIoTset.

Source: Kaggle "dhoogla/ciciotdataset2023" (typed Parquet of the official CSVs,
duplicates retained, CC BY-NC-SA 4.0). It is read from the kagglehub cache
(outside OneDrive) and never copied into the project.
Citation: Neto et al., "CICIoT2023: A real-time dataset and benchmark for
large-scale attacks in IoT environment", Sensors 23(13), 2023.

What build_ciciot_datasets() does:
  1. Streams the Parquet row group by row group and keeps a capped sample per
     attack label (rare attacks keep every row), so RAM stays low.
  2. Drops identifier/metadata columns: source_file (names the capture file,
     i.e. the label), the label columns, and rate_was_infinite.
  3. Removes exact duplicate rows BEFORE splitting, so copies of one flow window
     cannot land in both train and test.
  4. Stratified 70/15/15 split (seed 42) on the fine label, signed log1p on
     heavy-tailed columns, StandardScaler fitted on train only.
  5. Writes two processed datasets with identical rows/splits:
       data/processed_ciciot/binary    target is_attack (Benign=0, Attack=1)
       data/processed_ciciot/category  target category (8 classes)

strict=True (written to data/processed_ciciot_strict/): the features were
extracted over packet windows whose SIZE depends on the capture: packet_count is
10 for every Benign/Recon row and 100 for every DDoS/DoS/Mirai row, and
total_sum = packet_count x mean_packet_size. Both are dropped, and the per-window
flag counts are turned into fractions of the window so they no longer encode it.
"""
import hashlib
from datetime import datetime

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src import config as cfg
from src.utils import save_json, setup_logger

logger = setup_logger(__name__)

KAGGLE_REF = "dhoogla/ciciotdataset2023"
CICIOT_DIR = cfg.DATA_DIR / "processed_ciciot"
CICIOT_STRICT_DIR = cfg.DATA_DIR / "processed_ciciot_strict"
WINDOW_SIZE_COLUMNS = ["packet_count", "total_sum"]
WINDOW_COUNT_COLUMNS = ["ack_count", "syn_count", "fin_count", "rst_count"]
NON_FEATURES = ["label", "category", "binary_label", "source_file", "rate_was_infinite"]


def ciciot_parquet_path():
    import kagglehub
    from pathlib import Path
    root = Path(kagglehub.dataset_download(KAGGLE_REF))  # returns the cached copy if present
    return next(root.rglob("ciciot2023.parquet"))


def _sample(path, cap_per_label, benign_cap, seed):
    pf = pq.ParquetFile(path)
    counts = {d["values"]: d["counts"] for d in pc.value_counts(pf.read(columns=["label"])["label"]).to_pylist()}
    keep_prob = {lab: min(1.0, (benign_cap if lab == "BenignTraffic" else cap_per_label) / n)
                 for lab, n in counts.items()}
    rng = np.random.default_rng(seed)
    columns = [c for c in pf.schema_arrow.names if c not in ("source_file", "rate_was_infinite", "binary_label")]
    parts = []
    for i in range(pf.num_row_groups):
        rg = pf.read_row_group(i, columns=columns)
        labels = rg.column("label").to_numpy(zero_copy_only=False)
        p = np.vectorize(keep_prob.get)(labels)
        mask = rng.random(len(labels)) < p
        if mask.any():
            parts.append(rg.filter(pa.array(mask)))
        if (i + 1) % 50 == 0:
            logger.info(f"  sampled row group {i + 1}/{pf.num_row_groups}")
    table = pa.concat_tables(parts)
    return table.to_pandas(), counts


def _fingerprint(feature_names, target_col):
    payload = "|".join(["ciciot2023", target_col, str(cfg.RANDOM_SEED)] + list(feature_names))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def build_ciciot_datasets(cap_per_label=40_000, benign_cap=400_000, strict=False):
    path = ciciot_parquet_path()
    logger.info(f"Sampling {path.name}: up to {cap_per_label:,} rows per attack label, {benign_cap:,} benign")
    df, full_counts = _sample(path, cap_per_label, benign_cap, cfg.RANDOM_SEED)
    if strict:
        window = df["packet_count"].astype(np.float64).replace(0, np.nan)
        for c in WINDOW_COUNT_COLUMNS:
            df[f"{c}_frac"] = (df[c] / window).fillna(0.0)
        df = df.drop(columns=WINDOW_SIZE_COLUMNS + WINDOW_COUNT_COLUMNS)
    feature_cols = [c for c in df.columns if c not in NON_FEATURES]
    logger.info(f"Sample: {len(df):,} rows, {len(feature_cols)} features (strict={strict})")

    # Exact duplicates (features + label) would leak across the split.
    n_before = len(df)
    df = df.drop_duplicates(subset=feature_cols + ["label"]).reset_index(drop=True)
    conflicting = int(df.duplicated(subset=feature_cols, keep=False).sum())
    logger.info(f"Removed {n_before - len(df):,} duplicate rows; {conflicting:,} rows share features "
                "with a different label (kept, they are genuinely ambiguous)")

    X = df[feature_cols].astype(np.float64)
    labels = df["label"].astype(str).to_numpy()
    category = df["category"].astype(str).to_numpy()
    del df

    idx = np.arange(len(X))
    idx_train, idx_temp = train_test_split(idx, test_size=0.30, random_state=cfg.RANDOM_SEED, stratify=labels)
    idx_val, idx_test = train_test_split(idx_temp, test_size=0.50, random_state=cfg.RANDOM_SEED,
                                         stratify=labels[idx_temp])

    # Heavy-tailed, non-negative columns (rate, sizes, IAT...) -> signed log1p; fixed transform, no fitting.
    train_max = X.iloc[idx_train].max()
    heavy = [c for c in feature_cols if train_max[c] > 100]
    X[heavy] = np.sign(X[heavy]) * np.log1p(np.abs(X[heavy]))
    scaler = StandardScaler().fit(X.iloc[idx_train])
    X_scaled = scaler.transform(X).astype(np.float32)
    del X

    targets = {
        "binary": ("is_attack", (category != "Benign").astype(np.int64), {0: "Benign", 1: "Attack"}),
    }
    cat_names = sorted(set(category))
    cat_index = {c: i for i, c in enumerate(cat_names)}
    targets["category"] = ("category", np.array([cat_index[c] for c in category]),
                           {i: c for i, c in enumerate(cat_names)})

    split_idx = {"train": idx_train, "val": idx_val, "test": idx_test}
    written = {}
    policy = "ciciot_flow_strict" if strict else "ciciot_flow"
    for name, (target_col, y, mapping) in targets.items():
        out = (CICIOT_STRICT_DIR if strict else CICIOT_DIR) / name
        out.mkdir(parents=True, exist_ok=True)
        for split, ids in split_idx.items():
            pd.DataFrame(X_scaled[ids], columns=feature_cols).to_parquet(out / f"X_{split}.parquet")
            pd.DataFrame({target_col: y[ids]}).to_parquet(out / f"y_{split}.parquet")
        meta = {
            "created_at": datetime.now().isoformat(),
            "dataset_file": f"kaggle:{KAGGLE_REF}/ciciot2023.parquet",
            "target_col": target_col,
            "feature_policy": policy,
            "canonicalize_numeric_tokens": False,
            "split_seed": cfg.RANDOM_SEED,
            "n_features": len(feature_cols),
            "n_classes": len(mapping),
            "train_size": int(len(idx_train)),
            "val_size": int(len(idx_val)),
            "test_size": int(len(idx_test)),
            "sampling": {"cap_per_label": cap_per_label, "benign_cap": benign_cap,
                         "full_label_counts": full_counts},
            "log1p_columns": heavy,
            "dropped_high_cardinality": [],
            "categorical_columns": [],
            "label_mapping": {str(k): v for k, v in mapping.items()},
            "feature_names": feature_cols,
            "fingerprint": _fingerprint(feature_cols, target_col + ("|strict" if strict else "")),
            "class_counts_train": {mapping[int(k)]: int(v) for k, v in
                                   zip(*np.unique(y[idx_train], return_counts=True))},
        }
        save_json(meta, out / "metadata.json")
        written[name] = out
        logger.info(f"Wrote {out} ({target_col}, {len(mapping)} classes, fingerprint {meta['fingerprint']})")
    return written
