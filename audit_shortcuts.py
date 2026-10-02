"""
audit_shortcuts.py

Two cheap checks for label shortcuts in Edge-IIoTset (both low-RAM):

1. Raw empty-token audit: for every string column, how each class writes an
   empty protocol field ("0" vs "0.0") versus real values. Streams the CSV in
   blocks, so the 1.2 GB file is never fully in memory.
     -> results/audit/empty_token_audit.csv

2. Processed shortcut scan: on whatever data/processed/ currently holds, fit a
   depth-1 decision stump per feature and one depth-3 tree on all features
   (subsampled). If a single feature or a 3-level tree already reaches ~100%,
   the task is still trivially separable.
     -> results/audit/shortcut_scan_<target>_<fingerprint>.csv / _tree.txt

Usage:
    python audit_shortcuts.py            # both checks
    python audit_shortcuts.py --raw      # only the raw token audit
    python audit_shortcuts.py --processed
"""
import argparse

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pv
from sklearn.metrics import balanced_accuracy_score
from sklearn.tree import DecisionTreeClassifier, export_text

from src import config as cfg
from src.data_loader import get_csv_files
from src.preprocessing import get_operational_features, load_processed_metadata
from src.utils import setup_logger

logger = setup_logger("audit_shortcuts")

EMPTY_TOKENS = ("0", "0.0")
SAMPLE_SIZE = 300_000


def raw_empty_token_audit():
    csv_path = get_csv_files()[0]
    features = get_operational_features()

    # Read every policy column as text and track which ones contain any
    # non-numeric value: only those become one-hot encoded string columns.
    # (In numeric columns "0" and "0.0" both parse to 0.0 and are harmless.)
    logger.info(f"Streaming {csv_path.name} over {len(features)} policy columns...")
    conv = pv.ConvertOptions(include_columns=features + ["Attack_label"],
                             column_types={c: pa.string() for c in features},
                             strings_can_be_null=True)
    counts = {}
    numeric = {c: True for c in features}
    with pv.open_csv(csv_path, read_options=pv.ReadOptions(block_size=16 << 20, use_threads=False), convert_options=conv) as reader:
        for batch in reader:
            label = batch.column("Attack_label").to_numpy()
            for c in features:
                col = batch.column(c)
                if numeric[c]:
                    try:
                        col.cast(pa.float64())
                    except pa.ArrowInvalid:
                        numeric[c] = False
                is_zero = pc.fill_null(pc.equal(col, "0"), False).to_numpy(zero_copy_only=False)
                is_zero_float = pc.fill_null(pc.equal(col, "0.0"), False).to_numpy(zero_copy_only=False)
                for lab in (0, 1):
                    m = label == lab
                    key = (c, lab)
                    z, zf = int(is_zero[m].sum()), int(is_zero_float[m].sum())
                    prev = counts.get(key, (0, 0, 0))
                    counts[key] = (prev[0] + z, prev[1] + zf, prev[2] + int(m.sum()) - z - zf)

    string_cols = [c for c in features if not numeric[c]]
    rows = []
    for c in string_cols:
        n0, n1 = counts[(c, 0)], counts[(c, 1)]
        # Each class writes empties with exactly one spelling, and they differ.
        normal_spelling = {s for s, n in zip(EMPTY_TOKENS, n0[:2]) if n > 0}
        attack_spelling = {s for s, n in zip(EMPTY_TOKENS, n1[:2]) if n > 0}
        separates = len(normal_spelling) == 1 and len(attack_spelling) == 1 and normal_spelling != attack_spelling
        rows.append({"column": c,
                     "normal_empty_as_0": n0[0], "normal_empty_as_0.0": n0[1], "normal_real_value": n0[2],
                     "attack_empty_as_0": n1[0], "attack_empty_as_0.0": n1[1], "attack_real_value": n1[2],
                     "empty_format_separates_classes": separates})
    out = pd.DataFrame(rows)
    out_path = cfg.AUDIT_DIR / "empty_token_audit.csv"
    out.to_csv(out_path, index=False)
    logger.info(f"Empty-token audit written to {out_path}")
    logger.info("\n" + out.to_string(index=False))
    return out


def _sample(X, y, n, rng):
    idx = rng.choice(len(y), size=min(n, len(y)), replace=False)
    return X[idx], y[idx]


def processed_shortcut_scan():
    meta = load_processed_metadata()
    rng = np.random.default_rng(cfg.GLOBAL_SEED)
    X_train = pd.read_parquet(cfg.PROCESSED_DATA_DIR / "X_train.parquet").to_numpy(dtype=np.float32)
    y_train = pd.read_parquet(cfg.PROCESSED_DATA_DIR / "y_train.parquet").to_numpy().ravel()
    X_tr, y_tr = _sample(X_train, y_train, SAMPLE_SIZE, rng)
    del X_train, y_train
    X_val = pd.read_parquet(cfg.PROCESSED_DATA_DIR / "X_val.parquet").to_numpy(dtype=np.float32)
    y_val = pd.read_parquet(cfg.PROCESSED_DATA_DIR / "y_val.parquet").to_numpy().ravel()
    X_va, y_va = _sample(X_val, y_val, SAMPLE_SIZE // 3, rng)
    del X_val, y_val

    names = meta["feature_names"]
    logger.info(f"Shortcut scan on {meta['target_col']} | policy={meta['feature_policy']} | "
                f"canonicalize={meta['canonicalize_numeric_tokens']} | {len(names)} features")

    rows = []
    for j, name in enumerate(names):
        stump = DecisionTreeClassifier(max_depth=1, random_state=cfg.GLOBAL_SEED)
        stump.fit(X_tr[:, [j]], y_tr)
        rows.append({"feature": name,
                     "stump_balanced_accuracy": balanced_accuracy_score(y_va, stump.predict(X_va[:, [j]]))})
    scan = pd.DataFrame(rows).sort_values("stump_balanced_accuracy", ascending=False)

    tree = DecisionTreeClassifier(max_depth=3, random_state=cfg.GLOBAL_SEED).fit(X_tr, y_tr)
    tree_bal_acc = balanced_accuracy_score(y_va, tree.predict(X_va))
    rules = export_text(tree, feature_names=list(names))

    tag = f"{meta['target_col']}_{meta['fingerprint']}"
    scan.to_csv(cfg.AUDIT_DIR / f"shortcut_scan_{tag}.csv", index=False)
    with open(cfg.AUDIT_DIR / f"shortcut_scan_{tag}_tree.txt", "w", encoding="utf-8") as f:
        f.write(f"target={meta['target_col']} policy={meta['feature_policy']} "
                f"canonicalize={meta['canonicalize_numeric_tokens']}\n")
        f.write(f"depth-3 tree balanced accuracy (val sample): {tree_bal_acc:.4f}\n\n{rules}")

    logger.info("Top single-feature stumps (balanced accuracy on a validation sample):\n"
                + scan.head(10).to_string(index=False))
    logger.info(f"Depth-3 tree on all features: balanced accuracy {tree_bal_acc:.4f}\n{rules}")
    return scan, tree_bal_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", action="store_true", help="only the raw empty-token audit")
    parser.add_argument("--processed", action="store_true", help="only the processed-data shortcut scan")
    args = parser.parse_args()
    run_all = not (args.raw or args.processed)
    if args.raw or run_all:
        raw_empty_token_audit()
    if args.processed or run_all:
        processed_shortcut_scan()


if __name__ == "__main__":
    main()
