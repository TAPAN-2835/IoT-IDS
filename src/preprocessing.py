import hashlib
from datetime import datetime

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from src import config as cfg
from src.data_loader import get_csv_files, load_policy_columns, canonical_token
from src.utils import setup_logger, save_json, load_json

logger = setup_logger(__name__)

def processed_metadata_path():
    """metadata.json next to the processed splits (follows cfg.PROCESSED_DATA_DIR at call time)."""
    return cfg.PROCESSED_DATA_DIR / "metadata.json"


def get_operational_features() -> list:
    """Load the operational features designated by the audit phase."""
    policy_filename = f"{cfg.FEATURE_POLICY}_feature_policy.csv" if hasattr(cfg, "FEATURE_POLICY") and cfg.FEATURE_POLICY != "operational" else "operational_feature_policy.csv"
    policy_path = cfg.AUDIT_DIR / policy_filename
    if not policy_path.exists():
        logger.error(f"{policy_filename} not found. Please run audit first.")
        return []

    policy_df = pd.read_csv(policy_path)
    return policy_df['feature'].tolist()


def canonicalize_column(series: pd.Series) -> pd.Series:
    """Merge numeric spellings of the same value ("0" / "0.0") in a string column.

    Works on the categories rather than every row, so it is cheap on 2M rows.
    """
    cat = series.astype("category")
    old_categories = cat.cat.categories
    if len(old_categories) == 0:
        return cat
    mapped = [canonical_token(c) for c in old_categories]
    new_categories = pd.Index(mapped).unique()
    lookup = new_categories.get_indexer(mapped)
    codes = cat.cat.codes.to_numpy()
    new_codes = np.where(codes >= 0, lookup[codes], -1)
    return pd.Series(pd.Categorical.from_codes(new_codes, categories=new_categories),
                     index=series.index, name=series.name)


def clean_data(df: pd.DataFrame, operational_features: list, target_col: str = None) -> pd.DataFrame:
    """Filter down to operational features and the target column."""
    target_col = target_col or cfg.DEFAULT_TARGET_COL
    keep_cols = operational_features.copy()
    if target_col in df.columns and target_col not in keep_cols:
        keep_cols.append(target_col)

    # Drop columns not in keep_cols
    drop_cols = [c for c in df.columns if c not in keep_cols]
    if drop_cols:
        logger.info(f"Dropping {len(drop_cols)} columns due to leakage policy.")
        df = df.drop(columns=drop_cols)

    # Drop rows where target is missing (skip the full-frame copy when there are none)
    if target_col in df.columns and df[target_col].isna().any():
        df = df.dropna(subset=[target_col])

    string_cols = [c for c in df.columns if c != target_col and (
        pd.api.types.is_string_dtype(df[c]) or pd.api.types.is_object_dtype(df[c])
        or isinstance(df[c].dtype, pd.CategoricalDtype))]

    # Canonicalise numeric-looking tokens BEFORE encoding, otherwise "0" and "0.0"
    # become separate one-hot columns that encode which capture a row came from.
    if cfg.CANONICALIZE_NUMERIC_TOKENS:
        for col in string_cols:
            df[col] = canonicalize_column(df[col])

    # Drop high cardinality string columns to prevent OOM in OneHotEncoder
    # e.g., tcp.payload which has hundreds of thousands of unique values
    high_cardinality_cols = []
    for col in string_cols:
        unique_count = df[col].nunique()
        if unique_count > cfg.MAX_CATEGORIES:
            high_cardinality_cols.append(col)
            logger.warning(f"Dropping {col} due to high cardinality ({unique_count} unique values).")

    if high_cardinality_cols:
        df = df.drop(columns=high_cardinality_cols)

    return df

def build_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    """Build the preprocessing pipeline based on training data dtypes."""
    numeric_features = X_train.select_dtypes(include=['int64', 'float64', 'float32', 'int32']).columns.tolist()
    categorical_features = X_train.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False, dtype=np.float32))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'
    )

    return preprocessor


def feature_fingerprint(feature_names, target_col, feature_policy, canonicalized) -> str:
    """Short hash identifying a processed-data schema, stored with every experiment."""
    payload = "|".join([target_col, str(feature_policy), str(canonicalized), str(cfg.RANDOM_SEED)]
                       + list(feature_names))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def load_processed_metadata() -> dict:
    """Describe what data/processed/ currently holds (target, policy, features)."""
    path = processed_metadata_path()
    if not path.exists():
        try:
            # Fallback for when metadata.json is missing but models/ are present
            feature_names = load_json(cfg.MODELS_DIR / "feature_columns.json")["features"]
            label_map = load_json(cfg.MODELS_DIR / "label_mapping.json")
            return {
                "target_col": "Attack_label",
                "feature_policy": "strict_no_mqtt",
                "n_features": len(feature_names),
                "feature_names": feature_names,
                "label_mapping": label_map,
                "n_classes": len(label_map),
                "categorical_columns": [],
                "canonicalize_numeric_tokens": cfg.CANONICALIZE_NUMERIC_TOKENS,
                "fingerprint": "fallback"
            }
        except Exception:
            pass
        raise FileNotFoundError(
            f"{path} not found: data/processed/ was produced by an older "
            "pipeline version. Re-run preprocessing (run_preprocessing_pipeline) first.")
    return load_json(path)


def run_preprocessing_pipeline(target_col: str = None) -> dict:
    target_col = target_col or cfg.DEFAULT_TARGET_COL
    csv_files = get_csv_files()
    if not csv_files:
        logger.error("No CSV files found in data/raw/. Please check data/raw/README.md.")
        return {}

    target_csv = csv_files[0]
    if target_csv.name != "DNN-EdgeIIoT-dataset.csv":
        logger.warning(f"Primary DNN-EdgeIIoT-dataset.csv not found! Falling back to: {target_csv.name}")
    logger.info(f"Processing dataset: {target_csv.name} | target={target_col} | "
                f"policy={cfg.FEATURE_POLICY} | canonicalize={cfg.CANONICALIZE_NUMERIC_TOKENS}")

    op_features = get_operational_features()
    if not op_features:
        return {}

    # Low-RAM load: only policy columns, payload-sized string columns skipped while streaming.
    df, dropped_early = load_policy_columns(target_csv, op_features, target_col,
                                            max_categories=cfg.MAX_CATEGORIES,
                                            canonicalize=cfg.CANONICALIZE_NUMERIC_TOKENS)
    df = clean_data(df, op_features, target_col)

    if target_col not in df.columns:
        logger.error(f"Target column {target_col} not found in the dataset.")
        return {}

    y = df.pop(target_col)  # in place, avoids copying every feature column
    if isinstance(y.dtype, pd.CategoricalDtype):
        y = y.astype(str)
    X = df
    del df
    _log_memory("loaded")

    logger.info("Splitting dataset into train/val/test...")
    # Math: test_size for initial split is (val_ratio + test_ratio)
    test_val_ratio = cfg.VAL_RATIO + cfg.TEST_RATIO
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_val_ratio, random_state=cfg.RANDOM_SEED, stratify=y
    )
    del X, y

    # Split temp into val and test
    test_ratio_of_temp = cfg.TEST_RATIO / test_val_ratio
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=test_ratio_of_temp, random_state=cfg.RANDOM_SEED, stratify=y_temp
    )
    del X_temp, y_temp

    logger.info(f"Train size: {X_train.shape[0]}, Val size: {X_val.shape[0]}, Test size: {X_test.shape[0]}")

    # Encode Labels
    logger.info("Encoding targets...")
    label_encoder = LabelEncoder()
    y_train_enc = label_encoder.fit_transform(y_train)
    y_val_enc = label_encoder.transform(y_val)
    y_test_enc = label_encoder.transform(y_test)

    # Save label mappings
    label_mapping = {int(i): str(cls) for i, cls in enumerate(label_encoder.classes_)}
    save_json(label_mapping, cfg.MODELS_DIR / "label_mapping.json")

    logger.info("Building and fitting preprocessor on TRAINING DATA ONLY...")
    preprocessor = build_preprocessor(X_train)
    categorical_columns = list(preprocessor.transformers[1][2])

    # FIT ON TRAIN
    X_train_proc = preprocessor.fit_transform(X_train).astype(np.float32, copy=False)
    del X_train
    _log_memory("fitted")

    # Feature columns after transformation
    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        # Fallback if scikit-learn version issues
        feature_names = [f"feature_{i}" for i in range(X_train_proc.shape[1])]
    feature_names = list(feature_names)

    save_json({"features": feature_names}, cfg.MODELS_DIR / "feature_columns.json")

    # Save preprocessor
    joblib.dump(preprocessor, cfg.MODELS_DIR / "preprocessor.joblib")

    # Save processed splits one at a time (float32) to keep peak RAM low
    logger.info("Saving processed datasets...")
    pd.DataFrame(X_train_proc, columns=feature_names).to_parquet(cfg.PROCESSED_DATA_DIR / "X_train.parquet")
    del X_train_proc

    # TRANSFORM ON VAL AND TEST
    logger.info("Transforming validation and test data...")
    for split_name, X_split in (("val", X_val), ("test", X_test)):
        X_proc = preprocessor.transform(X_split).astype(np.float32, copy=False)
        pd.DataFrame(X_proc, columns=feature_names).to_parquet(cfg.PROCESSED_DATA_DIR / f"X_{split_name}.parquet")
        del X_proc

    pd.Series(y_train_enc, name=target_col).to_frame().to_parquet(cfg.PROCESSED_DATA_DIR / "y_train.parquet")
    pd.Series(y_val_enc, name=target_col).to_frame().to_parquet(cfg.PROCESSED_DATA_DIR / "y_val.parquet")
    pd.Series(y_test_enc, name=target_col).to_frame().to_parquet(cfg.PROCESSED_DATA_DIR / "y_test.parquet")

    metadata = {
        "created_at": datetime.now().isoformat(),
        "dataset_file": target_csv.name,
        "target_col": target_col,
        "feature_policy": cfg.FEATURE_POLICY,
        "canonicalize_numeric_tokens": cfg.CANONICALIZE_NUMERIC_TOKENS,
        "split_seed": cfg.RANDOM_SEED,
        "n_features": len(feature_names),
        "n_classes": len(label_mapping),
        "train_size": int(len(y_train_enc)),
        "val_size": int(len(y_val_enc)),
        "test_size": int(len(y_test_enc)),
        "dropped_high_cardinality": dropped_early,
        "categorical_columns": categorical_columns,
        "label_mapping": label_mapping,
        "feature_names": feature_names,
        "fingerprint": feature_fingerprint(feature_names, target_col, cfg.FEATURE_POLICY,
                                           cfg.CANONICALIZE_NUMERIC_TOKENS),
    }
    save_json(metadata, processed_metadata_path())
    _log_memory("done")
    logger.info(f"Preprocessing pipeline completed successfully. Fingerprint: {metadata['fingerprint']}")
    return metadata


def _log_memory(stage: str) -> None:
    """Log current and peak RAM of this process (peak is Windows-only)."""
    try:
        import psutil
        mem = psutil.Process().memory_info()
        peak = getattr(mem, "peak_wset", None)
        logger.info(f"[memory] {stage}: {mem.rss / 1e9:.2f} GB"
                    + (f" (peak {peak / 1e9:.2f} GB)" if peak else ""))
    except ImportError:
        pass

if __name__ == "__main__":
    run_preprocessing_pipeline()
