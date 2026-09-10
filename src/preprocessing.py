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
from src.data_loader import get_csv_files, load_dataset_full
from src.utils import setup_logger, save_json

logger = setup_logger(__name__)

def get_operational_features() -> list:
    """Load the operational features designated by the audit phase."""
    policy_path = cfg.AUDIT_DIR / "operational_feature_policy.csv"
    if not policy_path.exists():
        logger.error("operational_feature_policy.csv not found. Please run audit first.")
        return []
    
    policy_df = pd.read_csv(policy_path)
    return policy_df['feature'].tolist()

def clean_data(df: pd.DataFrame, operational_features: list) -> pd.DataFrame:
    """Filter down to operational features and the target column."""
    keep_cols = operational_features.copy()
    if cfg.DEFAULT_TARGET_COL in df.columns and cfg.DEFAULT_TARGET_COL not in keep_cols:
        keep_cols.append(cfg.DEFAULT_TARGET_COL)
        
    # Drop columns not in keep_cols
    drop_cols = [c for c in df.columns if c not in keep_cols]
    if drop_cols:
        logger.info(f"Dropping {len(drop_cols)} columns due to leakage policy.")
        df = df.drop(columns=drop_cols)
        
    # Drop rows where target is missing
    if cfg.DEFAULT_TARGET_COL in df.columns:
        df = df.dropna(subset=[cfg.DEFAULT_TARGET_COL])
        
    # Drop high cardinality string columns to prevent OOM in OneHotEncoder
    # e.g., tcp.payload which has hundreds of thousands of unique values
    high_cardinality_cols = []
    for col in df.columns:
        if col != cfg.DEFAULT_TARGET_COL:
            if pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]):
                unique_count = df[col].nunique()
                if unique_count > 100:
                    high_cardinality_cols.append(col)
                    logger.warning(f"Dropping {col} due to high cardinality ({unique_count} unique values).")
    
    if high_cardinality_cols:
        df = df.drop(columns=high_cardinality_cols)
        
    return df

def build_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    """Build the preprocessing pipeline based on training data dtypes."""
    numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X_train.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'
    )
    
    return preprocessor

def run_preprocessing_pipeline() -> None:
    csv_files = get_csv_files()
    if not csv_files:
        logger.error("No CSV files found in data/raw/. Please check data/raw/README.md.")
        return
        
    target_csv = csv_files[0]
    if target_csv.name != "DNN-EdgeIIoT-dataset.csv":
        logger.warning(f"Primary DNN-EdgeIIoT-dataset.csv not found! Falling back to: {target_csv.name}")
    logger.info(f"Processing dataset: {target_csv.name}")
    
    df = load_dataset_full(target_csv)
    
    op_features = get_operational_features()
    if not op_features:
        return
        
    df = clean_data(df, op_features)
    
    if cfg.DEFAULT_TARGET_COL not in df.columns:
        logger.error(f"Target column {cfg.DEFAULT_TARGET_COL} not found in the dataset.")
        return
        
    X = df.drop(columns=[cfg.DEFAULT_TARGET_COL])
    y = df[cfg.DEFAULT_TARGET_COL]
    
    logger.info("Splitting dataset into train/val/test...")
    # Math: test_size for initial split is (val_ratio + test_ratio)
    test_val_ratio = cfg.VAL_RATIO + cfg.TEST_RATIO
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_val_ratio, random_state=cfg.RANDOM_SEED, stratify=y
    )
    
    # Split temp into val and test
    test_ratio_of_temp = cfg.TEST_RATIO / test_val_ratio
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=test_ratio_of_temp, random_state=cfg.RANDOM_SEED, stratify=y_temp
    )
    
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
    
    # FIT ON TRAIN
    X_train_proc = preprocessor.fit_transform(X_train)
    
    # TRANSFORM ON VAL AND TEST
    logger.info("Transforming validation and test data...")
    X_val_proc = preprocessor.transform(X_val)
    X_test_proc = preprocessor.transform(X_test)
    
    # Feature columns after transformation
    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        # Fallback if scikit-learn version issues
        feature_names = [f"feature_{i}" for i in range(X_train_proc.shape[1])]
        
    save_json({"features": list(feature_names)}, cfg.MODELS_DIR / "feature_columns.json")
    
    # Save preprocessor
    joblib.dump(preprocessor, cfg.MODELS_DIR / "preprocessor.joblib")
    
    # Save processed numpy arrays or dataframes (using parquet for efficiency)
    logger.info("Saving processed datasets...")
    pd.DataFrame(X_train_proc, columns=feature_names).to_parquet(cfg.PROCESSED_DATA_DIR / "X_train.parquet")
    pd.DataFrame(X_val_proc, columns=feature_names).to_parquet(cfg.PROCESSED_DATA_DIR / "X_val.parquet")
    pd.DataFrame(X_test_proc, columns=feature_names).to_parquet(cfg.PROCESSED_DATA_DIR / "X_test.parquet")
    
    pd.Series(y_train_enc, name=cfg.DEFAULT_TARGET_COL).to_frame().to_parquet(cfg.PROCESSED_DATA_DIR / "y_train.parquet")
    pd.Series(y_val_enc, name=cfg.DEFAULT_TARGET_COL).to_frame().to_parquet(cfg.PROCESSED_DATA_DIR / "y_val.parquet")
    pd.Series(y_test_enc, name=cfg.DEFAULT_TARGET_COL).to_frame().to_parquet(cfg.PROCESSED_DATA_DIR / "y_test.parquet")
    
    logger.info("Preprocessing pipeline completed successfully.")

if __name__ == "__main__":
    run_preprocessing_pipeline()
