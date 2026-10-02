import pytest
import numpy as np
import pandas as pd
from unittest import mock
from src import data_loader, audit, preprocessing
from src.config import DEFAULT_TARGET_COL

def test_dataset_loading_schema(dummy_dataset_path):
    """Test that schema discovery correctly parses columns and rows."""
    metadata = data_loader.scan_dataset(dummy_dataset_path)
    
    assert metadata['num_rows'] == 1000
    assert 'ip.src_host' in metadata['columns']
    assert DEFAULT_TARGET_COL in metadata['columns']

def test_leakage_detection():
    """Test that leakage detection identifies IP, ports, and time correctly."""
    cat, reason, action = audit.evaluate_leakage_candidate('ip.src_host', 'object')
    assert 'Drop' in action
    
    cat, reason, action = audit.evaluate_leakage_candidate('tcp.port', 'int64')
    assert 'Drop' in action
    
    cat, reason, action = audit.evaluate_leakage_candidate('frame.time', 'object')
    assert 'Drop' in action
    
    cat, reason, action = audit.evaluate_leakage_candidate('sensor_reading_1', 'float64')
    assert 'Keep' == action

@mock.patch('src.preprocessing.get_operational_features')
def test_train_test_split_and_preprocessing_fit(mock_get_op, dummy_dataset_path):
    """
    Explicitly test that the preprocessor is fit on TRAINING data only.
    We'll do this by mocking the pipeline behavior or manually tracing it.
    """
    # Define operational features (excluding IP and Port)
    mock_get_op.return_value = ['arp.opcode', 'sensor_reading_1', 'sensor_reading_2']
    
    df = pd.read_csv(dummy_dataset_path)
    df_clean = preprocessing.clean_data(df, mock_get_op.return_value)
    
    assert 'ip.src_host' not in df_clean.columns
    assert 'tcp.port' not in df_clean.columns
    assert DEFAULT_TARGET_COL in df_clean.columns
    
    X = df_clean.drop(columns=[DEFAULT_TARGET_COL])
    y = df_clean[DEFAULT_TARGET_COL]
    
    # We will manually do the split to test the build_preprocessor
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    preprocessor = preprocessing.build_preprocessor(X_train)
    
    # Fit strictly on train
    X_train_proc = preprocessor.fit_transform(X_train)
    
    # Check that transform works on test without re-fitting
    # If there were unknown categories, it should handle them because of handle_unknown='ignore'
    X_test_proc = preprocessor.transform(X_test)
    
    assert X_train_proc.shape[0] == 700
    assert X_test_proc.shape[0] == 300
    
    # Extract the scaler from the preprocessor to prove it was fit on train
    scaler = preprocessor.named_transformers_['num'].named_steps['scaler']
    
    # The scaler's mean should exactly match X_train's mean for numerical features
    num_features = X_train.select_dtypes(include=['int64', 'float64']).columns
    train_means = X_train[num_features].mean().values
    
    import numpy as np
    np.testing.assert_almost_equal(scaler.mean_, train_means, decimal=5)
    
    # Ensure it doesn't match the FULL dataset's mean (proving no leakage)
    full_means = X[num_features].mean().values
    with pytest.raises(AssertionError):
        np.testing.assert_almost_equal(scaler.mean_, full_means, decimal=5)


def test_canonicalize_merges_empty_token_spellings():
    """'0' and '0.0' must become one category, otherwise one-hot encoding leaks the capture file."""
    s = pd.Series(["0", "0.0", "0", "MQTT", None, "1461073", "1461073.0"])
    out = preprocessing.canonicalize_column(s)
    assert out[0] == out[1] == out[2] == "0.0"
    assert out[3] == "MQTT"
    assert pd.isna(out[4])
    assert out[5] == out[6]
    assert out.nunique() == 3


def test_clean_data_removes_zero_format_shortcut():
    """After cleaning, the empty-field spelling no longer separates the classes."""
    df = pd.DataFrame({
        "dns.qry.name.len": ["0"] * 5 + ["0.0"] * 5,
        "tcp.flags": [1.0] * 10,
        "Attack_label": [0] * 5 + [1] * 5,
    })
    out = preprocessing.clean_data(df, ["dns.qry.name.len", "tcp.flags"], "Attack_label")
    assert out["dns.qry.name.len"].nunique() == 1

    preprocessor = preprocessing.build_preprocessor(out.drop(columns=["Attack_label"]))
    X = preprocessor.fit_transform(out.drop(columns=["Attack_label"]))
    assert X.shape == (10, 2)  # one numeric column + a single one-hot column


def test_display_names_keep_one_hot_category():
    from src.explainability import _display_names
    names = ["num__tcp.flags", "cat__mqtt.topic_0", "cat__mqtt.topic_0.0", "cat__http.request.method_GET"]
    out = _display_names(names, ["mqtt.topic", "http.request.method"])
    assert out == ["tcp.flags", "mqtt.topic=0", "mqtt.topic=0.0", "http.request.method=GET"]
    assert len(set(out)) == len(out)


def test_shap_refuses_mismatched_training_data(tmp_path):
    """A model trained without MQTT features must not be explained with MQTT data."""
    import json
    from src.explainability import _check_data_matches_experiment
    (tmp_path / "experiment_record.json").write_text(json.dumps({"features_used": ["num__tcp.flags"]}))
    meta = {"fingerprint": "abc", "n_features": 2, "feature_policy": "operational",
            "feature_names": ["num__tcp.flags", "num__mqtt.len"]}
    with pytest.raises(RuntimeError):
        _check_data_matches_experiment("E_test", tmp_path, meta)

    meta_ok = dict(meta, feature_names=["num__tcp.flags"], n_features=1)
    _check_data_matches_experiment("E_test", tmp_path, meta_ok)  # does not raise
