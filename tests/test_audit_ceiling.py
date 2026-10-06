import json
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, mock_open

from audit_ceiling import ceiling

def test_ceiling_computation():
    # Mock data where 2 rows have identical features (one attack, one normal)
    # The normal label appears more, so it's a "mostly Normal" pattern.
    mock_meta = {"feature_policy": "test"}
    
    X_tr_data = pd.DataFrame({'f1': [1, 1, 1, 2], 'f2': [1, 1, 1, 2]})
    # Pattern (1,1) is Normal (0) twice, Attack (1) once. Mostly Normal (mean = 0.33 < 0.5)
    # Pattern (2,2) is Attack (1) once. Mostly Attack (mean = 1.0 >= 0.5)
    y_tr_data = pd.DataFrame({'label': [0, 0, 1, 1]})
    
    X_te_data = pd.DataFrame({'f1': [1, 2, 3], 'f2': [1, 2, 3]})
    # Pattern (1,1) is Attack (1) in test -> should be a hidden attack
    # Pattern (2,2) is Attack (1) in test -> should be a detected attack
    # Pattern (3,3) is Unseen.
    y_te_data = pd.DataFrame({'label': [1, 1, 0]})
    
    with patch('audit_ceiling.json.load', return_value=mock_meta):
        with patch('audit_ceiling.pd.read_parquet') as mock_read:
            with patch('builtins.open', mock_open(read_data='{}')):
                def side_effect(path):
                    name = str(path)
                    if 'X_train' in name: return X_tr_data
                    if 'y_train' in name: return y_tr_data
                    if 'X_test' in name: return X_te_data
                    if 'y_test' in name: return y_te_data
                mock_read.side_effect = side_effect
                
                import pathlib
                res = ceiling(pathlib.Path("dummy_path"))
                
                assert res['train_rows'] == 4
                assert res['distinct_train_patterns'] == 2
                
                # Test has 3 rows, 2 seen in train (1,1) and (2,2). Unseen: (3,3)
                assert np.isclose(res['test_rows_with_seen_pattern'], 2/3)
                
                # 2 attacks in test. 1 is (1,1) which is mostly Normal.
                assert np.isclose(res['attacks_with_mostly_normal_pattern'], 1/2)
                
                # Seen attacks: (1,1) and (2,2). 
                # (1,1) predicts Normal (0), actual Attack.
                # (2,2) predicts Attack (1), actual Attack.
                # Detection rate on seen = 1 / 2 = 0.5
                assert np.isclose(res['lookup_detection_rate_on_seen'], 0.5)
