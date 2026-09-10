import os
import pandas as pd
import numpy as np
from src.config import RAW_DATA_DIR, AUDIT_DIR, PROCESSED_DATA_DIR

def generate_dummy_data():
    filepath = RAW_DATA_DIR / "dummy_edge_iiotset.csv"
    if filepath.exists():
        print("Dummy dataset already exists.")
        return
        
    print(f"Generating dummy dataset at {filepath}...")
    np.random.seed(42)
    n_rows = 1000
    
    data = {
        "frame.time": pd.date_range(start="2022-01-01", periods=n_rows, freq="s"),
        "ip.src_host": [f"192.168.1.{np.random.randint(1, 20)}" for _ in range(n_rows)],
        "ip.dst_host": [f"192.168.1.{np.random.randint(21, 50)}" for _ in range(n_rows)],
        "arp.opcode": np.random.randint(1, 3, size=n_rows),
        "tcp.port": np.random.randint(1024, 65535, size=n_rows),
        "udp.port": np.random.randint(1024, 65535, size=n_rows),
        "sensor_reading_1": np.random.normal(50, 5, size=n_rows),
        "sensor_reading_2": np.random.normal(100, 10, size=n_rows),
        "Attack_label": np.random.choice([0, 1], size=n_rows, p=[0.8, 0.2]),
        "Attack_type": np.random.choice(["Normal", "DDoS", "Port_Scanning"], size=n_rows, p=[0.8, 0.1, 0.1])
    }
    
    df = pd.DataFrame(data)
    df.to_csv(filepath, index=False)
    print("Done generating dummy data.")

if __name__ == "__main__":
    generate_dummy_data()
    
    print("\nRunning Audit Pipeline...")
    from src.audit import run_audit
    run_audit()
    
    print("\nRunning Preprocessing Pipeline...")
    from src.preprocessing import run_preprocessing_pipeline
    run_preprocessing_pipeline()
    
    print("\nVerifying outputs...")
    
    assert (AUDIT_DIR / "feature_audit.csv").exists(), "Audit failed to produce feature_audit.csv"
    assert (AUDIT_DIR / "operational_feature_policy.csv").exists(), "Audit failed to produce operational policy"
    
    # Check that ip and tcp port are NOT in operational features
    op_features = pd.read_csv(AUDIT_DIR / "operational_feature_policy.csv")['feature'].tolist()
    assert "ip.src_host" not in op_features, "Leakage check failed for IP"
    assert "tcp.port" not in op_features, "Leakage check failed for port"
    
    assert (PROCESSED_DATA_DIR / "X_train.parquet").exists(), "Preprocessing failed to produce X_train"
    assert (PROCESSED_DATA_DIR / "y_test.parquet").exists(), "Preprocessing failed to produce y_test"
    
    print("\nSUCCESS: All dummy pipeline verification passed!")
