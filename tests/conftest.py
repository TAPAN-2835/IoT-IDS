import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from src.config import RAW_DATA_DIR

@pytest.fixture(scope="session")
def dummy_dataset_path(tmp_path_factory):
    """Creates a synthetic dummy dataset mimicking Edge-IIoTset schema for testing."""
    # We use tmp_path_factory to avoid writing to the real data/raw/ dir during tests
    # But for the full pipeline execution we'll drop it in raw/ if no data exists.
    # Wait, the prompt said: "Use a clearly separated synthetic/dummy test fixture ONLY for unit testing. Do NOT mix synthetic data with actual research results."
    
    test_dir = tmp_path_factory.mktemp("raw")
    filepath = test_dir / "dummy_edge_iiotset.csv"
    
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
    
    return filepath
