import kagglehub
import os
import shutil
from pathlib import Path

def main():
    print("Downloading dataset from Kaggle...")
    path = kagglehub.dataset_download("mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot")
    print("Path to dataset files:", path)
    
    target_dir = Path("data/raw")
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Locate DNN-EdgeIIoT-dataset.csv
    downloaded_path = Path(path)
    dnn_file = None
    
    for root, dirs, files in os.walk(downloaded_path):
        for file in files:
            if file == "DNN-EdgeIIoT-dataset.csv":
                dnn_file = Path(root) / file
                break
        if dnn_file:
            break
            
    if dnn_file:
        target_file = target_dir / "DNN-EdgeIIoT-dataset.csv"
        print(f"Moving {dnn_file} to {target_file}")
        shutil.copy2(dnn_file, target_file)
        print("Successfully copied dataset to data/raw/")
    else:
        print("ERROR: DNN-EdgeIIoT-dataset.csv not found in the downloaded files!")

if __name__ == "__main__":
    main()
