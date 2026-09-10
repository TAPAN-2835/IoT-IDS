# Edge-IIoTset Dataset

This folder is the designated location for the Edge-IIoTset dataset used in this project. 
To ensure reproducibility and manage large data dependencies safely, the dataset must be downloaded manually and placed here.

## Dataset Details
- **Official Source**: [Mendeley Data / Kaggle (mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot)]
- **Paper Reference**: Ferrag, M. A., Friha, O., Hamouda, D., Maglaras, L., & Janicke, H. (2022). Edge-IIoTset: A New Comprehensive Realistic Cyber Security Dataset of IoT and IIoT Applications for Centralized and Federated Learning. IEEE Access.
- **Where to download**: You can download it via [Kaggle](https://www.kaggle.com/datasets/mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot).

## Instructions

1. Download the dataset zip file from Kaggle or Mendeley Data.
2. Extract the CSV file(s) into this `data/raw/` directory.
3. Supported filename variants include:
   - `DNN-EdgeIIoT-dataset.csv` (PRIMARY TARGET FOR EXPERIMENTS)
   - `ML-EdgeIIoT-dataset.csv` (Only use if primary is unavailable or testing classical ML)
4. The project code will automatically discover any CSV files placed here and select `DNN-EdgeIIoT-dataset.csv` for the deep learning pipeline. 

## Dataset Metadata Registration

When placing the dataset here, the data loader pipeline will automatically record the dataset version, file size, SHA-256 hash, and row/column counts into the audit logs during the dataset audit phase.
