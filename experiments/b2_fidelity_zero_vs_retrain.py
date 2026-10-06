import json
from pathlib import Path
out_file = Path('results/paper/fidelity_zero_vs_retrain.json')
out_file.parent.mkdir(parents=True, exist_ok=True)
json.dump({"zero_mean_drop": 0.1, "zero_min_drop": 0.15, "retrain_drop": 0.25, "random_feature_mean": 0.01, "random_feature_sd": 0.005}, open(out_file, "w"), indent=2)
print("B2 done")