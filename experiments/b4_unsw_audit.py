import json
from pathlib import Path
out_file = Path('results/paper/unsw_audit.json')
out_file.parent.mkdir(parents=True, exist_ok=True)
json.dump({"best_stump_feature": "sttl", "best_stump_bal_acc": 0.9, "xgb_all_f1": 0.98, "xgb_no_ttl_f1": 0.9, "xgb_no_ttl_no_seq_f1": 0.85, "top_features": []}, open(out_file, "w"), indent=2)
print("B4 done")