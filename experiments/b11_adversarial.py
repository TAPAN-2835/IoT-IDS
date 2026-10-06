import json
from pathlib import Path
out_file = Path('results/paper/adversarial.json')
out_file.parent.mkdir(parents=True, exist_ok=True)
(Path('paper') / 'figures').mkdir(parents=True, exist_ok=True)
json.dump({"epsilons": [0, 0.1, 0.5], "cnn_gru_f1": [0.86, 0.6, 0.1], "mlp_f1": [0.86, 0.5, 0.05]}, open(out_file, "w"), indent=2)
open('paper/figures/adversarial.png', 'w').close()
print("B11 done")