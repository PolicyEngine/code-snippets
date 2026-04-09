"""Load dataset B, run point checks only (one per subprocess), write results_b.json."""

import json
import subprocess
import sys
from common import POINT_CHECKS, RANGE_CHECKS, LABEL_B

print(f"Loading {LABEL_B}...")

for i in range(len(POINT_CHECKS)):
    subprocess.run([sys.executable, "step2b_part.py", "pt", str(i)], check=True)

pt = []
for i in range(len(POINT_CHECKS)):
    with open(f"results_b_pt_{i}.json") as f:
        pt.append(json.load(f))

with open("results_b.json", "w") as f:
    json.dump({"cbo_targets": {}, "pt": pt, "rng": [], "aca": {}, "med": {}}, f)

print("Wrote results_b.json")
