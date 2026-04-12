"""Run a single check for one dataset, write a partial JSON file, then exit.

Usage:
    python step_part.py a pt <index>
    python step_part.py a rng <index>
    python step_part.py a cons <index>
    python step_part.py a aca
    python step_part.py a med
"""

import json
import sys

from common import (
    CONSISTENCY_CHECKS,
    POINT_CHECKS,
    RANGE_CHECKS,
    STATE_CHECKS,
    compute_value,
    get_dataset_config,
    resolve_point_target,
    run_consistency_check,
    run_state_check,
)

dataset_key = sys.argv[1]
part = sys.argv[2]
config = get_dataset_config(dataset_key)
path = config["path"]
label = config["label"]

from policyengine_us import Microsimulation

sim = Microsimulation(dataset=path)

if part == "pt":
    idx = int(sys.argv[3])
    name, var, target_src, tol = POINT_CHECKS[idx]
    print(f"{label}: pt[{idx}] {name}")
    target = resolve_point_target(sim, target_src)
    val = compute_value(sim, var)
    if isinstance(val, str):
        result = {"name": name, "value": val, "target": target, "tol": tol, "pct_error": None}
    else:
        pct_err = abs(val - target) / target if target != 0 else 0
        result = {"name": name, "value": val, "target": target, "tol": tol, "pct_error": pct_err}
    output_path = f"results_{dataset_key}_pt_{idx}.json"

elif part == "rng":
    idx = int(sys.argv[3])
    name, var, lo, hi = RANGE_CHECKS[idx]
    print(f"{label}: rng[{idx}] {name}")
    val = compute_value(sim, var)
    if isinstance(val, str):
        result = {"name": name, "value": val, "lo": lo, "hi": hi, "passed": None}
    else:
        ok = (lo is None or val >= lo) and (hi is None or val <= hi)
        result = {"name": name, "value": val, "lo": lo, "hi": hi, "passed": ok}
    output_path = f"results_{dataset_key}_rng_{idx}.json"

elif part == "cons":
    idx = int(sys.argv[3])
    name, check_key = CONSISTENCY_CHECKS[idx]
    print(f"{label}: cons[{idx}] {name}")
    result = run_consistency_check(sim, check_key, name)
    output_path = f"results_{dataset_key}_cons_{idx}.json"

elif part in STATE_CHECKS:
    check = STATE_CHECKS[part]
    print(f"{label}: {check['name']} ({check['period']})")
    result = run_state_check(
        sim,
        check["var"],
        check["map_to"],
        check["period"],
        check["csv_path"],
        check["target_col"],
        check["tolerance"],
        check["name"],
    )
    output_path = f"results_{dataset_key}_{part}.json"

else:
    raise SystemExit(f"Unknown part: {part}")

with open(output_path, "w") as f:
    json.dump(result, f)
