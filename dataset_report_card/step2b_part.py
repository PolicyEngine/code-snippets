"""Run a single check for dataset B, write partial JSON, then exit.

Usage:
    python step2b_part.py pt <index>
    python step2b_part.py rng <index>
    python step2b_part.py aca
    python step2b_part.py med
"""

import json
import sys
from common import (
    PATH_B, LABEL_B, CAL_DIR,
    POINT_CHECKS, RANGE_CHECKS, FALLBACK_CBO,
    compute_value, run_state_check,
)

part = sys.argv[1]

from policyengine_us import Microsimulation
sim = Microsimulation(dataset=PATH_B)

if part == "pt":
    idx = int(sys.argv[2])
    name, var, target_src, tol = POINT_CHECKS[idx]
    print(f"  pt[{idx}]: {name}")
    if isinstance(target_src, str) and target_src.startswith("cbo:"):
        key = target_src[4:]
        try:
            target = float(sim.tax_benefit_system.parameters(2024).calibration.gov.cbo._children[key])
        except Exception:
            target = FALLBACK_CBO[key]
    else:
        target = target_src
    val = compute_value(sim, var)
    if isinstance(val, str):
        result = {"name": name, "value": val, "target": target, "tol": tol, "pct_error": None}
    else:
        pct_err = abs(val - target) / target if target != 0 else 0
        result = {"name": name, "value": val, "target": target, "tol": tol, "pct_error": pct_err}
    with open(f"results_b_pt_{idx}.json", "w") as f:
        json.dump(result, f)

elif part == "rng":
    idx = int(sys.argv[2])
    name, var, lo, hi = RANGE_CHECKS[idx]
    print(f"  rng[{idx}]: {name}")
    val = compute_value(sim, var)
    if isinstance(val, str):
        result = {"name": name, "value": val, "lo": lo, "hi": hi, "passed": None}
    else:
        ok = (lo is None or val >= lo) and (hi is None or val <= hi)
        result = {"name": name, "value": val, "lo": lo, "hi": hi, "passed": ok}
    with open(f"results_b_rng_{idx}.json", "w") as f:
        json.dump(result, f)

elif part == "aca":
    print(f"ACA state check (2025) on {LABEL_B}...")
    result = run_state_check(
        sim, "aca_ptc", "household", 2025,
        CAL_DIR / "aca_spending_and_enrollment_2024.csv",
        "spending", 0.70, "ACA PTC by state",
    )
    with open("results_b_aca.json", "w") as f:
        json.dump(result, f)

elif part == "med":
    print(f"Medicaid state check (2025) on {LABEL_B}...")
    result = run_state_check(
        sim, "medicaid_enrolled", "household", 2025,
        CAL_DIR / "medicaid_enrollment_2024.csv",
        "enrollment", 0.45, "Medicaid enrollment",
    )
    with open("results_b_med.json", "w") as f:
        json.dump(result, f)
