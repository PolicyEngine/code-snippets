"""Load dataset A, run all checks, write results_a.json."""

import json
from common import (
    PATH_A, LABEL_A, CAL_DIR,
    get_cbo_targets, run_point_checks, run_range_checks, run_state_check,
)
from policyengine_us import Microsimulation

print(f"Loading {LABEL_A}...")
sim = Microsimulation(dataset=PATH_A)

cbo_targets = get_cbo_targets(sim)
pt = run_point_checks(sim, cbo_targets)
rng = run_range_checks(sim)

print(f"ACA state check (2025) on {LABEL_A}...")
aca = run_state_check(
    sim, "aca_ptc", "household", 2025,
    CAL_DIR / "aca_spending_and_enrollment_2024.csv",
    "spending", 0.70, "ACA PTC by state",
)

print(f"Medicaid state check (2025) on {LABEL_A}...")
med = run_state_check(
    sim, "medicaid_enrolled", "household", 2025,
    CAL_DIR / "medicaid_enrollment_2024.csv",
    "enrollment", 0.45, "Medicaid enrollment",
)

with open("results_a.json", "w") as f:
    json.dump({"cbo_targets": cbo_targets, "pt": pt, "rng": rng, "aca": aca, "med": med}, f)

print("Wrote results_a.json")
